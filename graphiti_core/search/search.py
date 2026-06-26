"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timedelta
from math import exp
from time import time
from typing import Any

from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.driver.driver import GraphDriver
from graphiti_core.edges import EntityEdge
from graphiti_core.embedder.client import EMBEDDING_DIM
from graphiti_core.errors import SearchRerankerError
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.helpers import semaphore_gather, validate_group_ids
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodicNode
from graphiti_core.search.search_config import (
    DEFAULT_SEARCH_LIMIT,
    CommunityReranker,
    CommunitySearchConfig,
    CommunitySearchMethod,
    EdgeReranker,
    EdgeSearchConfig,
    EdgeSearchMethod,
    EpisodeReranker,
    EpisodeSearchConfig,
    NodeReranker,
    NodeSearchConfig,
    NodeSearchMethod,
    SearchConfig,
    SearchResults,
    TemporalDecayConfig,
    TemporalDecayFunction,
)
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import (
    community_fulltext_search,
    community_similarity_search,
    edge_bfs_search,
    edge_fulltext_search,
    edge_similarity_search,
    episode_fulltext_search,
    episode_mentions_reranker,
    get_embeddings_for_communities,
    get_embeddings_for_edges,
    get_embeddings_for_nodes,
    maximal_marginal_relevance,
    node_bfs_search,
    node_distance_reranker,
    node_fulltext_search,
    node_similarity_search,
    rrf,
)
from graphiti_core.tracer import NoOpTracer, Tracer
from graphiti_core.utils.datetime_utils import ensure_utc, utc_now

logger = logging.getLogger(__name__)


def _enum_value(value: Any) -> Any:
    return value.value if hasattr(value, 'value') else value


def _resolve_tracer(search_tracer: Tracer | None) -> Tracer:
    return search_tracer if search_tracer is not None else NoOpTracer()


def calculate_temporal_decay(days: float, config: TemporalDecayConfig) -> float:
    distance = max(0.0, days)
    scale = config.scale_days
    if config.function == TemporalDecayFunction.gaussian:
        return exp(-((distance**2) / (2 * (scale**2))))
    if config.function == TemporalDecayFunction.ebbinghaus:
        return exp(-(distance / scale))
    if config.function == TemporalDecayFunction.half_life:
        return 0.5 ** (distance / scale)
    raise ValueError(f'Unsupported temporal decay function: {config.function}')


def edge_temporal_distance_days(edge: EntityEdge, reference_time: datetime) -> float | None:
    seconds_per_day = timedelta(days=1).total_seconds()

    reference = ensure_utc(reference_time)
    if reference is None:
        return None

    # Prioritize valid_at date even if it is in the future
    valid_from = (
        ensure_utc(getattr(edge, 'valid_at', None)) 
        or ensure_utc(getattr(edge, 'created_at', None))
    )

    valid_until_candidates = [
        value
        for value in [
            ensure_utc(getattr(edge, 'invalid_at', None)),
            ensure_utc(getattr(edge, 'expired_at', None)),
        ]
        if value
    ]
    valid_until = min(valid_until_candidates) if valid_until_candidates else None

    if valid_from is not None and valid_until is not None:
        if valid_from <= reference <= valid_until:
            return 0.0
        return min(
            abs((reference - valid_from).total_seconds()),
            abs((reference - valid_until).total_seconds()),
        ) / seconds_per_day

    # If only one of valid_from or valid_until is present, calculate the distance to that boundary
    if valid_from is not None:
        boundary = valid_from
    elif valid_until is not None:
        boundary = valid_until
    else:
        return None
    
    return abs((reference - boundary).total_seconds()) / seconds_per_day


def edge_temporal_score(edge: EntityEdge, config: TemporalDecayConfig) -> float:
    reference_time = config.reference_time or utc_now()
    distance = edge_temporal_distance_days(edge, reference_time)
    if distance is None:
        logger.warning(f"Edge {edge.uuid} has no valid temporal boundaries. There must always exist created_at attribute!")
        return 0.0
    return calculate_temporal_decay(distance, config)


def normalize_scores(scores: list[float], target_count: int) -> list[float]:
    padded_scores = [
        float(scores[index]) if index < len(scores) else 0.0 for index in range(target_count)
    ]
    if not padded_scores:
        return []

    min_score = min(padded_scores)
    max_score = max(padded_scores)
    if max_score == min_score:
        return [1.0] * len(padded_scores)

    score_range = max_score - min_score
    return [(score - min_score) / score_range for score in padded_scores]


def apply_temporal_decay(
    edges: list[EntityEdge],
    base_scores: list[float],
    config: TemporalDecayConfig,
) -> tuple[list[EntityEdge], list[float]]:
    normalized_base_scores = normalize_scores(base_scores, len(edges))
    weighted_edges: list[tuple[EntityEdge, float]] = []

    for edge, base_score in zip(edges, normalized_base_scores, strict=True):
        temporal_score = edge_temporal_score(edge, config)
        final_score = (1 - config.temporal_weight) * base_score + config.temporal_weight * temporal_score
        weighted_edges.append((edge, final_score))

    weighted_edges.sort(key=lambda item: item[1], reverse=True)
    return [edge for edge, _ in weighted_edges], [score for _, score in weighted_edges]


@contextmanager
def _trace_phase(
    search_tracer: Tracer,
    name: str,
    attributes: dict[str, Any] | None = None,
) -> Generator[Any, None, None]:
    with search_tracer.start_span(name) as span:
        if attributes:
            span.add_attributes(attributes)
        try:
            yield span
            span.set_status('ok')
        except Exception as e:
            span.set_status('error', str(e))
            span.record_exception(e)
            raise


async def search(
    clients: GraphitiClients,
    query: str,
    group_ids: list[str] | None,
    config: SearchConfig,
    search_filter: SearchFilters,
    center_node_uuid: str | None = None,
    bfs_origin_node_uuids: list[str] | None = None,
    query_vector: list[float] | None = None,
    driver: GraphDriver | None = None,
) -> SearchResults:
    start = time()
    validate_group_ids(group_ids)

    driver = driver or clients.driver
    embedder = clients.embedder
    cross_encoder = clients.cross_encoder
    search_tracer = _resolve_tracer(getattr(clients, 'tracer', None))

    if query.strip() == '':
        return SearchResults()

    if (
        (
            config.edge_config
            and EdgeSearchMethod.cosine_similarity in config.edge_config.search_methods
        )
        or (config.edge_config and EdgeReranker.mmr == config.edge_config.reranker)
        or (
            config.node_config
            and NodeSearchMethod.cosine_similarity in config.node_config.search_methods
        )
        or (config.node_config and NodeReranker.mmr == config.node_config.reranker)
        or (
            config.community_config
            and CommunitySearchMethod.cosine_similarity in config.community_config.search_methods
        )
        or (config.community_config and CommunityReranker.mmr == config.community_config.reranker)
    ):
        with _trace_phase(
            search_tracer,
            'search.embed_query_vector',
            {
                'query.length': len(query),
                'query_vector.provided': query_vector is not None,
            },
        ) as span:
            search_vector = (
                query_vector
                if query_vector is not None
                else await embedder.create(input_data=[query.replace('\n', ' ')])
            )
            span.add_attributes({'query_vector.dimension': len(search_vector)})
    else:
        search_vector = [0.0] * EMBEDDING_DIM

    # if group_ids is empty, set it to None
    group_ids = group_ids if group_ids and group_ids != [''] else None
    with _trace_phase(
        search_tracer,
        'search.execute_scopes',
        {
            'group_id.count': len(group_ids or []),
            'scope.edges': config.edge_config is not None,
            'scope.nodes': config.node_config is not None,
            'scope.episodes': config.episode_config is not None,
            'scope.communities': config.community_config is not None,
            'limit': config.limit,
        },
    ) as span:
        (
            (edges, edge_reranker_scores),
            (nodes, node_reranker_scores),
            (episodes, episode_reranker_scores),
            (communities, community_reranker_scores),
        ) = await semaphore_gather(
            edge_search(
                driver,
                cross_encoder,
                query,
                search_vector,
                group_ids,
                config.edge_config,
                search_filter,
                config.temporal_decay_config,
                center_node_uuid,
                bfs_origin_node_uuids,
                config.limit,
                config.reranker_min_score,
                search_tracer,
            ),
            node_search(
                driver,
                cross_encoder,
                query,
                search_vector,
                group_ids,
                config.node_config,
                search_filter,
                center_node_uuid,
                bfs_origin_node_uuids,
                config.limit,
                config.reranker_min_score,
                search_tracer,
            ),
            episode_search(
                driver,
                cross_encoder,
                query,
                search_vector,
                group_ids,
                config.episode_config,
                search_filter,
                config.limit,
                config.reranker_min_score,
                search_tracer,
            ),
            community_search(
                driver,
                cross_encoder,
                query,
                search_vector,
                group_ids,
                config.community_config,
                config.limit,
                config.reranker_min_score,
                search_tracer,
            ),
        )
        span.add_attributes(
            {
                'result.edges': len(edges),
                'result.nodes': len(nodes),
                'result.episodes': len(episodes),
                'result.communities': len(communities),
            }
        )

    results = SearchResults(
        edges=edges,
        edge_reranker_scores=edge_reranker_scores,
        nodes=nodes,
        node_reranker_scores=node_reranker_scores,
        episodes=episodes,
        episode_reranker_scores=episode_reranker_scores,
        communities=communities,
        community_reranker_scores=community_reranker_scores,
    )

    latency = (time() - start) * 1000

    logger.debug(f'search returned context in {latency} ms')

    return results


async def edge_search(
    driver: GraphDriver,
    cross_encoder: CrossEncoderClient,
    query: str,
    query_vector: list[float],
    group_ids: list[str] | None,
    config: EdgeSearchConfig | None,
    search_filter: SearchFilters,
    temporal_decay_config: TemporalDecayConfig | None = None,
    center_node_uuid: str | None = None,
    bfs_origin_node_uuids: list[str] | None = None,
    limit=DEFAULT_SEARCH_LIMIT,
    reranker_min_score: float = 0,
    search_tracer: Tracer | None = None,
) -> tuple[list[EntityEdge], list[float]]:
    if config is None:
        return [], []
    search_tracer = _resolve_tracer(search_tracer)

    with _trace_phase(
        search_tracer,
        'search.edge_search',
        {
            'limit': limit,
            'reranker': _enum_value(config.reranker),
            'search_methods': [_enum_value(method) for method in config.search_methods],
            'bfs_origin_count': len(bfs_origin_node_uuids or []),
            'center_node_uuid.provided': center_node_uuid is not None,
        },
    ) as span:
        # Build search tasks based on configured search methods
        search_tasks = []
        if EdgeSearchMethod.bm25 in config.search_methods:
            search_tasks.append(
                edge_fulltext_search(driver, query, search_filter, group_ids, 2 * limit)
            )
        if EdgeSearchMethod.cosine_similarity in config.search_methods:
            search_tasks.append(
                edge_similarity_search(
                    driver,
                    query_vector,
                    None,
                    None,
                    search_filter,
                    group_ids,
                    2 * limit,
                    config.sim_min_score,
                )
            )
        if EdgeSearchMethod.bfs in config.search_methods:
            search_tasks.append(
                edge_bfs_search(
                    driver,
                    bfs_origin_node_uuids,
                    config.bfs_max_depth,
                    search_filter,
                    group_ids,
                    2 * limit,
                )
            )

        # Execute only the configured search methods
        search_results: list[list[EntityEdge]] = []
        if search_tasks:
            with _trace_phase(
                search_tracer,
                'search.edge_search.execute_methods',
                {
                    'method_count': len(search_tasks),
                    'candidate_limit': 2 * limit,
                },
            ) as method_span:
                search_results = list(await semaphore_gather(*search_tasks))
                method_span.add_attributes(
                    {
                        'result_set_count': len(search_results),
                        'non_empty_result_sets': sum(1 for result in search_results if result),
                    }
                )

        if EdgeSearchMethod.bfs in config.search_methods and bfs_origin_node_uuids is None:
            source_node_uuids = [
                edge.source_node_uuid for result in search_results for edge in result
            ]
            with _trace_phase(
                search_tracer,
                'search.edge_search.expand_bfs',
                {
                    'origin_node_count': len(source_node_uuids),
                    'candidate_limit': 2 * limit,
                },
            ):
                search_results.append(
                    await edge_bfs_search(
                        driver,
                        source_node_uuids,
                        config.bfs_max_depth,
                        search_filter,
                        group_ids,
                        2 * limit,
                    )
                )

        edge_uuid_map = {edge.uuid: edge for result in search_results for edge in result}

        reranked_uuids: list[str] = []
        edge_scores: list[float] = []
        with _trace_phase(
            search_tracer,
            'search.edge_search.rerank',
            {
                'candidate_count': len(edge_uuid_map),
                'result_set_count': len(search_results),
                'reranker': _enum_value(config.reranker),
            },
        ):
            if (
                config.reranker == EdgeReranker.rrf
                or config.reranker == EdgeReranker.episode_mentions
            ):
                search_result_uuids = [[edge.uuid for edge in result] for result in search_results]

                reranked_uuids, edge_scores = rrf(search_result_uuids, min_score=reranker_min_score)
            elif config.reranker == EdgeReranker.mmr:
                with _trace_phase(
                    search_tracer,
                    'search.edge_search.load_embeddings',
                    {'candidate_count': len(edge_uuid_map)},
                ):
                    search_result_uuids_and_vectors = await get_embeddings_for_edges(
                        driver, list(edge_uuid_map.values())
                    )
                with _trace_phase(
                    search_tracer,
                    'search.edge_search.compute_mmr',
                    {'candidate_count': len(search_result_uuids_and_vectors)},
                ):
                    reranked_uuids, edge_scores = maximal_marginal_relevance(
                        query_vector,
                        search_result_uuids_and_vectors,
                        config.mmr_lambda,
                        reranker_min_score,
                    )
            elif config.reranker == EdgeReranker.cross_encoder:
                fact_to_uuid_map = {
                    edge.fact: edge.uuid for edge in list(edge_uuid_map.values())[:limit]
                }
                with _trace_phase(
                    search_tracer,
                    'search.edge_search.cross_encoder_rank',
                    {'candidate_count': len(fact_to_uuid_map)},
                ):
                    reranked_facts = await cross_encoder.rank(query, list(fact_to_uuid_map.keys()))
                reranked_uuids = [
                    fact_to_uuid_map[fact]
                    for fact, score in reranked_facts
                    if score >= reranker_min_score
                ]
                edge_scores = [score for _, score in reranked_facts if score >= reranker_min_score]
            elif config.reranker == EdgeReranker.node_distance:
                if center_node_uuid is None:
                    raise SearchRerankerError('No center node provided for Node Distance reranker')

                with _trace_phase(
                    search_tracer,
                    'search.edge_search.seed_rrf',
                    {'result_set_count': len(search_results)},
                ):
                    sorted_result_uuids, _ = rrf(
                        [[edge.uuid for edge in result] for result in search_results],
                        min_score=reranker_min_score,
                    )
                sorted_results = [edge_uuid_map[uuid] for uuid in sorted_result_uuids]

                source_to_edge_uuid_map = defaultdict(list)
                for edge in sorted_results:
                    source_to_edge_uuid_map[edge.source_node_uuid].append(edge.uuid)

                source_uuids = [source_node_uuid for source_node_uuid in source_to_edge_uuid_map]

                with _trace_phase(
                    search_tracer,
                    'search.edge_search.node_distance_rank',
                    {
                        'source_node_count': len(source_uuids),
                        'center_node_uuid.provided': center_node_uuid is not None,
                    },
                ):
                    reranked_node_uuids, edge_scores = await node_distance_reranker(
                        driver, source_uuids, center_node_uuid, min_score=reranker_min_score
                    )

                for node_uuid in reranked_node_uuids:
                    reranked_uuids.extend(source_to_edge_uuid_map[node_uuid])

        reranked_edges = [edge_uuid_map[uuid] for uuid in reranked_uuids]

        if config.reranker == EdgeReranker.episode_mentions:
            reranked_edges.sort(reverse=True, key=lambda edge: len(edge.episodes))

            if temporal_decay_config is not None:
                edge_scores = [float(len(edge.episodes)) for edge in reranked_edges]
            else:
                max_mentions = max((len(edge.episodes) for edge in reranked_edges), default=1)
                edge_scores = [len(edge.episodes) / max_mentions for edge in reranked_edges]

        if temporal_decay_config is not None:
            with _trace_phase(
                search_tracer,
                'search.edge_search.temporal_decay',
                {
                    'candidate_count': len(reranked_edges),
                    'temporal_decay.function': temporal_decay_config.function.value,
                    'temporal_decay.scale_days': temporal_decay_config.scale_days,
                    'temporal_decay.weight': temporal_decay_config.temporal_weight,
                    'temporal_decay.reference_time.provided': (
                        temporal_decay_config.reference_time is not None
                    ),
                },
            ):
                reranked_edges, edge_scores = apply_temporal_decay(
                    reranked_edges,
                    edge_scores,
                    temporal_decay_config,
                )

        span.add_attributes(
            {
                'candidate_count': len(edge_uuid_map),
                'reranked_count': len(reranked_edges),
                'returned_count': min(len(reranked_edges), limit),
            }
        )

        return reranked_edges[:limit], edge_scores[:limit]


async def node_search(
    driver: GraphDriver,
    cross_encoder: CrossEncoderClient,
    query: str,
    query_vector: list[float],
    group_ids: list[str] | None,
    config: NodeSearchConfig | None,
    search_filter: SearchFilters,
    center_node_uuid: str | None = None,
    bfs_origin_node_uuids: list[str] | None = None,
    limit=DEFAULT_SEARCH_LIMIT,
    reranker_min_score: float = 0,
    search_tracer: Tracer | None = None,
) -> tuple[list[EntityNode], list[float]]:
    if config is None:
        return [], []
    search_tracer = _resolve_tracer(search_tracer)

    with _trace_phase(
        search_tracer,
        'search.node_search',
        {
            'limit': limit,
            'reranker': _enum_value(config.reranker),
            'search_methods': [_enum_value(method) for method in config.search_methods],
            'bfs_origin_count': len(bfs_origin_node_uuids or []),
            'center_node_uuid.provided': center_node_uuid is not None,
        },
    ) as span:
        # Build search tasks based on configured search methods
        search_tasks = []
        if NodeSearchMethod.bm25 in config.search_methods:
            search_tasks.append(
                node_fulltext_search(driver, query, search_filter, group_ids, 2 * limit)
            )
        if NodeSearchMethod.cosine_similarity in config.search_methods:
            search_tasks.append(
                node_similarity_search(
                    driver,
                    query_vector,
                    search_filter,
                    group_ids,
                    2 * limit,
                    config.sim_min_score,
                )
            )
        if NodeSearchMethod.bfs in config.search_methods:
            search_tasks.append(
                node_bfs_search(
                    driver,
                    bfs_origin_node_uuids,
                    search_filter,
                    config.bfs_max_depth,
                    group_ids,
                    2 * limit,
                )
            )

        # Execute only the configured search methods
        search_results: list[list[EntityNode]] = []
        if search_tasks:
            with _trace_phase(
                search_tracer,
                'search.node_search.execute_methods',
                {
                    'method_count': len(search_tasks),
                    'candidate_limit': 2 * limit,
                },
            ) as method_span:
                search_results = list(await semaphore_gather(*search_tasks))
                method_span.add_attributes(
                    {
                        'result_set_count': len(search_results),
                        'non_empty_result_sets': sum(1 for result in search_results if result),
                    }
                )

        if NodeSearchMethod.bfs in config.search_methods and bfs_origin_node_uuids is None:
            origin_node_uuids = [node.uuid for result in search_results for node in result]
            with _trace_phase(
                search_tracer,
                'search.node_search.expand_bfs',
                {
                    'origin_node_count': len(origin_node_uuids),
                    'candidate_limit': 2 * limit,
                },
            ):
                search_results.append(
                    await node_bfs_search(
                        driver,
                        origin_node_uuids,
                        search_filter,
                        config.bfs_max_depth,
                        group_ids,
                        2 * limit,
                    )
                )

        search_result_uuids = [[node.uuid for node in result] for result in search_results]
        node_uuid_map = {node.uuid: node for result in search_results for node in result}

        reranked_uuids: list[str] = []
        node_scores: list[float] = []
        with _trace_phase(
            search_tracer,
            'search.node_search.rerank',
            {
                'candidate_count': len(node_uuid_map),
                'result_set_count': len(search_results),
                'reranker': _enum_value(config.reranker),
            },
        ):
            if config.reranker == NodeReranker.rrf:
                reranked_uuids, node_scores = rrf(search_result_uuids, min_score=reranker_min_score)
            elif config.reranker == NodeReranker.mmr:
                with _trace_phase(
                    search_tracer,
                    'search.node_search.load_embeddings',
                    {'candidate_count': len(node_uuid_map)},
                ):
                    search_result_uuids_and_vectors = await get_embeddings_for_nodes(
                        driver, list(node_uuid_map.values())
                    )

                with _trace_phase(
                    search_tracer,
                    'search.node_search.compute_mmr',
                    {'candidate_count': len(search_result_uuids_and_vectors)},
                ):
                    reranked_uuids, node_scores = maximal_marginal_relevance(
                        query_vector,
                        search_result_uuids_and_vectors,
                        config.mmr_lambda,
                        reranker_min_score,
                    )
            elif config.reranker == NodeReranker.cross_encoder:
                name_to_uuid_map = {node.name: node.uuid for node in list(node_uuid_map.values())}

                with _trace_phase(
                    search_tracer,
                    'search.node_search.cross_encoder_rank',
                    {'candidate_count': len(name_to_uuid_map)},
                ):
                    reranked_node_names = await cross_encoder.rank(
                        query, list(name_to_uuid_map.keys())
                    )
                reranked_uuids = [
                    name_to_uuid_map[name]
                    for name, score in reranked_node_names
                    if score >= reranker_min_score
                ]
                node_scores = [
                    score for _, score in reranked_node_names if score >= reranker_min_score
                ]
            elif config.reranker == NodeReranker.episode_mentions:
                with _trace_phase(
                    search_tracer,
                    'search.node_search.episode_mentions_rank',
                    {'candidate_count': len(node_uuid_map)},
                ):
                    reranked_uuids, node_scores = await episode_mentions_reranker(
                        driver, search_result_uuids, min_score=reranker_min_score
                    )
            elif config.reranker == NodeReranker.node_distance:
                if center_node_uuid is None:
                    raise SearchRerankerError('No center node provided for Node Distance reranker')
                with _trace_phase(
                    search_tracer,
                    'search.node_search.seed_rrf',
                    {'result_set_count': len(search_results)},
                ):
                    seeded_uuids = rrf(search_result_uuids, min_score=reranker_min_score)[0]
                with _trace_phase(
                    search_tracer,
                    'search.node_search.node_distance_rank',
                    {
                        'source_node_count': len(seeded_uuids),
                        'center_node_uuid.provided': center_node_uuid is not None,
                    },
                ):
                    reranked_uuids, node_scores = await node_distance_reranker(
                        driver,
                        seeded_uuids,
                        center_node_uuid,
                        min_score=reranker_min_score,
                    )

        reranked_nodes = [node_uuid_map[uuid] for uuid in reranked_uuids]

        span.add_attributes(
            {
                'candidate_count': len(node_uuid_map),
                'reranked_count': len(reranked_nodes),
                'returned_count': min(len(reranked_nodes), limit),
            }
        )

        return reranked_nodes[:limit], node_scores[:limit]


async def episode_search(
    driver: GraphDriver,
    cross_encoder: CrossEncoderClient,
    query: str,
    _query_vector: list[float],
    group_ids: list[str] | None,
    config: EpisodeSearchConfig | None,
    search_filter: SearchFilters,
    limit=DEFAULT_SEARCH_LIMIT,
    reranker_min_score: float = 0,
    search_tracer: Tracer | None = None,
) -> tuple[list[EpisodicNode], list[float]]:
    if config is None:
        return [], []
    search_tracer = _resolve_tracer(search_tracer)

    with _trace_phase(
        search_tracer,
        'search.episode_search',
        {
            'limit': limit,
            'reranker': _enum_value(config.reranker),
            'search_methods': [_enum_value(method) for method in config.search_methods],
        },
    ) as span:
        with _trace_phase(
            search_tracer,
            'search.episode_search.execute_methods',
            {'candidate_limit': 2 * limit},
        ):
            search_results: list[list[EpisodicNode]] = list(
                await semaphore_gather(
                    *[
                        episode_fulltext_search(driver, query, search_filter, group_ids, 2 * limit),
                    ]
                )
            )

        search_result_uuids = [[episode.uuid for episode in result] for result in search_results]
        episode_uuid_map = {
            episode.uuid: episode for result in search_results for episode in result
        }

        reranked_uuids: list[str] = []
        episode_scores: list[float] = []
        with _trace_phase(
            search_tracer,
            'search.episode_search.rerank',
            {
                'candidate_count': len(episode_uuid_map),
                'result_set_count': len(search_results),
                'reranker': _enum_value(config.reranker),
            },
        ):
            if config.reranker == EpisodeReranker.rrf:
                reranked_uuids, episode_scores = rrf(
                    search_result_uuids, min_score=reranker_min_score
                )
            elif config.reranker == EpisodeReranker.cross_encoder:
                with _trace_phase(
                    search_tracer,
                    'search.episode_search.seed_rrf',
                    {'result_set_count': len(search_results)},
                ):
                    rrf_result_uuids, episode_scores = rrf(
                        search_result_uuids, min_score=reranker_min_score
                    )
                rrf_results = [episode_uuid_map[uuid] for uuid in rrf_result_uuids][:limit]

                content_to_uuid_map = {episode.content: episode.uuid for episode in rrf_results}

                with _trace_phase(
                    search_tracer,
                    'search.episode_search.cross_encoder_rank',
                    {'candidate_count': len(content_to_uuid_map)},
                ):
                    reranked_contents = await cross_encoder.rank(
                        query, list(content_to_uuid_map.keys())
                    )
                reranked_uuids = [
                    content_to_uuid_map[content]
                    for content, score in reranked_contents
                    if score >= reranker_min_score
                ]
                episode_scores = [
                    score for _, score in reranked_contents if score >= reranker_min_score
                ]

        reranked_episodes = [episode_uuid_map[uuid] for uuid in reranked_uuids]
        span.add_attributes(
            {
                'candidate_count': len(episode_uuid_map),
                'reranked_count': len(reranked_episodes),
                'returned_count': min(len(reranked_episodes), limit),
            }
        )

        return reranked_episodes[:limit], episode_scores[:limit]


async def community_search(
    driver: GraphDriver,
    cross_encoder: CrossEncoderClient,
    query: str,
    query_vector: list[float],
    group_ids: list[str] | None,
    config: CommunitySearchConfig | None,
    limit=DEFAULT_SEARCH_LIMIT,
    reranker_min_score: float = 0,
    search_tracer: Tracer | None = None,
) -> tuple[list[CommunityNode], list[float]]:
    if config is None:
        return [], []
    search_tracer = _resolve_tracer(search_tracer)

    with _trace_phase(
        search_tracer,
        'search.community_search',
        {
            'limit': limit,
            'reranker': _enum_value(config.reranker),
            'search_methods': [_enum_value(method) for method in config.search_methods],
        },
    ) as span:
        with _trace_phase(
            search_tracer,
            'search.community_search.execute_methods',
            {'candidate_limit': 2 * limit},
        ):
            search_results: list[list[CommunityNode]] = list(
                await semaphore_gather(
                    *[
                        community_fulltext_search(driver, query, group_ids, 2 * limit),
                        community_similarity_search(
                            driver, query_vector, group_ids, 2 * limit, config.sim_min_score
                        ),
                    ]
                )
            )

        search_result_uuids = [
            [community.uuid for community in result] for result in search_results
        ]
        community_uuid_map = {
            community.uuid: community for result in search_results for community in result
        }

        reranked_uuids: list[str] = []
        community_scores: list[float] = []
        with _trace_phase(
            search_tracer,
            'search.community_search.rerank',
            {
                'candidate_count': len(community_uuid_map),
                'result_set_count': len(search_results),
                'reranker': _enum_value(config.reranker),
            },
        ):
            if config.reranker == CommunityReranker.rrf:
                reranked_uuids, community_scores = rrf(
                    search_result_uuids, min_score=reranker_min_score
                )
            elif config.reranker == CommunityReranker.mmr:
                with _trace_phase(
                    search_tracer,
                    'search.community_search.load_embeddings',
                    {'candidate_count': len(community_uuid_map)},
                ):
                    search_result_uuids_and_vectors = await get_embeddings_for_communities(
                        driver, list(community_uuid_map.values())
                    )

                with _trace_phase(
                    search_tracer,
                    'search.community_search.compute_mmr',
                    {'candidate_count': len(search_result_uuids_and_vectors)},
                ):
                    reranked_uuids, community_scores = maximal_marginal_relevance(
                        query_vector,
                        search_result_uuids_and_vectors,
                        config.mmr_lambda,
                        reranker_min_score,
                    )
            elif config.reranker == CommunityReranker.cross_encoder:
                name_to_uuid_map = {
                    node.name: node.uuid for result in search_results for node in result
                }
                with _trace_phase(
                    search_tracer,
                    'search.community_search.cross_encoder_rank',
                    {'candidate_count': len(name_to_uuid_map)},
                ):
                    reranked_nodes = await cross_encoder.rank(query, list(name_to_uuid_map.keys()))
                reranked_uuids = [
                    name_to_uuid_map[name]
                    for name, score in reranked_nodes
                    if score >= reranker_min_score
                ]
                community_scores = [
                    score for _, score in reranked_nodes if score >= reranker_min_score
                ]

        reranked_communities = [community_uuid_map[uuid] for uuid in reranked_uuids]
        span.add_attributes(
            {
                'candidate_count': len(community_uuid_map),
                'reranked_count': len(reranked_communities),
                'returned_count': min(len(reranked_communities), limit),
            }
        )

        return reranked_communities[:limit], community_scores[:limit]
