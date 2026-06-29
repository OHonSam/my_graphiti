"""
Turn-gold LongMemEval reranker evaluator.

This runner mirrors the LongMemEval baseline ingestion/search flow, but scores
retrieval against sessions that contain at least one turn with ``has_answer:
true``. It reports evidence coverage metrics at K = 3, 5, and 10.

Examples:
    python tests/evals/data/longmemeval_data/reranker_eval.py sample
    python tests/evals/data/longmemeval_data/reranker_eval.py sample --main-only
    python tests/evals/data/longmemeval_data/reranker_eval.py ingest --reset-group
    uv run python tests/evals/data/longmemeval_data/reranker_eval.py evaluate
    uv run python tests/evals/data/longmemeval_data/reranker_eval.py run --main-only --reset-group
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import NAMESPACE_URL, uuid5

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUTPUT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = OUTPUT_DIR / 'longmemeval_oracle.json'
DEFAULT_RESULTS = OUTPUT_DIR / 'reranker_eval_results_v1.jsonl'
DEFAULT_SUMMARY = OUTPUT_DIR / 'reranker_eval_summary_v1.json'
DEFAULT_SAMPLE_SIZE = 10
DEFAULT_SEED = 13
DEFAULT_DISTRACTORS_PER_QUESTION = 3
DEFAULT_QUESTION_TYPES = ['temporal-reasoning', 'knowledge-update']
DEFAULT_TEMPORAL_DECAYS = ['none', 'gaussian', 'ebbinghaus', 'half_life']
TEMPORAL_DECAY_CHOICES = ['none', 'gaussian', 'ebbinghaus', 'half_life']
METRIC_KS = (3, 5, 10)
RERANKER_NAMES = ['edge_rrf', 'edge_mmr', 'edge_episode_mentions', 'edge_cross_encoder']


@dataclass(frozen=True)
class LongMemEvalQuestion:
    question_id: str
    question_type: str
    question: str
    answer: str
    question_time: datetime
    haystack_session_ids: list[str]
    haystack_dates: list[datetime]
    haystack_sessions: list[list[dict[str, Any]]]
    answer_session_ids: list[str]


@dataclass(frozen=True)
class EpisodeCase:
    eval_question_id: str
    source_question_id: str
    question_type: str
    session_id: str
    episode_uuid: str
    episode_name: str
    reference_time: datetime
    body: str
    is_answer_session: bool
    is_distractor: bool
    date_delta_seconds: float | None = None


def parse_lme_datetime(value: str) -> datetime:
    dt = datetime.strptime(value.strip(), '%Y/%m/%d (%a) %H:%M')
    return dt.replace(tzinfo=timezone.utc)


def stable_episode_uuid(question_id: str, session_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, f'graphiti:longmemeval:{question_id}:{session_id}'))


def is_abstention_question(question_id: str) -> bool:
    return question_id.endswith('_abs')


def format_session_transcript(session: list[dict[str, Any]]) -> str:
    return '\n'.join(
        f'{turn.get("role", "unknown")}: {turn.get("content", "").strip()}'
        for turn in session
        if turn.get('content', '').strip()
    )


def load_questions(path: Path) -> list[LongMemEvalQuestion]:
    raw_items = json.loads(path.read_text(encoding='utf-8'))
    questions: list[LongMemEvalQuestion] = []

    for item in raw_items:
        question_id = item['question_id']
        haystack_session_ids = item['haystack_session_ids']
        haystack_dates = [parse_lme_datetime(value) for value in item['haystack_dates']]
        haystack_sessions = item['haystack_sessions']

        if not (
            len(haystack_session_ids) == len(haystack_dates) == len(haystack_sessions)
        ):
            raise ValueError(f'{question_id}: haystack ids, dates, and sessions differ in length')

        questions.append(
            LongMemEvalQuestion(
                question_id=question_id,
                question_type=item['question_type'],
                question=item['question'],
                answer=item.get('answer', ''),
                question_time=parse_lme_datetime(item['question_date']),
                haystack_session_ids=haystack_session_ids,
                haystack_dates=haystack_dates,
                haystack_sessions=haystack_sessions,
                answer_session_ids=item.get('answer_session_ids', []),
            )
        )

    return questions


def has_answer_turn(session: list[dict[str, Any]]) -> bool:
    return any(turn.get('has_answer') is True for turn in session)


def gold_session_ids(question: LongMemEvalQuestion) -> list[str]:
    return [
        session_id
        for session_id, session in zip(
            question.haystack_session_ids,
            question.haystack_sessions,
            strict=True,
        )
        if has_answer_turn(session)
    ]


def select_questions(
    questions: Sequence[LongMemEvalQuestion],
    sample_size: int,
    seed: int,
    question_types: Sequence[str],
) -> list[LongMemEvalQuestion]:
    eligible = [
        question
        for question in questions
        if question.question_type in question_types
        and not is_abstention_question(question.question_id)
        and gold_session_ids(question)
    ]
    by_type: dict[str, list[LongMemEvalQuestion]] = defaultdict(list)
    for question in eligible:
        by_type[question.question_type].append(question)

    rng = random.Random(seed)
    selected: list[LongMemEvalQuestion] = []
    selected_ids: set[str] = set()
    type_count = max(1, len(question_types))
    base_count = sample_size // type_count
    remainder = sample_size % type_count

    for index, question_type in enumerate(question_types):
        target_count = base_count + (1 if index < remainder else 0)
        candidates = sorted(by_type.get(question_type, []), key=lambda q: q.question_id)
        rng.shuffle(candidates)
        for question in candidates[:target_count]:
            selected.append(question)
            selected_ids.add(question.question_id)

    if len(selected) < sample_size:
        remaining = [
            question
            for question in sorted(eligible, key=lambda q: q.question_id)
            if question.question_id not in selected_ids
        ]
        rng.shuffle(remaining)
        selected.extend(remaining[: sample_size - len(selected)])

    return selected[:sample_size]


def load_all_and_sampled_questions(
    args: argparse.Namespace,
) -> tuple[list[LongMemEvalQuestion], list[LongMemEvalQuestion]]:
    questions = load_questions(Path(args.dataset))
    sampled = select_questions(questions, args.sample_size, args.seed, args.question_types)
    return questions, sampled


def nearest_date_delta_seconds(reference_time: datetime, target_dates: Sequence[datetime]) -> float:
    if not target_dates:
        return float('inf')
    return min(abs((reference_time - target).total_seconds()) for target in target_dates)


def select_distractor_episode_cases(
    all_questions: Sequence[LongMemEvalQuestion],
    selected_questions: Sequence[LongMemEvalQuestion],
    distractors_per_question: int,
) -> dict[str, list[EpisodeCase]]:
    if distractors_per_question < 0:
        raise ValueError('distractors_per_question must be non-negative')
    if distractors_per_question == 0:
        return {question.question_id: [] for question in selected_questions}

    selected_question_ids = {question.question_id for question in selected_questions}
    used_session_ids = {
        session_id
        for question in selected_questions
        for session_id in question.haystack_session_ids
    }
    distractors_by_question: dict[str, list[EpisodeCase]] = {}

    for eval_question in selected_questions:
        blocked_session_ids = set(eval_question.haystack_session_ids) | set(
            eval_question.answer_session_ids
        )
        candidates: list[tuple[float, str, str, EpisodeCase]] = []

        for source_question in all_questions:
            if source_question.question_type != eval_question.question_type:
                continue
            if source_question.question_id in selected_question_ids:
                continue
            if is_abstention_question(source_question.question_id):
                continue

            for session_id, reference_time, session in zip(
                source_question.haystack_session_ids,
                source_question.haystack_dates,
                source_question.haystack_sessions,
                strict=True,
            ):
                if session_id in blocked_session_ids or session_id in used_session_ids:
                    continue

                date_delta = nearest_date_delta_seconds(
                    reference_time, eval_question.haystack_dates
                )
                episode_uuid = stable_episode_uuid(source_question.question_id, session_id)
                candidates.append(
                    (
                        date_delta,
                        source_question.question_id,
                        session_id,
                        EpisodeCase(
                            eval_question_id=eval_question.question_id,
                            source_question_id=source_question.question_id,
                            question_type=eval_question.question_type,
                            session_id=session_id,
                            episode_uuid=episode_uuid,
                            episode_name=f'{source_question.question_id}:{session_id}',
                            reference_time=reference_time,
                            body=format_session_transcript(session),
                            is_answer_session=False,
                            is_distractor=True,
                            date_delta_seconds=date_delta,
                        ),
                    )
                )

        chosen: list[EpisodeCase] = []
        for _, _, session_id, episode in sorted(
            candidates,
            key=lambda item: (item[0], item[1], item[2]),
        ):
            if session_id in used_session_ids:
                continue
            chosen.append(episode)
            used_session_ids.add(session_id)
            if len(chosen) >= distractors_per_question:
                break

        distractors_by_question[eval_question.question_id] = chosen

    return distractors_by_question


def effective_distractors_per_question(args: argparse.Namespace) -> int:
    return 0 if args.main_only else args.distractors_per_question


def build_sample_episode_cases(
    questions: Iterable[LongMemEvalQuestion],
) -> list[EpisodeCase]:
    episodes: list[EpisodeCase] = []

    for question in questions:
        gold_ids = set(gold_session_ids(question))
        for session_id, reference_time, session in zip(
            question.haystack_session_ids,
            question.haystack_dates,
            question.haystack_sessions,
            strict=True,
        ):
            episode_uuid = stable_episode_uuid(question.question_id, session_id)
            episodes.append(
                EpisodeCase(
                    eval_question_id=question.question_id,
                    source_question_id=question.question_id,
                    question_type=question.question_type,
                    session_id=session_id,
                    episode_uuid=episode_uuid,
                    episode_name=f'{question.question_id}:{session_id}',
                    reference_time=reference_time,
                    body=format_session_transcript(session),
                    is_answer_session=session_id in gold_ids,
                    is_distractor=False,
                )
            )

    return episodes


def build_episode_cases(
    selected_questions: Sequence[LongMemEvalQuestion],
    all_questions: Sequence[LongMemEvalQuestion],
    distractors_per_question: int,
) -> list[EpisodeCase]:
    main_episodes = build_sample_episode_cases(selected_questions)
    if distractors_per_question == 0:
        return main_episodes

    distractors_by_question = select_distractor_episode_cases(
        all_questions,
        selected_questions,
        distractors_per_question,
    )
    distractor_episodes = [
        episode
        for question in selected_questions
        for episode in distractors_by_question.get(question.question_id, [])
    ]

    deduped: dict[str, EpisodeCase] = {}
    for episode in [*main_episodes, *distractor_episodes]:
        deduped.setdefault(episode.episode_uuid, episode)
    return list(deduped.values())


def build_gold_session_ids_by_question(
    questions: Iterable[LongMemEvalQuestion],
) -> dict[str, list[str]]:
    return {question.question_id: gold_session_ids(question) for question in questions}


def build_episode_uuid_to_session_id(
    episodes: Iterable[EpisodeCase],
) -> dict[str, str]:
    return {episode.episode_uuid: episode.session_id for episode in episodes}


def build_distractor_session_ids_by_question(
    episodes: Iterable[EpisodeCase],
) -> dict[str, list[str]]:
    distractors_by_question: dict[str, list[str]] = defaultdict(list)
    for episode in episodes:
        if episode.is_distractor:
            distractors_by_question[episode.eval_question_id].append(episode.session_id)
    return {
        question_id: sessions
        for question_id, sessions in sorted(distractors_by_question.items())
    }


def isoformat_or_none(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def matched_gold_sessions(
    edge: Any,
    episode_uuid_to_session_id: dict[str, str],
    gold_ids: set[str],
) -> list[str]:
    return sorted(
        {
            episode_uuid_to_session_id[episode_uuid]
            for episode_uuid in getattr(edge, 'episodes', [])
            if episode_uuid in episode_uuid_to_session_id
            and episode_uuid_to_session_id[episode_uuid] in gold_ids
        }
    )


def ndcg_at_k(edge_matches: Sequence[Sequence[str]], gold_ids: set[str], k: int) -> float:
    if not gold_ids:
        return 0.0

    covered: set[str] = set()
    dcg = 0.0
    for rank, matches in enumerate(edge_matches[:k], start=1):
        new_matches = set(matches) - covered
        if not new_matches:
            continue
        covered.update(new_matches)
        dcg += len(new_matches) / math.log2(rank + 1)

    ideal_count = min(len(gold_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return 0.0 if idcg == 0 else min(dcg / idcg, 1.0)


def score_edges(
    question: LongMemEvalQuestion,
    edges: Sequence[Any],
    episode_uuid_to_session_id: dict[str, str],
) -> dict[str, Any]:
    gold_ids = set(gold_session_ids(question))
    edge_matches = [
        matched_gold_sessions(edge, episode_uuid_to_session_id, gold_ids) for edge in edges
    ]
    gold_session_ranks: dict[str, int | None] = {session_id: None for session_id in sorted(gold_ids)}
    gold_session_first_edge_uuids: dict[str, str | None] = {
        session_id: None for session_id in sorted(gold_ids)
    }
    for rank, (edge, matches) in enumerate(zip(edges, edge_matches, strict=True), start=1):
        for session_id in matches:
            if gold_session_ranks[session_id] is None:
                gold_session_ranks[session_id] = rank
                gold_session_first_edge_uuids[session_id] = edge.uuid

    scored: dict[str, Any] = {
        'gold_session_ids': sorted(gold_ids),
        'gold_session_ranks': gold_session_ranks,
        'gold_session_first_edge_uuids': gold_session_first_edge_uuids,
        'edge_matched_gold_session_ids': edge_matches,
    }

    for k in METRIC_KS:
        covered = {
            session_id for matches in edge_matches[:k] for session_id in matches if session_id
        }
        missing = sorted(gold_ids - covered)
        scored[f'evidence_recall_at_{k}'] = 0.0 if not gold_ids else len(covered) / len(gold_ids)
        scored[f'any_evidence_hit_at_{k}'] = bool(covered)
        scored[f'all_evidence_hit_at_{k}'] = bool(gold_ids and covered >= gold_ids)
        scored[f'ndcg_at_{k}'] = ndcg_at_k(edge_matches, gold_ids, k)
        scored[f'missing_gold_session_ids_at_{k}'] = missing

    return scored


def build_top_edge_diagnostics(
    edges: Sequence[Any],
    scores: Sequence[float],
    edge_matches: Sequence[Sequence[str]],
    save_top_k: int,
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for index, edge in enumerate(edges[:save_top_k]):
        diagnostics.append(
            {
                'rank': index + 1,
                'edge_uuid': edge.uuid,
                'fact': edge.fact,
                'score': scores[index] if index < len(scores) else None,
                'episode_uuids': edge.episodes,
                'matched_gold_session_ids': list(edge_matches[index])
                if index < len(edge_matches)
                else [],
                'created_at': isoformat_or_none(edge.created_at),
                'valid_at': isoformat_or_none(edge.valid_at),
                'invalid_at': isoformat_or_none(edge.invalid_at),
                'expired_at': isoformat_or_none(edge.expired_at),
            }
        )
    return diagnostics


def load_env() -> None:
    cwd_env = Path.cwd() / '.env'
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        return

    if cwd_env.exists():
        load_dotenv(cwd_env, override=False)
    local_env = OUTPUT_DIR / '.env'
    if local_env.exists():
        load_dotenv(local_env, override=False)
    load_dotenv(override=False)


def get_neo4j_database(default_user: str | None) -> str:
    configured = os.getenv('NEO4J_DATABASE')
    if configured:
        return configured
    if default_user and default_user != 'neo4j':
        return default_user
    return 'neo4j'


def resolve_graph_group_id(args: argparse.Namespace) -> str:
    if args.graph_group_id:
        return args.graph_group_id
    return get_neo4j_database(os.getenv('NEO4J_USER') or os.getenv('NEO4J_USERNAME'))


def build_graphiti(args: argparse.Namespace, *, cross_encoder_provider: str = 'openai'):
    from graphiti_core import Graphiti
    from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
    from graphiti_core.driver.neo4j_driver import Neo4jDriver
    from graphiti_core.embedder import OpenAIEmbedder, OpenAIEmbedderConfig
    from graphiti_core.llm_client import LLMConfig, OpenAIClient

    neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    neo4j_user = os.getenv('NEO4J_USER') or os.getenv('NEO4J_USERNAME') or 'neo4j'
    neo4j_password = os.getenv('NEO4J_PASSWORD')
    neo4j_database = get_neo4j_database(neo4j_user)
    if not neo4j_uri or not neo4j_user or not neo4j_password:
        raise ValueError('NEO4J_URI, NEO4J_USER/NEO4J_USERNAME, and NEO4J_PASSWORD are required')

    driver = Neo4jDriver(
        uri=neo4j_uri,
        user=neo4j_user,
        password=neo4j_password,
        database=neo4j_database,
    )
    llm_config = LLMConfig(
        api_key=os.getenv('OPENAI_API_KEY'),
        model=args.llm_model,
        small_model=args.llm_small_model,
        temperature=0,
    )
    embedder_config = OpenAIEmbedderConfig(
        api_key=os.getenv('OPENAI_API_KEY'),
        embedding_model=args.embedding_model,
    )

    if cross_encoder_provider == 'openai':
        cross_encoder = OpenAIRerankerClient(
            config=LLMConfig(
                api_key=os.getenv('OPENAI_API_KEY'),
                model=args.openai_reranker_model,
                temperature=0,
            )
        )
    elif cross_encoder_provider == 'gemini':
        from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient

        cross_encoder = GeminiRerankerClient(
            config=LLMConfig(
                api_key=os.getenv('GEMINI_API_KEY'),
                model=args.gemini_reranker_model,
                temperature=0,
            )
        )
    elif cross_encoder_provider == 'bge':
        from graphiti_core.cross_encoder.bge_reranker_client import BGERerankerClient

        cross_encoder = BGERerankerClient()
    else:
        raise ValueError(f'Unsupported cross encoder provider: {cross_encoder_provider}')

    return Graphiti(
        graph_driver=driver,
        llm_client=OpenAIClient(config=llm_config),
        embedder=OpenAIEmbedder(config=embedder_config),
        cross_encoder=cross_encoder,
    )


def use_graph_group_database(graphiti: Any, group_id: str) -> None:
    if graphiti.driver._database != group_id:
        graphiti.driver = graphiti.driver.clone(database=group_id)
        graphiti.clients.driver = graphiti.driver


async def reset_group(graphiti: Any, group_id: str) -> None:
    await graphiti.driver.execute_query(
        'MATCH (n) WHERE n.group_id = $group_id DETACH DELETE n',
        group_id=group_id,
    )


async def get_existing_episode_names(graphiti: Any, group_id: str) -> set[str]:
    records, _, _ = await graphiti.driver.execute_query(
        'MATCH (e:Episodic {group_id: $group_id}) RETURN e.name AS name',
        group_id=group_id,
        routing_='r',
    )
    return {record['name'] for record in records if record.get('name')}


async def ensure_stable_episode_node(
    graphiti: Any,
    episode: EpisodeCase,
    graph_group_id: str,
) -> None:
    from graphiti_core.nodes import EpisodeType, EpisodicNode
    from graphiti_core.utils.datetime_utils import utc_now

    node = EpisodicNode(
        uuid=episode.episode_uuid,
        name=episode.episode_name,
        group_id=graph_group_id,
        labels=[],
        source=EpisodeType.message,
        source_description='LongMemEval oracle haystack session',
        content=episode.body,
        created_at=utc_now(),
        valid_at=episode.reference_time,
    )
    await node.save(graphiti.driver)


async def add_episode_with_retries(
    graphiti: Any,
    episode: EpisodeCase,
    graph_group_id: str,
    args: argparse.Namespace,
) -> None:
    from graphiti_core.llm_client import RateLimitError
    from graphiti_core.nodes import EpisodeType

    for attempt in range(args.max_ingest_retries + 1):
        try:
            await ensure_stable_episode_node(graphiti, episode, graph_group_id)
            await graphiti.add_episode(
                name=episode.episode_name,
                episode_body=episode.body,
                source_description='LongMemEval oracle haystack session',
                reference_time=episode.reference_time,
                source=EpisodeType.message,
                group_id=graph_group_id,
                uuid=episode.episode_uuid,
                previous_episode_uuids=[],
            )
            return
        except RateLimitError:
            if attempt >= args.max_ingest_retries:
                raise
            delay = min(
                args.ingest_retry_max_delay,
                args.ingest_retry_base_delay * (2**attempt),
            )
            print(
                f'Rate limit while ingesting {episode.episode_name}; retrying in {delay:.1f}s '
                f'({attempt + 1}/{args.max_ingest_retries})'
            )
            await asyncio.sleep(delay)


def build_search_config(reranker_name: str, limit: int):
    from graphiti_core.search.search_config_recipes import (
        EDGE_HYBRID_SEARCH_CROSS_ENCODER,
        EDGE_HYBRID_SEARCH_EPISODE_MENTIONS,
        EDGE_HYBRID_SEARCH_MMR,
        EDGE_HYBRID_SEARCH_RRF,
    )

    recipes = {
        'edge_rrf': EDGE_HYBRID_SEARCH_RRF,
        'edge_mmr': EDGE_HYBRID_SEARCH_MMR,
        'edge_episode_mentions': EDGE_HYBRID_SEARCH_EPISODE_MENTIONS,
        'edge_cross_encoder': EDGE_HYBRID_SEARCH_CROSS_ENCODER,
    }
    try:
        config = recipes[reranker_name].model_copy(deep=True)
    except KeyError as exc:
        raise ValueError(f'Unsupported reranker: {reranker_name}') from exc
    config.limit = limit
    return config


def build_temporal_decay_config(
    temporal_decay: str,
    question: LongMemEvalQuestion,
    args: argparse.Namespace,
):
    if temporal_decay == 'none':
        return None

    from graphiti_core.search.search_config import TemporalDecayConfig, TemporalDecayFunction

    return TemporalDecayConfig(
        function=TemporalDecayFunction(temporal_decay),
        reference_time=question.question_time,
        scale_days=args.temporal_decay_scale_days,
        temporal_weight=args.temporal_decay_weight,
    )


async def evaluate_one_reranker(
    args: argparse.Namespace,
    graphiti: Any,
    questions: list[LongMemEvalQuestion],
    reranker_name: str,
    temporal_decay: str,
    graph_group_id: str,
    episode_uuid_to_session_id: dict[str, str],
    distractor_session_ids_by_question: dict[str, list[str]],
) -> list[dict[str, Any]]:
    base_config = build_search_config(reranker_name, args.limit)
    rows: list[dict[str, Any]] = []

    for index, question in enumerate(questions, start=1):
        config = base_config.model_copy(deep=True)
        config.temporal_decay_config = build_temporal_decay_config(
            temporal_decay,
            question,
            args,
        )
        start = perf_counter()
        result = await graphiti.search_(
            question.question,
            config=config,
            group_ids=[graph_group_id],
        )
        latency_ms = round((perf_counter() - start) * 1000, 2)
        scores = result.edge_reranker_scores
        scored = score_edges(question, result.edges, episode_uuid_to_session_id)
        top_edges = build_top_edge_diagnostics(
            result.edges,
            scores,
            scored['edge_matched_gold_session_ids'],
            args.save_top_k,
        )
        distractor_session_ids = distractor_session_ids_by_question.get(question.question_id, [])
        row = {
            'reranker': reranker_name,
            'temporal_decay': temporal_decay,
            'temporal_decay_scale_days': args.temporal_decay_scale_days,
            'temporal_decay_weight': args.temporal_decay_weight,
            'cross_encoder_provider': args.cross_encoder_provider,
            'question_id': question.question_id,
            'question_type': question.question_type,
            'question': question.question,
            'answer': question.answer,
            'question_time': question.question_time.isoformat(),
            'gold_session_ids': scored['gold_session_ids'],
            'answer_session_ids': question.answer_session_ids,
            'haystack_session_ids': question.haystack_session_ids,
            'haystack_session_count': len(question.haystack_session_ids),
            'distractor_session_ids': distractor_session_ids,
            'distractor_session_count': len(distractor_session_ids),
            'main_only': args.main_only,
            'latency_ms': latency_ms,
            'result_count': len(result.edges),
            'top_edges': top_edges,
            'top_facts': [edge.fact for edge in result.edges[: args.save_top_k]],
            'top_scores': scores[: args.save_top_k],
            **{k: v for k, v in scored.items() if k != 'edge_matched_gold_session_ids'},
        }
        rows.append(row)
        print(
            f'[{reranker_name}/{temporal_decay} {index}/{len(questions)}] '
            f'{question.question_id}: recall@10={scored["evidence_recall_at_10"]:.3f}'
        )

    return rows


def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(items)
    if total == 0:
        return {}

    output: dict[str, Any] = {'queries': total}
    for k in METRIC_KS:
        output[f'evidence_recall_at_{k}'] = round(
            sum(float(item[f'evidence_recall_at_{k}']) for item in items) / total, 4
        )
        output[f'any_evidence_hit_at_{k}'] = round(
            sum(1 for item in items if item[f'any_evidence_hit_at_{k}']) / total, 4
        )
        output[f'all_evidence_hit_at_{k}'] = round(
            sum(1 for item in items if item[f'all_evidence_hit_at_{k}']) / total, 4
        )
        output[f'ndcg_at_{k}'] = round(
            sum(float(item[f'ndcg_at_{k}']) for item in items) / total, 4
        )
    output['avg_latency_ms'] = round(
        sum(float(item['latency_ms']) for item in items) / total, 2
    )
    return output


def build_gold_session_rankings(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            'question_id': item['question_id'],
            'question_type': item['question_type'],
            'gold_session_ids': item['gold_session_ids'],
            'gold_session_ranks': item['gold_session_ranks'],
            'gold_session_first_edge_uuids': item['gold_session_first_edge_uuids'],
            'missing_gold_session_ids_at_10': item['missing_gold_session_ids_at_10'],
            'evidence_recall_at_10': item['evidence_recall_at_10'],
            'all_evidence_hit_at_10': item['all_evidence_hit_at_10'],
            'ndcg_at_10': item['ndcg_at_10'],
        }
        for item in sorted(items, key=lambda row: (row['question_type'], row['question_id']))
    ]


def summarize(
    rows: list[dict[str, Any]],
    selected_questions: Sequence[LongMemEvalQuestion],
    episodes: Sequence[EpisodeCase],
    args: argparse.Namespace,
) -> dict[str, Any]:
    by_reranker_decay: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_type_reranker_decay: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        by_reranker_decay[(row['reranker'], row['temporal_decay'])].append(row)
        by_type_reranker_decay[
            (row['question_type'], row['reranker'], row['temporal_decay'])
        ].append(row)

    type_counts = Counter(question.question_type for question in selected_questions)
    distractor_session_ids_by_question = build_distractor_session_ids_by_question(episodes)
    return {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'dataset': str(Path(args.dataset)),
        'sample_size': args.sample_size,
        'seed': args.seed,
        'question_types': args.question_types,
        'main_only': args.main_only,
        'distractors_per_question': effective_distractors_per_question(args),
        'temporal_decays': args.temporal_decays,
        'temporal_decay_scale_days': args.temporal_decay_scale_days,
        'temporal_decay_weight': args.temporal_decay_weight,
        'metric_ks': list(METRIC_KS),
        'selected_question_ids': [question.question_id for question in selected_questions],
        'selected_question_type_counts': dict(sorted(type_counts.items())),
        'gold_session_ids_by_question': build_gold_session_ids_by_question(selected_questions),
        'sampled_haystack_session_count': sum(
            len(question.haystack_session_ids) for question in selected_questions
        ),
        'unique_episode_count': len(episodes),
        'unique_main_episode_count': sum(1 for episode in episodes if not episode.is_distractor),
        'unique_distractor_episode_count': sum(1 for episode in episodes if episode.is_distractor),
        'distractor_session_ids_by_question': distractor_session_ids_by_question,
        'total_rows': len(rows),
        'overall_reranker_temporal_decays': {
            f'{reranker}::{temporal_decay}': aggregate(items)
            for (reranker, temporal_decay), items in sorted(by_reranker_decay.items())
        },
        'gold_session_rankings_by_reranker_temporal_decay': {
            f'{reranker}::{temporal_decay}': build_gold_session_rankings(items)
            for (reranker, temporal_decay), items in sorted(by_reranker_decay.items())
        },
        'question_type_reranker_temporal_decays': {
            f'{question_type}::{reranker}::{temporal_decay}': aggregate(items)
            for (question_type, reranker, temporal_decay), items in sorted(
                by_type_reranker_decay.items()
            )
        },
    }


async def ingest_dataset(args: argparse.Namespace) -> None:
    all_questions, questions = load_all_and_sampled_questions(args)
    episodes = build_episode_cases(
        questions,
        all_questions,
        effective_distractors_per_question(args),
    )
    distractor_session_ids_by_question = build_distractor_session_ids_by_question(episodes)
    graph_group_id = resolve_graph_group_id(args)
    graphiti = build_graphiti(args)

    try:
        use_graph_group_database(graphiti, graph_group_id)
        await graphiti.build_indices_and_constraints()
        if args.reset_group:
            print(f'Resetting graph group/database partition: {graph_group_id}')
            await reset_group(graphiti, graph_group_id)

        existing_episode_names = (
            await get_existing_episode_names(graphiti, graph_group_id)
            if args.skip_existing
            else set()
        )

        print_selected_questions(questions, distractor_session_ids_by_question)
        print(
            'Planned episodes: '
            f'{len(episodes)} total, '
            f'{sum(1 for episode in episodes if not episode.is_distractor)} main, '
            f'{sum(1 for episode in episodes if episode.is_distractor)} distractors'
        )
        for index, episode in enumerate(episodes, start=1):
            if episode.episode_name in existing_episode_names:
                print(f'[{index}/{len(episodes)}] skipped existing {episode.episode_name}')
                continue
            try:
                await add_episode_with_retries(graphiti, episode, graph_group_id, args)
                existing_episode_names.add(episode.episode_name)
            except Exception as exc:
                print(
                    f'[{index}/{len(episodes)}] failed to ingest {episode.episode_name}: {exc}'
                )
                continue

            label = f'{episode.source_question_id}/{episode.session_id}'
            if episode.is_distractor:
                label = f'{label} distractor_for={episode.eval_question_id}'
            print(f'[{index}/{len(episodes)}] ingested {label}')
            if args.ingest_delay > 0:
                await asyncio.sleep(args.ingest_delay)
    finally:
        await graphiti.close()


async def evaluate_dataset(args: argparse.Namespace) -> None:
    all_questions, questions = load_all_and_sampled_questions(args)
    episodes = build_episode_cases(
        questions,
        all_questions,
        effective_distractors_per_question(args),
    )
    distractor_session_ids_by_question = build_distractor_session_ids_by_question(episodes)
    graph_group_id = resolve_graph_group_id(args)
    episode_uuid_to_session_id = build_episode_uuid_to_session_id(episodes)
    graphiti = build_graphiti(args, cross_encoder_provider=args.cross_encoder_provider)
    all_rows: list[dict[str, Any]] = []

    try:
        use_graph_group_database(graphiti, graph_group_id)
        print_selected_questions(questions, distractor_session_ids_by_question)
        for reranker_name in args.rerankers:
            for temporal_decay in args.temporal_decays:
                rows = await evaluate_one_reranker(
                    args,
                    graphiti,
                    questions,
                    reranker_name,
                    temporal_decay,
                    graph_group_id,
                    episode_uuid_to_session_id,
                    distractor_session_ids_by_question,
                )
                all_rows.extend(rows)
    finally:
        await graphiti.close()

    results_path = Path(args.results)
    summary_path = Path(args.summary)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open('w', encoding='utf-8') as file:
        for row in all_rows:
            file.write(json.dumps(row, ensure_ascii=False) + '\n')

    summary = summarize(all_rows, questions, episodes, args)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Wrote per-query rows: {results_path}')
    print(f'Wrote summary: {summary_path}')
    print(json.dumps(summary['overall_reranker_temporal_decays'], indent=2, ensure_ascii=False))


async def run(args: argparse.Namespace) -> None:
    await ingest_dataset(args)
    await evaluate_dataset(args)


def print_selected_questions(
    questions: Sequence[LongMemEvalQuestion],
    distractor_session_ids_by_question: dict[str, list[str]] | None = None,
) -> None:
    print('Selected LongMemEval questions:')
    for question in questions:
        distractor_count = 0
        if distractor_session_ids_by_question is not None:
            distractor_count = len(distractor_session_ids_by_question.get(question.question_id, []))
        print(
            f'- {question.question_id} [{question.question_type}] '
            f'haystack={len(question.haystack_session_ids)} '
            f'gold={len(gold_session_ids(question))} '
            f'distractors={distractor_count}'
        )


def sample(args: argparse.Namespace) -> None:
    all_questions, questions = load_all_and_sampled_questions(args)
    episodes = build_episode_cases(
        questions,
        all_questions,
        effective_distractors_per_question(args),
    )
    distractor_session_ids_by_question = build_distractor_session_ids_by_question(episodes)
    print_selected_questions(questions, distractor_session_ids_by_question)
    print(
        json.dumps(
            {
                'sample_size': len(questions),
                'seed': args.seed,
                'question_types': args.question_types,
                'main_only': args.main_only,
                'distractors_per_question': effective_distractors_per_question(args),
                'selected_question_ids': [question.question_id for question in questions],
                'selected_question_type_counts': dict(
                    sorted(Counter(question.question_type for question in questions).items())
                ),
                'gold_session_ids_by_question': build_gold_session_ids_by_question(questions),
                'sampled_haystack_session_count': sum(
                    len(question.haystack_session_ids) for question in questions
                ),
                'unique_episode_count': len(episodes),
                'unique_main_episode_count': sum(
                    1 for episode in episodes if not episode.is_distractor
                ),
                'unique_distractor_episode_count': sum(
                    1 for episode in episodes if episode.is_distractor
                ),
                'distractor_session_ids_by_question': distractor_session_ids_by_question,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Run turn-gold LongMemEval reranker evaluation for Graphiti.'
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument('--dataset', default=str(DEFAULT_DATASET))
        subparser.add_argument('--sample-size', type=int, default=DEFAULT_SAMPLE_SIZE)
        subparser.add_argument('--seed', type=int, default=DEFAULT_SEED)
        subparser.add_argument(
            '--distractors-per-question',
            type=int,
            default=DEFAULT_DISTRACTORS_PER_QUESTION,
            help='Same-type nearby sessions to add per sampled question.',
        )
        subparser.add_argument(
            '--main-only',
            action='store_true',
            help='Use only sampled haystack sessions and ignore distractors.',
        )
        subparser.add_argument('--question-types', nargs='+', default=DEFAULT_QUESTION_TYPES)
        subparser.add_argument('--graph-group-id')
        subparser.add_argument('--llm-model', default='gpt-4.1')
        subparser.add_argument('--llm-small-model', default='gpt-4.1-mini')
        subparser.add_argument('--embedding-model', default='text-embedding-3-small')
        subparser.add_argument('--openai-reranker-model', default='gpt-4.1-nano')
        subparser.add_argument('--gemini-reranker-model', default='gemini-2.5-flash')
        subparser.add_argument(
            '--cross-encoder-provider',
            choices=['openai', 'gemini', 'bge'],
            default='bge',
        )

    def add_ingest_args(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument('--reset-group', action='store_true')
        subparser.add_argument('--ingest-delay', type=float, default=0.2)
        subparser.add_argument('--max-ingest-retries', type=int, default=8)
        subparser.add_argument('--ingest-retry-base-delay', type=float, default=2.0)
        subparser.add_argument('--ingest-retry-max-delay', type=float, default=60.0)
        subparser.add_argument(
            '--skip-existing',
            action=argparse.BooleanOptionalAction,
            default=True,
        )

    def add_evaluate_args(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument('--rerankers', nargs='+', default=RERANKER_NAMES)
        subparser.add_argument(
            '--temporal-decays',
            nargs='+',
            choices=TEMPORAL_DECAY_CHOICES,
            default=DEFAULT_TEMPORAL_DECAYS,
        )
        subparser.add_argument('--temporal-decay-scale-days', type=float, default=30.0)
        subparser.add_argument('--temporal-decay-weight', type=float, default=0.4)
        subparser.add_argument('--limit', type=int, default=10)
        subparser.add_argument('--save-top-k', type=int, default=10)
        subparser.add_argument('--results', default=str(DEFAULT_RESULTS))
        subparser.add_argument('--summary', default=str(DEFAULT_SUMMARY))

    sample_parser = subparsers.add_parser('sample', help='Print the deterministic sample.')
    add_common(sample_parser)

    ingest_parser = subparsers.add_parser('ingest', help='Ingest sampled sessions into Graphiti.')
    add_common(ingest_parser)
    add_ingest_args(ingest_parser)

    evaluate_parser = subparsers.add_parser(
        'evaluate', help='Evaluate rerankers on an ingested LongMemEval sample.'
    )
    add_common(evaluate_parser)
    add_evaluate_args(evaluate_parser)

    run_parser = subparsers.add_parser('run', help='Ingest then evaluate rerankers.')
    add_common(run_parser)
    add_ingest_args(run_parser)
    add_evaluate_args(run_parser)

    return parser


async def main() -> None:
    load_env()
    args = build_arg_parser().parse_args()
    if args.command == 'sample':
        sample(args)
    elif args.command == 'ingest':
        await ingest_dataset(args)
    elif args.command == 'evaluate':
        await evaluate_dataset(args)
    elif args.command == 'run':
        await run(args)
    else:
        raise SystemExit(f'Unsupported command: {args.command}')


if __name__ == '__main__':
    asyncio.run(main())
