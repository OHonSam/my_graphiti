"""
Baseline evaluator for the synthetic temporal reranking dataset.

The runner ingests cleaned synthetic scenarios into Graphiti, then evaluates
EntityEdge retrieval with different search/reranker recipes. V1 scoring is
keyword-based: a result is correct when its fact contains all gold keywords, and
negative-hit diagnostics are computed from negative_fact_contains.

Examples:
    uv run python tests/evals/data/synthetic_temporal_reranking_data/baseline_rerankers.py run --reset-group
    uv run python tests/evals/data/synthetic_temporal_reranking_data/baseline_rerankers.py evaluate --rerankers edge_rrf edge_mmr edge_episode_mentions edge_cross_encoder
    uv run python tests/evals/data/synthetic_temporal_reranking_data/baseline_rerankers.py ingest 
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import nltk
from nltk.corpus import stopwords
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from graphiti_core import Graphiti
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.driver.neo4j_driver import Neo4jDriver
from graphiti_core.embedder import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client import LLMConfig, OpenAIClient, RateLimitError
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config import SearchConfig
from graphiti_core.search.search_config_recipes import (
    EDGE_HYBRID_SEARCH_CROSS_ENCODER,
    EDGE_HYBRID_SEARCH_EPISODE_MENTIONS,
    EDGE_HYBRID_SEARCH_MMR,
    EDGE_HYBRID_SEARCH_RRF,
)

OUTPUT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = OUTPUT_DIR / 'temporal_scenarios_v1.jsonl'
DEFAULT_RESULTS = OUTPUT_DIR / 'baseline_reranker_results_v1.jsonl'
DEFAULT_SUMMARY = OUTPUT_DIR / 'baseline_reranker_summary_v1.json'
DEFAULT_GROUP_ID = 'temporal_benchmark_v1'

RERANKER_RECIPES = {
    'edge_rrf': EDGE_HYBRID_SEARCH_RRF,
    'edge_mmr': EDGE_HYBRID_SEARCH_MMR,
    'edge_episode_mentions': EDGE_HYBRID_SEARCH_EPISODE_MENTIONS,
    'edge_cross_encoder': EDGE_HYBRID_SEARCH_CROSS_ENCODER,
}

TOKEN_RE = re.compile(r'[a-z0-9]+')


def load_stopwords() -> set[str]:
    try:
        return set(stopwords.words('english'))
    except LookupError as exc:
        raise RuntimeError(
            'NLTK stopwords corpus is missing. Run: '
            'uv run python -c "import nltk; nltk.download(\'stopwords\')"'
        ) from exc


STOPWORDS = load_stopwords()


@dataclass(frozen=True)
class QueryCase:
    scenario_id: str
    domain: str
    query_id: str
    query: str
    query_time: datetime
    temporal_profile: str
    gold_answer: str
    gold_fact_contains: list[str]
    negative_fact_contains: list[str]


@dataclass(frozen=True)
class EpisodeCase:
    scenario_id: str
    episode_id: str
    reference_time: datetime
    body: str


def normalize_text(value: str) -> str:
    return re.sub(r'\s+', ' ', value).strip().lower()


def tokenize(value: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(normalize_text(value)) if token not in STOPWORDS]


def phrase_supported(phrase: str, text: str) -> bool:
    phrase_text = normalize_text(phrase)
    text_value = normalize_text(text)
    if not phrase_text:
        return False
    if phrase_text in text_value:
        return True

    phrase_tokens = tokenize(phrase)
    text_tokens = set(tokenize(text))
    if not phrase_tokens:
        return False

    matched = sum(1 for token in phrase_tokens if token in text_tokens)
    if len(phrase_tokens) <= 2:
        return matched == len(phrase_tokens)
    return matched >= max(2, len(phrase_tokens) - 1)


def all_phrases_supported(phrases: list[str], text: str) -> bool:
    return bool(phrases) and all(phrase_supported(phrase, text) for phrase in phrases)


def any_phrase_supported(phrases: list[str], text: str) -> bool:
    return any(phrase_supported(phrase, text) for phrase in phrases)


def parse_utc(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if dt.tzinfo is None:
        raise ValueError(f'timestamp must be timezone-aware: {value}')
    return dt.astimezone(timezone.utc)


def load_dataset(path: Path) -> tuple[list[dict[str, Any]], list[EpisodeCase], list[QueryCase]]:
    scenarios: list[dict[str, Any]] = []
    episodes: list[EpisodeCase] = []
    queries: list[QueryCase] = []

    with path.open('r', encoding='utf-8') as file:
        for line_no, line in enumerate(file, start=1):
            if not line.strip():
                continue
            scenario = json.loads(line)
            scenarios.append(scenario)
            scenario_id = scenario['scenario_id']
            domain = scenario['domain']

            for episode in scenario['episodes']:
                episodes.append(
                    EpisodeCase(
                        scenario_id=scenario_id,
                        episode_id=episode['episode_id'],
                        reference_time=parse_utc(episode['reference_time']),
                        body=episode['body'],
                    )
                )

            for query in scenario['queries']:
                queries.append(
                    QueryCase(
                        scenario_id=scenario_id,
                        domain=domain,
                        query_id=query['query_id'],
                        query=query['query'],
                        query_time=parse_utc(query['query_time']),
                        temporal_profile=query['temporal_profile'],
                        gold_answer=query['gold_answer'],
                        gold_fact_contains=query['gold_fact_contains'],
                        negative_fact_contains=query.get('negative_fact_contains', []),
                    )
                )

    return scenarios, episodes, queries


def load_env() -> None:
    cwd_env = Path.cwd() / '.env'
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


def build_graphiti(args: argparse.Namespace, *, cross_encoder_provider: str = 'openai') -> Graphiti:
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


def resolve_graph_group_id(args: argparse.Namespace) -> str:
    if args.graph_group_id:
        return args.graph_group_id
    return get_neo4j_database(os.getenv('NEO4J_USER') or os.getenv('NEO4J_USERNAME'))


async def reset_group(graphiti: Graphiti, group_id: str) -> None:
    await graphiti.driver.execute_query(
        'MATCH (n) WHERE n.group_id = $group_id DETACH DELETE n',
        group_id=group_id,
    )


async def get_existing_episode_names(graphiti: Graphiti, group_id: str) -> set[str]:
    records, _, _ = await graphiti.driver.execute_query(
        'MATCH (e:Episodic {group_id: $group_id}) RETURN e.name AS name',
        group_id=group_id,
        routing_='r',
    )
    return {record['name'] for record in records if record.get('name')}


async def add_episode_with_retries(
    graphiti: Graphiti,
    episode: EpisodeCase,
    graph_group_id: str,
    args: argparse.Namespace,
) -> None:
    name = f'{episode.scenario_id}:{episode.episode_id}'
    for attempt in range(args.max_ingest_retries + 1):
        try:
            await graphiti.add_episode(
                name=name,
                episode_body=episode.body,
                source_description='synthetic temporal reranking benchmark',
                reference_time=episode.reference_time,
                source=EpisodeType.text,
                group_id=graph_group_id,
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
                f'Rate limit while ingesting {name}; retrying in {delay:.1f}s '
                f'({attempt + 1}/{args.max_ingest_retries})'
            )
            await asyncio.sleep(delay)


async def ingest_dataset(args: argparse.Namespace) -> None:
    _, episodes, _ = load_dataset(Path(args.dataset))
    graph_group_id = resolve_graph_group_id(args)
    graphiti = build_graphiti(args)

    try:
        await graphiti.build_indices_and_constraints()
        if args.reset_group:
            print(f'Resetting graph group/database partition: {graph_group_id}')
            await reset_group(graphiti, graph_group_id)

        existing_episode_names = (
            await get_existing_episode_names(graphiti, graph_group_id) if args.skip_existing else set()
        )

        for index, episode in enumerate(episodes, start=1):
            episode_name = f'{episode.scenario_id}:{episode.episode_id}'
            if episode_name in existing_episode_names:
                print(f'[{index}/{len(episodes)}] skipped existing {episode_name}')
                continue

            await add_episode_with_retries(graphiti, episode, graph_group_id, args)
            existing_episode_names.add(episode_name)
            print(f'[{index}/{len(episodes)}] ingested {episode.scenario_id}/{episode.episode_id}')
            if args.ingest_delay > 0:
                await asyncio.sleep(args.ingest_delay)
    finally:
        await graphiti.close()


def build_search_config(reranker_name: str, limit: int) -> SearchConfig:
    try:
        config = RERANKER_RECIPES[reranker_name].model_copy(deep=True)
    except KeyError as exc:
        raise ValueError(f'Unsupported reranker: {reranker_name}') from exc
    config.limit = limit
    return config


def score_query(query: QueryCase, facts: list[str], scores: list[float]) -> dict[str, Any]:
    gold_rank: int | None = None
    negative_ranks: list[int] = []

    for rank, fact in enumerate(facts, start=1):
        if gold_rank is None and all_phrases_supported(query.gold_fact_contains, fact):
            gold_rank = rank
        if any_phrase_supported(query.negative_fact_contains, fact):
            negative_ranks.append(rank)

    return {
        'gold_rank': gold_rank,
        'hit_at_1': gold_rank == 1,
        'hit_at_3': gold_rank is not None and gold_rank <= 3,
        'hit_at_5': gold_rank is not None and gold_rank <= 5,
        'mrr': 0.0 if gold_rank is None else 1.0 / gold_rank,
        'negative_above_gold': bool(
            gold_rank is not None and any(rank < gold_rank for rank in negative_ranks)
        ),
        'negative_ranks': negative_ranks,
        'top_fact': facts[0] if facts else None,
        'top_score': scores[0] if scores else None,
    }


async def evaluate_one_reranker(
    args: argparse.Namespace,
    graphiti: Graphiti,
    queries: list[QueryCase],
    reranker_name: str,
    graph_group_id: str,
) -> list[dict[str, Any]]:
    config = build_search_config(reranker_name, args.limit)
    rows: list[dict[str, Any]] = []

    for index, query in enumerate(queries, start=1):
        start = perf_counter()
        result = await graphiti.search_(
            query.query,
            config=config,
            group_ids=[graph_group_id],
        )
        latency_ms = round((perf_counter() - start) * 1000, 2)
        facts = [edge.fact for edge in result.edges]
        scores = result.edge_reranker_scores
        scored = score_query(query, facts, scores)
        row = {
            'reranker': reranker_name,
            'cross_encoder_provider': args.cross_encoder_provider,
            'scenario_id': query.scenario_id,
            'domain': query.domain,
            'query_id': query.query_id,
            'query': query.query,
            'query_time': query.query_time.isoformat(),
            'temporal_profile': query.temporal_profile,
            'gold_answer': query.gold_answer,
            'gold_fact_contains': query.gold_fact_contains,
            'negative_fact_contains': query.negative_fact_contains,
            'latency_ms': latency_ms,
            'result_count': len(facts),
            'top_facts': facts[: args.save_top_k],
            'top_scores': scores[: args.save_top_k],
            **scored,
        }
        rows.append(row)
        print(
            f'[{reranker_name} {index}/{len(queries)}] '
            f'{query.scenario_id}/{query.query_id}: rank={scored["gold_rank"]}'
        )

    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_reranker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_reranker_profile: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_reranker[row['reranker']].append(row)
        by_reranker_profile[(row['reranker'], row['temporal_profile'])].append(row)

    def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(items)
        if total == 0:
            return {}
        return {
            'queries': total,
            'hit_at_1': round(sum(1 for item in items if item['hit_at_1']) / total, 4),
            'hit_at_3': round(sum(1 for item in items if item['hit_at_3']) / total, 4),
            'hit_at_5': round(sum(1 for item in items if item['hit_at_5']) / total, 4),
            'mrr': round(sum(float(item['mrr']) for item in items) / total, 4),
            'negative_above_gold_rate': round(
                sum(1 for item in items if item['negative_above_gold']) / total, 4
            ),
            'avg_latency_ms': round(sum(float(item['latency_ms']) for item in items) / total, 2),
        }

    profile_counts = Counter(row['temporal_profile'] for row in rows)
    return {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'total_rows': len(rows),
        'profile_counts': dict(sorted(profile_counts.items())),
        'rerankers': {name: aggregate(items) for name, items in sorted(by_reranker.items())},
        'reranker_profiles': {
            f'{reranker}::{profile}': aggregate(items)
            for (reranker, profile), items in sorted(by_reranker_profile.items())
        },
    }


async def evaluate_dataset(args: argparse.Namespace) -> None:
    _, _, queries = load_dataset(Path(args.dataset))
    graph_group_id = resolve_graph_group_id(args)
    graphiti = build_graphiti(args, cross_encoder_provider=args.cross_encoder_provider)
    all_rows: list[dict[str, Any]] = []

    try:
        for reranker_name in args.rerankers:
            rows = await evaluate_one_reranker(args, graphiti, queries, reranker_name, graph_group_id)
            all_rows.extend(rows)
    finally:
        await graphiti.close()

    results_path = Path(args.results)
    summary_path = Path(args.summary)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open('w', encoding='utf-8') as file:
        for row in all_rows:
            file.write(json.dumps(row, ensure_ascii=False) + '\n')

    summary = summarize(all_rows)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Wrote per-query rows: {results_path}')
    print(f'Wrote summary: {summary_path}')
    print(json.dumps(summary['rerankers'], indent=2, ensure_ascii=False))


async def run(args: argparse.Namespace) -> None:
    await ingest_dataset(args)
    await evaluate_dataset(args)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run temporal reranking baselines for Graphiti.')
    subparsers = parser.add_subparsers(dest='command', required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument('--dataset', default=str(DEFAULT_DATASET))
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
        subparser.add_argument('--ingest-delay', type=float, default=0.5)
        subparser.add_argument('--max-ingest-retries', type=int, default=8)
        subparser.add_argument('--ingest-retry-base-delay', type=float, default=2.0)
        subparser.add_argument('--ingest-retry-max-delay', type=float, default=60.0)
        subparser.add_argument('--skip-existing', action=argparse.BooleanOptionalAction, default=True)

    ingest_parser = subparsers.add_parser('ingest', help='Ingest cleaned scenarios into Graphiti.')
    add_common(ingest_parser)
    add_ingest_args(ingest_parser)

    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate rerankers on an ingested graph.')
    add_common(evaluate_parser)
    evaluate_parser.add_argument('--rerankers', nargs='+', default=list(RERANKER_RECIPES.keys()))
    evaluate_parser.add_argument('--limit', type=int, default=10)
    evaluate_parser.add_argument('--save-top-k', type=int, default=5)
    evaluate_parser.add_argument('--results', default=str(DEFAULT_RESULTS))
    evaluate_parser.add_argument('--summary', default=str(DEFAULT_SUMMARY))

    run_parser = subparsers.add_parser('run', help='Ingest then evaluate rerankers.')
    add_common(run_parser)
    add_ingest_args(run_parser)
    run_parser.add_argument('--rerankers', nargs='+', default=list(RERANKER_RECIPES.keys()))
    run_parser.add_argument('--limit', type=int, default=10)
    run_parser.add_argument('--save-top-k', type=int, default=5)
    run_parser.add_argument('--results', default=str(DEFAULT_RESULTS))
    run_parser.add_argument('--summary', default=str(DEFAULT_SUMMARY))

    return parser


async def main() -> None:
    load_env()
    args = build_arg_parser().parse_args()
    if args.command == 'ingest':
        await ingest_dataset(args)
    elif args.command == 'evaluate':
        await evaluate_dataset(args)
    elif args.command == 'run':
        await run(args)
    else:
        raise SystemExit(f'Unsupported command: {args.command}')


if __name__ == '__main__':
    asyncio.run(main())
