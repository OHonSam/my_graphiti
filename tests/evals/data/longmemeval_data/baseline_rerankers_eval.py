"""
Baseline reranker evaluator for a sampled LongMemEval oracle subset.

The runner ingests only the haystack sessions for a deterministic sample of
LongMemEval questions, then scores EntityEdge retrieval by provenance: an edge
is correct when one of its source episode UUIDs maps back to a LongMemEval
answer_session_id.

Examples:
    uv run python tests/evals/data/longmemeval_data/baseline_rerankers_eval.py sample --sample-size 10
    uv run python tests/evals/data/longmemeval_data/baseline_rerankers_eval.py ingest --sample-size 10 --reset-group
    uv run python tests/evals/data/longmemeval_data/baseline_rerankers_eval.py evaluate --sample-size 10 --rerankers edge_rrf edge_episode_mentions
    uv run python tests/evals/data/longmemeval_data/baseline_rerankers_eval.py run --sample-size 10 --rerankers edge_rrf edge_mmr edge_episode_mentions edge_cross_encoder --limit 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
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

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from graphiti_core import Graphiti
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.driver.neo4j_driver import Neo4jDriver
from graphiti_core.embedder import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client import LLMConfig, OpenAIClient, RateLimitError
from graphiti_core.nodes import EpisodeType, EpisodicNode
from graphiti_core.search.search_config import SearchConfig
from graphiti_core.search.search_config_recipes import (
    EDGE_HYBRID_SEARCH_CROSS_ENCODER,
    EDGE_HYBRID_SEARCH_EPISODE_MENTIONS,
    EDGE_HYBRID_SEARCH_MMR,
    EDGE_HYBRID_SEARCH_RRF,
)
from graphiti_core.utils.datetime_utils import utc_now

OUTPUT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = OUTPUT_DIR / 'longmemeval_oracle.json'
DEFAULT_RESULTS = OUTPUT_DIR / 'baseline_reranker_results_v1.jsonl'
DEFAULT_SUMMARY = OUTPUT_DIR / 'baseline_reranker_summary_v1.json'
DEFAULT_SAMPLE_SIZE = 10
DEFAULT_SEED = 13
DEFAULT_QUESTION_TYPES = ['temporal-reasoning', 'knowledge-update']

RERANKER_RECIPES = {
    'edge_rrf': EDGE_HYBRID_SEARCH_RRF,
    'edge_mmr': EDGE_HYBRID_SEARCH_MMR,
    'edge_episode_mentions': EDGE_HYBRID_SEARCH_EPISODE_MENTIONS,
    'edge_cross_encoder': EDGE_HYBRID_SEARCH_CROSS_ENCODER,
}


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
    question_id: str
    question_type: str
    session_id: str
    episode_uuid: str
    episode_name: str
    reference_time: datetime
    body: str
    is_answer_session: bool


def parse_lme_datetime(value: str) -> datetime:
    """Parse LongMemEval timestamps such as ``2023/04/10 (Mon) 17:50``."""
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
        and question.answer_session_ids
    ]
    by_type: dict[str, list[LongMemEvalQuestion]] = defaultdict(list)
    for question in eligible:
        by_type[question.question_type].append(question)

    rng = random.Random(seed)
    selected: list[LongMemEvalQuestion] = []
    selected_ids: set[str] = set()
    type_count = max(1, len(question_types))
    base = sample_size // type_count
    remainder = sample_size % type_count

    for index, question_type in enumerate(question_types):
        target_count = base + (1 if index < remainder else 0)
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


def build_episode_cases(questions: Iterable[LongMemEvalQuestion]) -> list[EpisodeCase]:
    episodes: list[EpisodeCase] = []

    for question in questions:
        answer_session_ids = set(question.answer_session_ids)
        for session_id, reference_time, session in zip(
            question.haystack_session_ids,
            question.haystack_dates,
            question.haystack_sessions,
            strict=True,
        ):
            episode_uuid = stable_episode_uuid(question.question_id, session_id)
            episodes.append(
                EpisodeCase(
                    question_id=question.question_id,
                    question_type=question.question_type,
                    session_id=session_id,
                    episode_uuid=episode_uuid,
                    episode_name=f'{question.question_id}:{session_id}',
                    reference_time=reference_time,
                    body=format_session_transcript(session),
                    is_answer_session=session_id in answer_session_ids,
                )
            )

    return episodes


def build_episode_uuid_to_session_id(
    questions: Iterable[LongMemEvalQuestion],
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for question in questions:
        for session_id in question.haystack_session_ids:
            mapping[stable_episode_uuid(question.question_id, session_id)] = session_id
    return mapping


def load_sampled_questions(args: argparse.Namespace) -> list[LongMemEvalQuestion]:
    questions = load_questions(Path(args.dataset))
    return select_questions(questions, args.sample_size, args.seed, args.question_types)


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


def use_graph_group_database(graphiti: Graphiti, group_id: str) -> None:
    if graphiti.driver._database != group_id:
        graphiti.driver = graphiti.driver.clone(database=group_id)
        graphiti.clients.driver = graphiti.driver


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


async def ensure_stable_episode_node(
    graphiti: Graphiti,
    episode: EpisodeCase,
    graph_group_id: str,
) -> None:
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
    graphiti: Graphiti,
    episode: EpisodeCase,
    graph_group_id: str,
    args: argparse.Namespace,
) -> None:
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


async def ingest_dataset(args: argparse.Namespace) -> None:
    questions = load_sampled_questions(args)
    episodes = build_episode_cases(questions)
    graph_group_id = resolve_graph_group_id(args)
    graphiti = build_graphiti(args)

    try:
        use_graph_group_database(graphiti, graph_group_id)
        await graphiti.build_indices_and_constraints()
        if args.reset_group:
            print(f'Resetting graph group/database partition: {graph_group_id}')
            await reset_group(graphiti, graph_group_id)

        existing_episode_names = (
            await get_existing_episode_names(graphiti, graph_group_id) if args.skip_existing else set()
        )

        print_selected_questions(questions)
        for index, episode in enumerate(episodes, start=1):
            if episode.episode_name in existing_episode_names:
                print(f'[{index}/{len(episodes)}] skipped existing {episode.episode_name}')
                continue

            await add_episode_with_retries(graphiti, episode, graph_group_id, args)
            existing_episode_names.add(episode.episode_name)
            print(
                f'[{index}/{len(episodes)}] ingested '
                f'{episode.question_id}/{episode.session_id}'
            )
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


def isoformat_or_none(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def score_edges(
    question: LongMemEvalQuestion,
    edges: Sequence[Any],
    episode_uuid_to_session_id: dict[str, str],
) -> dict[str, Any]:
    gold_session_ids = set(question.answer_session_ids)
    answer_session_ranks: dict[str, int] = {}
    edge_matches: list[list[str]] = []

    for rank, edge in enumerate(edges, start=1):
        matched_sessions = sorted(
            {
                episode_uuid_to_session_id[episode_uuid]
                for episode_uuid in getattr(edge, 'episodes', [])
                if episode_uuid in episode_uuid_to_session_id
                and episode_uuid_to_session_id[episode_uuid] in gold_session_ids
            }
        )
        edge_matches.append(matched_sessions)
        for session_id in matched_sessions:
            answer_session_ranks.setdefault(session_id, rank)

    first_evidence_rank = min(answer_session_ranks.values()) if answer_session_ranks else None

    def covered_at(k: int) -> set[str]:
        return {session_id for session_id, rank in answer_session_ranks.items() if rank <= k}

    def recall_at(k: int) -> float:
        if not gold_session_ids:
            return 0.0
        return len(covered_at(k)) / len(gold_session_ids)

    return {
        'first_evidence_rank': first_evidence_rank,
        'first_evidence_mrr': 0.0
        if first_evidence_rank is None
        else 1.0 / first_evidence_rank,
        'any_evidence_hit_at_1': bool(covered_at(1)),
        'any_evidence_hit_at_3': bool(covered_at(3)),
        'any_evidence_hit_at_5': bool(covered_at(5)),
        'evidence_recall_at_1': recall_at(1),
        'evidence_recall_at_3': recall_at(3),
        'evidence_recall_at_5': recall_at(5),
        'all_evidence_hit_at_5': bool(
            gold_session_ids and len(covered_at(5)) == len(gold_session_ids)
        ),
        'answer_session_ranks': dict(sorted(answer_session_ranks.items())),
        'edge_matched_answer_session_ids': edge_matches,
    }


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
                'matched_answer_session_ids': list(edge_matches[index])
                if index < len(edge_matches)
                else [],
                'created_at': isoformat_or_none(edge.created_at),
                'valid_at': isoformat_or_none(edge.valid_at),
                'invalid_at': isoformat_or_none(edge.invalid_at),
                'expired_at': isoformat_or_none(edge.expired_at),
            }
        )
    return diagnostics


async def evaluate_one_reranker(
    args: argparse.Namespace,
    graphiti: Graphiti,
    questions: list[LongMemEvalQuestion],
    reranker_name: str,
    graph_group_id: str,
    episode_uuid_to_session_id: dict[str, str],
) -> list[dict[str, Any]]:
    config = build_search_config(reranker_name, args.limit)
    rows: list[dict[str, Any]] = []

    for index, question in enumerate(questions, start=1):
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
            scored['edge_matched_answer_session_ids'],
            args.save_top_k,
        )
        row = {
            'reranker': reranker_name,
            'cross_encoder_provider': args.cross_encoder_provider,
            'question_id': question.question_id,
            'question_type': question.question_type,
            'question': question.question,
            'answer': question.answer,
            'question_time': question.question_time.isoformat(),
            'answer_session_ids': question.answer_session_ids,
            'haystack_session_ids': question.haystack_session_ids,
            'latency_ms': latency_ms,
            'result_count': len(result.edges),
            'top_edges': top_edges,
            'top_facts': [edge.fact for edge in result.edges[: args.save_top_k]],
            'top_scores': scores[: args.save_top_k],
            **{k: v for k, v in scored.items() if k != 'edge_matched_answer_session_ids'},
        }
        rows.append(row)
        print(
            f'[{reranker_name} {index}/{len(questions)}] '
            f'{question.question_id}: first_evidence_rank={scored["first_evidence_rank"]}'
        )

    return rows


def summarize(
    rows: list[dict[str, Any]],
    selected_questions: Sequence[LongMemEvalQuestion],
    args: argparse.Namespace,
) -> dict[str, Any]:
    by_reranker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_reranker_type: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_reranker[row['reranker']].append(row)
        by_reranker_type[(row['reranker'], row['question_type'])].append(row)

    def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(items)
        if total == 0:
            return {}
        return {
            'queries': total,
            'any_evidence_hit_at_1': round(
                sum(1 for item in items if item['any_evidence_hit_at_1']) / total, 4
            ),
            'any_evidence_hit_at_3': round(
                sum(1 for item in items if item['any_evidence_hit_at_3']) / total, 4
            ),
            'any_evidence_hit_at_5': round(
                sum(1 for item in items if item['any_evidence_hit_at_5']) / total, 4
            ),
            'evidence_recall_at_1': round(
                sum(float(item['evidence_recall_at_1']) for item in items) / total, 4
            ),
            'evidence_recall_at_3': round(
                sum(float(item['evidence_recall_at_3']) for item in items) / total, 4
            ),
            'evidence_recall_at_5': round(
                sum(float(item['evidence_recall_at_5']) for item in items) / total, 4
            ),
            'all_evidence_hit_at_5': round(
                sum(1 for item in items if item['all_evidence_hit_at_5']) / total, 4
            ),
            'first_evidence_mrr': round(
                sum(float(item['first_evidence_mrr']) for item in items) / total, 4
            ),
            'avg_latency_ms': round(sum(float(item['latency_ms']) for item in items) / total, 2),
        }

    profile_counts = Counter(question.question_type for question in selected_questions)
    return {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'dataset': str(Path(args.dataset)),
        'sample_size': args.sample_size,
        'seed': args.seed,
        'question_types': args.question_types,
        'selected_question_ids': [question.question_id for question in selected_questions],
        'selected_question_type_counts': dict(sorted(profile_counts.items())),
        'total_rows': len(rows),
        'rerankers': {name: aggregate(items) for name, items in sorted(by_reranker.items())},
        'reranker_question_types': {
            f'{reranker}::{question_type}': aggregate(items)
            for (reranker, question_type), items in sorted(by_reranker_type.items())
        },
    }


async def evaluate_dataset(args: argparse.Namespace) -> None:
    questions = load_sampled_questions(args)
    graph_group_id = resolve_graph_group_id(args)
    episode_uuid_to_session_id = build_episode_uuid_to_session_id(questions)
    graphiti = build_graphiti(args, cross_encoder_provider=args.cross_encoder_provider)
    all_rows: list[dict[str, Any]] = []

    try:
        use_graph_group_database(graphiti, graph_group_id)
        print_selected_questions(questions)
        for reranker_name in args.rerankers:
            rows = await evaluate_one_reranker(
                args,
                graphiti,
                questions,
                reranker_name,
                graph_group_id,
                episode_uuid_to_session_id,
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

    summary = summarize(all_rows, questions, args)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Wrote per-query rows: {results_path}')
    print(f'Wrote summary: {summary_path}')
    print(json.dumps(summary['rerankers'], indent=2, ensure_ascii=False))


async def run(args: argparse.Namespace) -> None:
    await ingest_dataset(args)
    await evaluate_dataset(args)


def print_selected_questions(questions: Sequence[LongMemEvalQuestion]) -> None:
    print('Selected LongMemEval questions:')
    for question in questions:
        print(f'- {question.question_id} [{question.question_type}]')


def sample(args: argparse.Namespace) -> None:
    questions = load_sampled_questions(args)
    print_selected_questions(questions)
    print(
        json.dumps(
            {
                'sample_size': len(questions),
                'seed': args.seed,
                'question_types': args.question_types,
                'selected_question_ids': [question.question_id for question in questions],
                'selected_question_type_counts': dict(
                    sorted(Counter(question.question_type for question in questions).items())
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Run sampled LongMemEval reranker baselines for Graphiti.'
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument('--dataset', default=str(DEFAULT_DATASET))
        subparser.add_argument('--sample-size', type=int, default=DEFAULT_SAMPLE_SIZE)
        subparser.add_argument('--seed', type=int, default=DEFAULT_SEED)
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
        subparser.add_argument('--skip-existing', action=argparse.BooleanOptionalAction, default=True)

    sample_parser = subparsers.add_parser('sample', help='Print the deterministic question sample.')
    add_common(sample_parser)

    ingest_parser = subparsers.add_parser('ingest', help='Ingest sampled sessions into Graphiti.')
    add_common(ingest_parser)
    add_ingest_args(ingest_parser)

    evaluate_parser = subparsers.add_parser(
        'evaluate', help='Evaluate rerankers on an ingested LongMemEval sample.'
    )
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
