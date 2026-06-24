"""
Generate synthetic temporal reranking benchmark data for Graphiti.

This script creates timestamped natural-language episodes and labeled temporal
queries. It does not ingest data into Graphiti; gold edge UUIDs should be
resolved after Graphiti extracts edges from the cleaned scenarios.

Examples:
    uv run python tests/evals/data/synthetic_temporal_reranking_data/main.py generate
    uv run python tests/evals/data/synthetic_temporal_reranking_data/main.py validate
    uv run python tests/evals/data/synthetic_temporal_reranking_data/main.py sample
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


OUTPUT_DIR = Path(__file__).resolve().parent
GROUP_ID = 'temporal_benchmark_v1'
DEFAULT_MODEL = 'gpt-4o'

RAW_FILENAME = 'raw_temporal_scenarios_v1.jsonl'
CLEAN_FILENAME = 'temporal_scenarios_v1.jsonl'
REJECTED_FILENAME = 'rejected_scenarios_v1.jsonl'
STATS_FILENAME = 'dataset_stats_v1.json'

TEMPORAL_PROFILES = {
    'current_state',
    'latest_observation',
    'point_in_time',
    'timeline',
    'semantic_fact',
}

TEMPORAL_PROFILES_REQUIRING_NEGATIVES = TEMPORAL_PROFILES - {'semantic_fact'}

DEFAULT_DOMAINS = [
    'employee_role_change',
    'subscription_plan_change',
    'product_price_change',
    'software_version_update',
    'incident_status_update',
    'policy_update',
    'project_owner_change',
    'course_enrollment_change',
    'device_location_change',
    'support_ticket_status_change',
]

PROFILE_PRIORITY = [
    'timeline',
    'point_in_time',
    'latest_observation',
    'current_state',
    'semantic_fact',
]

MIN_EPISODES_PER_SCENARIO = 3
MAX_EPISODES_PER_SCENARIO = 5
MIN_QUERIES_PER_SCENARIO = 3
MAX_QUERIES_PER_SCENARIO = 5

TEMPORAL_TAXONOMY_GUIDE = """
Required temporal_taxonomy:
- timeline: query asks for sequence, history, development, cause, or why/how.
  This has highest priority because causal/history questions need multiple
  ordered facts; treating them as current/latest would miss the explanation.
- point_in_time: query asks what was true at an explicit date/time/period.
  This overrides recency because the answer must be valid for that timestamp,
  not necessarily latest/current.
- latest_observation: query asks for the latest/newest/most recent known update,
  report, announcement, or log entry. This prefers newest evidence, not
  necessarily the currently valid long-term state.
- current_state: query asks what is true/active/current at query_time. This
  prefers facts valid at query_time.
- semantic_fact: no temporal ranking objective.

Priority for assigning temporal_profile when cues overlap:
1. timeline
2. point_in_time
3. latest_observation
4. current_state
5. semantic_fact

If a query contains cues for multiple profiles, assign the highest-priority
profile because it represents the dominant retrieval objective. Rewrite or
remove ambiguous phrases such as "latest current status".
"""

SCENARIO_BATCH_OUTPUT_TEMPLATE = """
Output template:
Return exactly one JSON object with this top-level shape. Do not wrap it in
Markdown fences and do not add commentary outside the JSON.

{
  "scenarios": [
    {
      "scenario_id": "employee_role_change_001",
      "domain": "employee_role_change",
      "group_id": "temporal_benchmark_v1",
      "episodes": [
        {
          "episode_id": "ep_001",
          "reference_time": "2024-01-10T09:00:00Z",
          "body": "Alice joined ACME as a Backend Engineer."
        },
        {
          "episode_id": "ep_002",
          "reference_time": "2024-06-15T09:00:00Z",
          "body": "Alice was promoted to Engineering Manager at ACME."
        },
        {
          "episode_id": "ep_003",
          "reference_time": "2025-02-01T09:00:00Z",
          "body": "Alice moved from Engineering Manager to Product Lead at ACME."
        }
      ],
      "queries": [
        {
          "query_id": "q_001",
          "query": "What is Alice's current role at ACME?",
          "query_time": "2025-03-01T00:00:00Z",
          "temporal_profile": "current_state",
          "gold_answer": "Product Lead",
          "gold_fact_contains": ["Alice", "Product Lead", "ACME"],
          "negative_fact_contains": ["Backend Engineer", "Engineering Manager"],
          "expected_reason": "The Product Lead fact supersedes Alice's earlier ACME roles."
        }
      ]
    }
  ]
}

Field constraints:
- scenario_id must be unique across the generated batch.
- episode_id and query_id must be unique within each scenario.
- group_id must exactly match the requested group_id.
- reference_time and query_time must be ISO-8601 UTC strings ending in "Z".
- temporal_profile must be one of: current_state, latest_observation,
  point_in_time, timeline, semantic_fact.
- gold_fact_contains must contain keywords that identify the expected extracted
  fact after Graphiti ingestion.
- negative_fact_contains must contain plausible outdated or wrong competing
  facts for every non-semantic_fact query.
"""

TIMELINE_PATTERNS = [
    r'\bhistory\b',
    r'\btimeline\b',
    r'\bsequence\b',
    r'\bdevelopment\b',
    r'\bevolution\b',
    r'\bwhy\b',
    r'\bhow did\b',
    r'\bwhat led to\b',
    r'\bcause\b',
    r'\bformed\b',
    r'\bformation\b',
    r'\bwhat changes\b',
    r'\bchanged\b',
    r'\bchanges\b',
    r'\bafter\b',
    r'\bfollowing\b',
    r'\bprogression\b',
]

MONTH_PATTERN = r'(january|february|march|april|may|june|july|august|september|october|november|december)'

POINT_IN_TIME_PATTERNS = [
    r'\bas of\b',
    r'\bat the time\b',
    r'\bduring\b',
    r'\bin q[1-4]\b',
    r'\bon \d{4}-\d{2}-\d{2}\b',
    rf'\bon {MONTH_PATTERN} \d{{1,2}}, \d{{4}}\b',
    rf'\bon {MONTH_PATTERN} \d{{1,2}}\b',
    rf'\bas of {MONTH_PATTERN} \d{{1,2}}, \d{{4}}\b',
    rf'\bin {MONTH_PATTERN}\b',
    r'\bin \d{4}\b',
]

LATEST_PATTERNS = [
    r'\blatest\b',
    r'\bnewest\b',
    r'\bmost recent\b',
    r'\brecent update\b',
]

CURRENT_PATTERNS = [
    r'\bcurrent\b',
    r'\bcurrently\b',
    r'\bnow\b',
    r'\bstill\b',
    r'\bactive\b',
    r'\bvalid\b',
    r'\bpresent\b',
]

TOKEN_RE = re.compile(r'[a-z0-9]+')
TEXT_MATCH_STOPWORDS = {
    'a',
    'an',
    'and',
    'are',
    'as',
    'at',
    'be',
    'been',
    'being',
    'by',
    'for',
    'from',
    'in',
    'is',
    'of',
    'on',
    'or',
    'the',
    'to',
    'was',
    'were',
    'with',
}


class EpisodeSpec(BaseModel):
    episode_id: str = Field(description='Unique episode id within the scenario')
    reference_time: datetime = Field(description='ISO-8601 UTC timestamp')
    body: str = Field(description='Natural-language episode text')

    @field_validator('reference_time', mode='before')
    @classmethod
    def parse_reference_time(cls, value: Any) -> datetime:
        return parse_utc_datetime(value)

    @field_validator('body')
    @classmethod
    def validate_body(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 20:
            raise ValueError('episode body is too short')
        return value


class QuerySpec(BaseModel):
    query_id: str = Field(description='Unique query id within the scenario')
    query: str
    query_time: datetime
    temporal_profile: str
    gold_answer: str
    gold_fact_contains: list[str]
    negative_fact_contains: list[str] = Field(default_factory=list)
    expected_reason: str

    @field_validator('query_time', mode='before')
    @classmethod
    def parse_query_time(cls, value: Any) -> datetime:
        return parse_utc_datetime(value)

    @field_validator('temporal_profile')
    @classmethod
    def validate_temporal_profile(cls, value: str) -> str:
        if value not in TEMPORAL_PROFILES:
            raise ValueError(f'unsupported temporal_profile: {value}')
        return value

    @field_validator('gold_fact_contains', 'negative_fact_contains')
    @classmethod
    def validate_keyword_list(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item and item.strip()]
        return dedupe_preserving_order(cleaned)

    @field_validator('query', 'gold_answer', 'expected_reason')
    @classmethod
    def validate_non_empty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError('field must not be empty')
        return value


class ScenarioSpec(BaseModel):
    scenario_id: str
    domain: str
    group_id: str = GROUP_ID
    episodes: list[EpisodeSpec]
    queries: list[QuerySpec]


class ScenarioBatch(BaseModel):
    scenarios: list[ScenarioSpec]


class RejectedScenario(BaseModel):
    scenario_id: str | None = None
    domain: str | None = None
    rejected_reasons: list[str]
    scenario: dict[str, Any] | None = None


def parse_utc_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
    else:
        raise ValueError(f'unsupported datetime value: {value!r}')

    if dt.tzinfo is None:
        raise ValueError('datetime must be timezone-aware')
    return dt.astimezone(timezone.utc)


def dedupe_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.lower()
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def normalize_text(value: str) -> str:
    return re.sub(r'\s+', ' ', value).strip().lower()


def tokenize_for_match(value: str) -> list[str]:
    return [
        token
        for token in TOKEN_RE.findall(normalize_text(value))
        if token not in TEXT_MATCH_STOPWORDS
    ]


def text_supports_phrase(phrase: str, text: str) -> bool:
    phrase_text = normalize_text(phrase)
    full_text = normalize_text(text)
    if not phrase_text:
        return False
    if phrase_text in full_text:
        return True

    phrase_tokens = tokenize_for_match(phrase)
    text_tokens = set(tokenize_for_match(text))
    if not phrase_tokens or not text_tokens:
        return False

    matched = sum(1 for token in phrase_tokens if token in text_tokens)
    if len(phrase_tokens) <= 2:
        return matched == len(phrase_tokens)
    return matched >= max(2, len(phrase_tokens) - 1)


def answer_is_supported(answer: str, keywords: list[str], episode_text: str) -> bool:
    if text_supports_phrase(answer, episode_text):
        return True
    if any(
        text_supports_phrase(answer, keyword) or text_supports_phrase(keyword, answer)
        for keyword in keywords
    ):
        return True
    return bool(keywords) and all(
        text_supports_phrase(keyword, episode_text) for keyword in keywords
    )


def contains_any(text: str, patterns: list[str]) -> bool:
    normalized = normalize_text(text)
    return any(re.search(pattern, normalized) for pattern in patterns)


def infer_profile_from_query(query: str) -> str | None:
    """Return a strong cue-based profile, or None when the query is ambiguous."""
    if contains_any(query, TIMELINE_PATTERNS):
        return 'timeline'
    if contains_any(query, POINT_IN_TIME_PATTERNS):
        return 'point_in_time'
    if contains_any(query, LATEST_PATTERNS):
        return 'latest_observation'
    if contains_any(query, CURRENT_PATTERNS):
        return 'current_state'
    return None


def validate_scenario(scenario: ScenarioSpec) -> list[str]:
    errors: list[str] = []

    if len(scenario.episodes) < MIN_EPISODES_PER_SCENARIO:
        errors.append(f'scenario must contain at least {MIN_EPISODES_PER_SCENARIO} episodes')
    # if len(scenario.queries) < MIN_QUERIES_PER_SCENARIO:
    #     errors.append(f'scenario must contain at least {MIN_QUERIES_PER_SCENARIO} queries')

    episode_ids = [episode.episode_id for episode in scenario.episodes]
    query_ids = [query.query_id for query in scenario.queries]
    if len(set(episode_ids)) != len(episode_ids):
        errors.append('episode_id values must be unique within the scenario')
    if len(set(query_ids)) != len(query_ids):
        errors.append('query_id values must be unique within the scenario')

    episode_text = normalize_text(' '.join(episode.body for episode in scenario.episodes))
    temporal_query_count = 0

    for query in scenario.queries:
        if query.temporal_profile in TEMPORAL_PROFILES_REQUIRING_NEGATIVES:
            temporal_query_count += 1
            if not query.negative_fact_contains:
                errors.append(f'{query.query_id}: temporal query must include negative facts')

        inferred_profile = infer_profile_from_query(query.query)
        if inferred_profile is not None and inferred_profile != query.temporal_profile:
            errors.append(
                f'{query.query_id}: temporal_profile={query.temporal_profile} conflicts '
                f'with strong cue-based inference={inferred_profile}'
            )

        gold_keywords = query.gold_fact_contains
        if not query.gold_fact_contains:
            errors.append(f'{query.query_id}: gold_fact_contains must not be empty')
        elif not answer_is_supported(query.gold_answer, gold_keywords, episode_text):
            errors.append(
                f'{query.query_id}: gold_answer is not recoverable from episode text '
                'or gold_fact_contains'
            )

        if query.negative_fact_contains:
            missing_negatives = [
                item for item in query.negative_fact_contains if not text_supports_phrase(item, episode_text)
            ]
            if missing_negatives:
                errors.append(
                    f'{query.query_id}: negative facts not found in episodes: {missing_negatives}'
                )

    if temporal_query_count == 0:
        errors.append('scenario must contain at least one temporal query')

    sorted_episode_times = sorted(episode.reference_time for episode in scenario.episodes)
    if sorted_episode_times != [episode.reference_time for episode in scenario.episodes]:
        errors.append('episodes should be ordered by reference_time')

    return errors


def build_generator_messages(
    domain: str,
    scenario_count: int,
    group_id: str,
    message_cls: type[Any],
) -> list[Any]:
    system = (
        'You generate high-quality synthetic benchmark data for temporal fact reranking '
        'in a knowledge graph. Return only fictional data. Output strict JSON only.'
    )
    user = f"""
Create {scenario_count} independent scenarios for domain "{domain}".

The benchmark tests whether reranking can prefer temporally correct facts over
semantically similar but outdated facts.

{TEMPORAL_TAXONOMY_GUIDE}

{SCENARIO_BATCH_OUTPUT_TEMPLATE}

Requirements:
- group_id must be "{group_id}".
- Each scenario must have {MIN_EPISODES_PER_SCENARIO} to {MAX_EPISODES_PER_SCENARIO} timestamped episodes.
- Each scenario must have {MIN_QUERIES_PER_SCENARIO} to {MAX_QUERIES_PER_SCENARIO} queries.
- Episode bodies must be natural-language paragraphs, not triples.
- Each scenario must include at least one fact that changes over time.
- Each temporal query must include a plausible outdated wrong answer in negative_fact_contains.
- gold_fact_contains should identify the expected fact after Graphiti extracts edges.
- Use ISO-8601 UTC timestamps.
- Use fictional names, companies, products, tickets, policies, and institutions only.
- Avoid ambiguous queries with multiple valid gold answers.
- Make old and new facts semantically similar enough that semantic search could confuse them.
"""
    return [message_cls(role='system', content=system), message_cls(role='user', content=user.strip())]


def build_critic_messages(
    scenarios: list[ScenarioSpec],
    group_id: str,
    message_cls: type[Any],
) -> list[Any]:
    scenario_json = json.dumps(
        [scenario.model_dump(mode='json') for scenario in scenarios],
        ensure_ascii=False,
        indent=2,
    )
    system = (
        'You are a strict benchmark data critic. Repair only clear issues. '
        'Remove scenarios that remain ambiguous. Return strict JSON only.'
    )
    user = f"""
Review and repair these synthetic temporal reranking scenarios.

Return a ScenarioBatch containing only accepted or repaired scenarios.
Use group_id "{group_id}".

{TEMPORAL_TAXONOMY_GUIDE}

{SCENARIO_BATCH_OUTPUT_TEMPLATE}

Reject by omission if:
- dates are inconsistent,
- the gold answer is ambiguous,
- temporal_profile conflicts with the priority rules or dominant retrieval objective,
- temporal queries lack negative_fact_contains,
- gold_fact_contains cannot identify the expected answer,
- episode bodies are too unnatural for knowledge graph extraction.

Scenarios:
{scenario_json}
"""
    return [message_cls(role='system', content=system), message_cls(role='user', content=user.strip())]


def build_repair_messages(
    rejected_scenarios: list[RejectedScenario],
    group_id: str,
    message_cls: type[Any],
) -> list[Any]:
    repair_json = json.dumps(
        [
            {
                'scenario_id': item.scenario_id,
                'domain': item.domain,
                'repair_reasons': item.rejected_reasons,
                'scenario': item.scenario,
            }
            for item in rejected_scenarios
        ],
        ensure_ascii=False,
        indent=2,
    )
    system = (
        'You repair synthetic benchmark data using concrete critic or local validation errors. '
        'Preserve useful scenarios whenever they can be made unambiguous. '
        'Return strict JSON only.'
    )
    user = f"""
Repair these scenarios using the provided repair_reasons.

Return a ScenarioBatch containing repaired scenarios that should pass local
validation. Use group_id "{group_id}".

{TEMPORAL_TAXONOMY_GUIDE}

{SCENARIO_BATCH_OUTPUT_TEMPLATE}

Repair guidance:
- Prefer the smallest edit that fixes each validation error.
- If temporal_profile is semantically correct but the query lacks an explicit
  cue, rewrite the query minimally to expose the intended temporal objective.
- If a gold or negative fact failed because of wording mismatch, adjust
  gold_fact_contains or negative_fact_contains to concise phrases supported by
  the episode text, or lightly rewrite the episode body in natural language.
- Keep scenario_id stable unless the error is a duplicate scenario_id.
- Keep episode_id and query_id stable unless they are duplicated.
- Do not make episode bodies look like triples or validator hacks.
- Omit a scenario only if it remains ambiguous after repair.

Rejected scenarios with critic or validation reasons:
{repair_json}
"""
    return [message_cls(role='system', content=system), message_cls(role='user', content=user.strip())]


async def call_llm(
    client: Any,
    messages: list[Any],
    max_tokens: int,
) -> ScenarioBatch:
    response = await client.generate_response(
        messages,
        response_model=ScenarioBatch,
        max_tokens=max_tokens,
        prompt_name='synthetic_temporal_reranking_data',
    )
    return ScenarioBatch.model_validate(response)


def scenario_to_jsonl(scenario: ScenarioSpec) -> str:
    return json.dumps(scenario.model_dump(mode='json'), ensure_ascii=False)


def rejected_to_jsonl(rejected: RejectedScenario) -> str:
    return json.dumps(rejected.model_dump(mode='json'), ensure_ascii=False)


def read_scenarios_jsonl(path: Path) -> list[ScenarioSpec]:
    scenarios: list[ScenarioSpec] = []
    if not path.exists():
        return scenarios

    with path.open('r', encoding='utf-8') as file:
        for line_no, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                scenarios.append(ScenarioSpec.model_validate_json(stripped))
            except Exception as exc:  # noqa: BLE001 - CLI should report the bad line.
                raise ValueError(f'failed to parse {path}:{line_no}: {exc}') from exc
    return scenarios


def write_jsonl(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as file:
        for row in rows:
            file.write(row + '\n')


def build_stats(
    clean_scenarios: list[ScenarioSpec],
    rejected: list[RejectedScenario],
    raw_count: int,
    model: str | None = None,
) -> dict[str, Any]:
    profile_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    episode_counts: list[int] = []
    query_counts: list[int] = []

    for scenario in clean_scenarios:
        domain_counts[scenario.domain] += 1
        episode_counts.append(len(scenario.episodes))
        query_counts.append(len(scenario.queries))
        profile_counts.update(query.temporal_profile for query in scenario.queries)

    total_queries = sum(profile_counts.values())
    temporal_queries = total_queries - profile_counts.get('semantic_fact', 0)
    rejected_reasons = Counter(reason for item in rejected for reason in item.rejected_reasons)

    return {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'model': model,
        'raw_scenarios': raw_count,
        'clean_scenarios': len(clean_scenarios),
        'rejected_scenarios': len(rejected),
        'total_queries': total_queries,
        'temporal_queries': temporal_queries,
        'temporal_query_percent': round((temporal_queries / total_queries) * 100, 2)
        if total_queries
        else 0,
        'profile_counts': dict(sorted(profile_counts.items())),
        'domain_counts': dict(sorted(domain_counts.items())),
        'avg_episodes_per_scenario': round(sum(episode_counts) / len(episode_counts), 2)
        if episode_counts
        else 0,
        'avg_queries_per_scenario': round(sum(query_counts) / len(query_counts), 2)
        if query_counts
        else 0,
        'rejection_reasons': dict(rejected_reasons.most_common()),
    }


async def generate(args: argparse.Namespace) -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    from graphiti_core.llm_client import LLMConfig, OpenAIClient
    from graphiti_core.prompts.models import Message

    if not os.getenv('OPENAI_API_KEY'):
        raise SystemExit('OPENAI_API_KEY must be set before running generation.')

    output_dir = Path(args.output_dir)
    model = args.model
    domains = args.domains or DEFAULT_DOMAINS[: args.max_domains]
    client = OpenAIClient(
        config=LLMConfig(
            api_key=os.getenv('OPENAI_API_KEY'),
            model=model,
            small_model=model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    )

    raw_scenarios: list[ScenarioSpec] = []
    clean_scenarios: list[ScenarioSpec] = []
    rejected: list[RejectedScenario] = []
    seen_scenario_ids: set[str] = set()

    for domain in domains:
        print(f'Generating {args.scenarios_per_domain} scenario(s) for {domain}...')
        batch = await call_llm(
            client,
            build_generator_messages(domain, args.scenarios_per_domain, args.group_id, Message),
            args.max_tokens,
        )
        raw_scenarios.extend(batch.scenarios)

        repair_candidates: list[RejectedScenario] = []
        if args.skip_critic:
            reviewed_scenarios = batch.scenarios
        else:
            critic_batch = await call_llm(
                client,
                build_critic_messages(batch.scenarios, args.group_id, Message),
                args.max_tokens,
            )
            reviewed_scenarios = critic_batch.scenarios
            reviewed_ids = {scenario.scenario_id for scenario in reviewed_scenarios}
            for scenario in batch.scenarios:
                if scenario.scenario_id not in reviewed_ids:
                    repair_candidates.append(
                        RejectedScenario(
                            scenario_id=scenario.scenario_id,
                            domain=scenario.domain,
                            rejected_reasons=['critic_removed'],
                            scenario=scenario.model_dump(mode='json'),
                        )
                    )

        if not args.skip_validation:
            for scenario in reviewed_scenarios:
                reasons = validate_scenario(scenario)
                if scenario.scenario_id in seen_scenario_ids:
                    reasons.append('duplicate scenario_id across generated dataset')
                if reasons:
                    repair_candidates.append(
                        RejectedScenario(
                            scenario_id=scenario.scenario_id,
                            domain=scenario.domain,
                            rejected_reasons=reasons,
                            scenario=scenario.model_dump(mode='json'),
                        )
                    )
                    continue

                seen_scenario_ids.add(scenario.scenario_id)
                clean_scenarios.append(scenario)
        else:
            for scenario in reviewed_scenarios:
                if scenario.scenario_id in seen_scenario_ids:
                    rejected.append(
                        RejectedScenario(
                            scenario_id=scenario.scenario_id,
                            domain=scenario.domain,
                            rejected_reasons=['duplicate scenario_id across generated dataset'],
                            scenario=scenario.model_dump(mode='json'),
                        )
                    )
                    continue
                seen_scenario_ids.add(scenario.scenario_id)
                clean_scenarios.append(scenario)

        if repair_candidates and not args.skip_repair:
            print(f'Repairing {len(repair_candidates)} scenario(s) for {domain}...')
            repair_batch = await call_llm(
                client,
                build_repair_messages(repair_candidates, args.group_id, Message),
                args.max_tokens,
            )
            repaired_ids: set[str] = set()
            repair_failures: list[RejectedScenario] = []

            for scenario in repair_batch.scenarios:
                repaired_ids.add(scenario.scenario_id)
                reasons = [] if args.skip_validation else validate_scenario(scenario)
                if scenario.scenario_id in seen_scenario_ids:
                    reasons.append('duplicate scenario_id across generated dataset')
                if reasons:
                    repair_failures.append(
                        RejectedScenario(
                            scenario_id=scenario.scenario_id,
                            domain=scenario.domain,
                            rejected_reasons=['repair_failed', *reasons],
                            scenario=scenario.model_dump(mode='json'),
                        )
                    )
                    continue

                seen_scenario_ids.add(scenario.scenario_id)
                clean_scenarios.append(scenario)

            for item in repair_candidates:
                if item.scenario_id not in repaired_ids:
                    rejected.append(
                        RejectedScenario(
                            scenario_id=item.scenario_id,
                            domain=item.domain,
                            rejected_reasons=['repair_omitted', *item.rejected_reasons],
                            scenario=item.scenario,
                        )
                    )
            rejected.extend(repair_failures)
        else:
            rejected.extend(repair_candidates)

    write_jsonl(output_dir / RAW_FILENAME, [scenario_to_jsonl(s) for s in raw_scenarios])
    write_jsonl(output_dir / CLEAN_FILENAME, [scenario_to_jsonl(s) for s in clean_scenarios])
    write_jsonl(output_dir / REJECTED_FILENAME, [rejected_to_jsonl(item) for item in rejected])

    stats = build_stats(clean_scenarios, rejected, len(raw_scenarios), model)
    (output_dir / STATS_FILENAME).write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )

    print_summary(stats, output_dir)


def validate(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    scenarios = read_scenarios_jsonl(input_path)
    rejected: list[RejectedScenario] = []
    clean: list[ScenarioSpec] = []
    seen_ids: set[str] = set()

    for scenario in scenarios:
        reasons = validate_scenario(scenario)
        if scenario.scenario_id in seen_ids:
            reasons.append('duplicate scenario_id across dataset')
        if reasons:
            rejected.append(
                RejectedScenario(
                    scenario_id=scenario.scenario_id,
                    domain=scenario.domain,
                    rejected_reasons=reasons,
                    scenario=scenario.model_dump(mode='json'),
                )
            )
        else:
            seen_ids.add(scenario.scenario_id)
            clean.append(scenario)

    stats = build_stats(clean, rejected, len(scenarios), model=None)
    print_summary(stats, input_path.parent)
    if rejected:
        print('\nValidation failures:')
        for item in rejected[:10]:
            print(f'- {item.scenario_id}: {"; ".join(item.rejected_reasons)}')
        raise SystemExit(1)


def sample(args: argparse.Namespace) -> None:
    scenarios = read_scenarios_jsonl(Path(args.input))
    queries: list[tuple[ScenarioSpec, QuerySpec]] = [
        (scenario, query) for scenario in scenarios for query in scenario.queries
    ]
    rng = random.Random(args.seed)
    selected = rng.sample(queries, min(args.sample_size, len(queries)))

    for index, (scenario, query) in enumerate(selected, start=1):
        print(f'\n[{index}] {scenario.scenario_id} / {query.query_id}')
        print(f'Domain: {scenario.domain}')
        print(f'Profile: {query.temporal_profile}')
        print(f'Query: {query.query}')
        print(f'Query time: {query.query_time.isoformat()}')
        print(f'Gold answer: {query.gold_answer}')
        print(f'Gold contains: {query.gold_fact_contains}')
        print(f'Negative contains: {query.negative_fact_contains}')
        print(f'Reason: {query.expected_reason}')


def print_summary(stats: dict[str, Any], output_dir: Path) -> None:
    print('\nSynthetic temporal reranking dataset summary')
    print(f'Output directory: {output_dir}')
    print(f'Raw scenarios: {stats["raw_scenarios"]}')
    print(f'Clean scenarios: {stats["clean_scenarios"]}')
    print(f'Rejected scenarios: {stats["rejected_scenarios"]}')
    print(f'Total queries: {stats["total_queries"]}')
    print(f'Temporal query percent: {stats["temporal_query_percent"]}%')
    print(f'Profile counts: {stats["profile_counts"]}')
    print(f'Domain counts: {stats["domain_counts"]}')


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Generate and validate synthetic temporal reranking benchmark data.'
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    generate_parser = subparsers.add_parser('generate', help='Generate scenarios with an LLM.')
    generate_parser.add_argument('--model', default=DEFAULT_MODEL)
    generate_parser.add_argument('--temperature', type=float, default=0.2)
    generate_parser.add_argument('--max-tokens', type=int, default=12000)
    generate_parser.add_argument('--output-dir', default=str(OUTPUT_DIR))
    generate_parser.add_argument('--group-id', default=GROUP_ID)
    generate_parser.add_argument('--scenarios-per-domain', type=int, default=2)
    generate_parser.add_argument('--max-domains', type=int, default=len(DEFAULT_DOMAINS))
    generate_parser.add_argument('--domains', nargs='*')
    generate_parser.add_argument('--skip-critic', action='store_true')
    generate_parser.add_argument('--skip-repair', action='store_true')
    generate_parser.add_argument('--skip-validation', action='store_true', help='Skip local heuristic validation of generated scenarios.')

    validate_parser = subparsers.add_parser('validate', help='Validate an existing JSONL file.')
    validate_parser.add_argument('--input', default=str(OUTPUT_DIR / CLEAN_FILENAME))

    sample_parser = subparsers.add_parser('sample', help='Print random queries for manual review.')
    sample_parser.add_argument('--input', default=str(OUTPUT_DIR / CLEAN_FILENAME))
    sample_parser.add_argument('--sample-size', type=int, default=20)
    sample_parser.add_argument('--seed', type=int, default=13)

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.command == 'generate':
        asyncio.run(generate(args))
    elif args.command == 'validate':
        validate(args)
    elif args.command == 'sample':
        sample(args)
    else:
        raise SystemExit(f'Unsupported command: {args.command}')


if __name__ == '__main__':
    main()
