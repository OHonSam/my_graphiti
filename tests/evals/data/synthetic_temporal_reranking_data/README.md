# Synthetic Temporal Reranking Benchmark

This directory contains the data-generation pipeline for a controlled Graphiti
temporal reranking benchmark.

The benchmark is designed to test whether reranking can prefer temporally
correct facts over semantically similar but outdated facts.

## Temporal Taxonomy

`temporal_taxonomy` is the full category system. Each generated query receives
one `temporal_profile`.

V1 profiles:

- `current_state`: asks what is true, active, or current at `query_time`.
- `latest_observation`: asks for the latest or most recent known update.
- `point_in_time`: asks what was true at an explicit date, time, or period.
- `timeline`: asks for sequence, history, development, cause, or why/how.
- `semantic_fact`: no temporal ranking objective.

Future extensions:

- `cumulative_set`: asks for all accumulated facts over time.
- `frequency_aggregate`: asks for most frequent or most mentioned facts.

## Files

Generated files:

```text
raw_temporal_scenarios_v1.jsonl
temporal_scenarios_v1.jsonl
rejected_scenarios_v1.jsonl
dataset_stats_v1.json
```

Each JSONL row is one scenario with timestamped episodes and labeled queries.

Gold Graphiti edge UUIDs are not generated here. They should be resolved after
Graphiti ingests each scenario and extracts `EntityEdge` facts.

## Generate Data

Set `OPENAI_API_KEY` in the repo `.env` or current shell.

Use `gpt-4o` first to inspect quality and cost:

```powershell
uv run python tests/evals/data/synthetic_temporal_reranking_benchmark/main.py generate --model gpt-4o
```

Generate a smaller smoke-test batch:

```powershell
uv run python tests/evals/data/synthetic_temporal_reranking_benchmark/main.py generate --model gpt-4o --max-domains 2 --scenarios-per-domain 1
```

Skip the critic pass only for debugging:

```powershell
uv run python tests/evals/data/synthetic_temporal_reranking_benchmark/main.py generate --model gpt-4o --skip-critic
```

## Validate Existing Data

```powershell
uv run python tests/evals/data/synthetic_temporal_reranking_benchmark/main.py validate
```

Validate a custom file:

```powershell
uv run python tests/evals/data/synthetic_temporal_reranking_benchmark/main.py validate --input path/to/file.jsonl
```

## Manual Spot Check

Print 20 random queries:

```powershell
uv run python tests/evals/data/synthetic_temporal_reranking_benchmark/main.py sample
```

Print a smaller sample:

```powershell
uv run python tests/evals/data/synthetic_temporal_reranking_benchmark/main.py sample --sample-size 5
```

## Quality Gates

The cleaned dataset is ready for Graphiti ingestion when:

- at least 80 valid queries exist,
- at least 60% are temporal profiles,
- every temporal query has plausible outdated negative facts,
- all local validators pass,
- manual spot check finds no serious ambiguity,
- episodes are natural-language text suitable for Graphiti extraction.

## Suggested Workflow

1. Generate a small smoke batch with `--max-domains 2 --scenarios-per-domain 1`.
2. Run `validate`.
3. Run `sample --sample-size 10`.
4. If quality looks good, generate the full batch.
5. Resolve gold edge UUIDs only after Graphiti ingestion.

## Baseline Reranker Evaluation

After generating and spot-checking `temporal_scenarios_v1.jsonl`, run the first baseline by ingesting the cleaned scenarios into Graphiti and comparing existing edge rerankers.

For Neo4j Aura, keep `NEO4J_DATABASE` set to your Aura database name. The runner uses that as the Graphiti `group_id` by default so Graphiti does not try to switch to a non-existent database named `temporal_benchmark_v1`.

Run ingestion and evaluation together:

```powershell
uv run python tests/evals/data/synthetic_temporal_reranking_data/baseline_rerankers.py run --reset-group --rerankers edge_rrf edge_mmr edge_episode_mentions edge_cross_encoder --cross-encoder-provider openai
```

Or run them as two steps:

```powershell
uv run python tests/evals/data/synthetic_temporal_reranking_data/baseline_rerankers.py ingest --reset-group
uv run python tests/evals/data/synthetic_temporal_reranking_data/baseline_rerankers.py evaluate --rerankers edge_rrf edge_mmr edge_episode_mentions edge_cross_encoder --cross-encoder-provider openai
```

Outputs:

```text
baseline_reranker_results_v1.jsonl
baseline_reranker_summary_v1.json
```

The V1 evaluator reports `hit_at_1`, `hit_at_3`, `hit_at_5`, `mrr`, average latency, and `negative_above_gold_rate`. Scoring is keyword-based using `gold_fact_contains` and `negative_fact_contains`; edge UUID labels can be added after Graphiti extraction is stable.

To compare cross-encoder backends, keep the ingested graph fixed and rerun only `evaluate`:

```powershell
uv run python tests/evals/data/synthetic_temporal_reranking_data/baseline_rerankers.py evaluate --rerankers edge_cross_encoder --cross-encoder-provider openai --results tests/evals/data/synthetic_temporal_reranking_data/baseline_openai_results_v1.jsonl --summary tests/evals/data/synthetic_temporal_reranking_data/baseline_openai_summary_v1.json
uv run python tests/evals/data/synthetic_temporal_reranking_data/baseline_rerankers.py evaluate --rerankers edge_cross_encoder --cross-encoder-provider bge --results tests/evals/data/synthetic_temporal_reranking_data/baseline_bge_results_v1.jsonl --summary tests/evals/data/synthetic_temporal_reranking_data/baseline_bge_summary_v1.json
```

`bge` requires the `sentence-transformers` extra. `gemini` requires the Gemini dependency and `GEMINI_API_KEY`.
