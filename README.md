# ScalableRAG

ScalableRAG ([paper](https://arxiv.org/pdf/2607.25135)) is an agentic retrieval system for question answering that closes the **capability–cost gap** in RAG: lightweight retrievers find local evidence but expose no operands for exact corpus-level operations (counts, intersections, aggregations), while knowledge-graph and schema-first systems obtain those operations only through heavy offline ingestion. ScalableRAG materializes structure **at query time** in a persistent **typed-set workspace**—named document and value sets combined through set algebra, typed extraction, aggregation, and diagnostics—so deterministic corpus operations do not require a pre-built graph.

**What you get:** one agentic stack that can run **exact corpus operations** (exhaustive narrowing, intersection, counting, aggregation) **and** match top agentic RAG on conventional QA—without building a knowledge graph or running LLM extraction over every document. **Zero-Ingestion ScalableRAG** starts from raw text only: no preprocessing, not even a vector index. It materializes named document and value sets at query time, composes them with set algebra and typed extraction, and beats every evaluated baseline on aggregative, corpus-operation workloads while matching the leading agentic baseline on ordinary multi-hop QA. **Limited-Ingestion ScalableRAG** carries that same workspace to **million-document** corpora with a lightweight SQLite FTS5 index (optional dense embeddings via `--with-embeddings`): indexed literals shrink large parent sets before exact regex on small sets, with constant-size pattern discovery at query time. In the [paper](https://arxiv.org/pdf/2607.25135) it is the **only evaluated system** to complete **5M** documents within the measurement budget with GPT-4.1 judge score above **80**, holding **13–16 s** inference and under **20 minutes** of ingestion—**minimal ingestion, conventional-QA parity, exact corpus operations, and million-document viability in a single architecture; no evaluated alternative provides all four.**

Evaluated against GraphRAG, HippoRAG2, SRAG, AutoSchemaKG, A-RAG, and others ([paper](https://arxiv.org/pdf/2607.25135), Table 1): ScalableRAG is built for questions that need **exact corpus operations**—exhaustive filtering, intersection, counting, and aggregation over **complete** document sets, not just top‑k passage retrieval. On those **aggregative and corpus-operation** workloads it **beats every evaluated baseline** (agentic, graph, and schema): gaps of **40+** judge points over agentic RAG and **36+** over the leading relational schema baseline on the purely aggregative partition, while ingestion-heavy structured systems that exist for aggregation plateau near **35** judge despite offline graphs and extraction. On **Transcripts**—a mixed local-lookup and corpus-operation benchmark—both variants lead (**83.50** Limited, **82.00** Zero; **+3.50** over A-RAG for Zero). On ordinary multi-hop and document-grounded QA, the **same** Zero-Ingestion system **matches** the strongest agentic baseline (**74.75** vs **74.64** mean judge on the four conventional benchmarks) with **no** preprocessing—not even a vector index. **Limited-Ingestion** tops macro judge overall (**73.70**) on the seven benchmark partitions below. Table 1 is GPT-4.1 judge accuracy (%); Table 2 is the scale study.

| System | MuSiQue | 2Wiki | Transcripts | FinanceBench | ComplexTR | Hotels-50 | Hotels-350 | Macro avg. |
|--------|---------|-------|-------------|--------------|-----------|-----------|------------|------------|
| **Limited-Ingestion ScalableRAG** | **58.91** | 86.76 | **83.50** | 75.58 | 75.57 | **73.15** | 62.44 | **73.70** |
| **Zero-Ingestion ScalableRAG** | **58.09** | 85.74 | **82.00** | 76.31 | 78.84 | **71.74** | 55.96 | **72.67** |
| A-RAG | 57.00 | **87.05** | 78.50 | **77.41** | 77.09 | 31.52 | 22.02 | 61.51 |
| HippoRAG2 | 50.43 | 73.99 | 76.00 | 61.08 | **81.34** | 32.97 | 19.69 | 56.50 |
| SRAG | 3.33 | 14.75 | 74.00 | 18.25 | 8.33 | 35.51 | 46.63 | 28.69 |
| GraphRAG | 15.47 | 35.55 | 35.50 | 23.54 | 13.75 | 1.45 | 1.04 | 18.04 |
| Vanilla RAG (top-100) | 38.49 | 43.80 | 71.00 | 32.09 | 77.59 | 30.80 | 13.99 | 43.97 |
| AutoSchemaKG | — | — | — | — | 80.40 | 24.64 | 22.80 | — |

Macro avg. is the **macro** GPT-4.1 judge aggregate over all seven partitions (Table 1 in the [paper](https://arxiv.org/pdf/2607.25135)). For the systems above it equals the unweighted mean of the seven per-dataset GPT-4.1 judge scores in each row (the table also reports a separate query-weighted aggregate in the paper).

**Scale (Table 2).** On nested 2Wiki+Wikipedia corpora (100 fixed questions), the usual tradeoff shows up in sharp form: **A-RAG** is strongest on judge score at **100K–500K**, but index cost grows with the corpus—latency goes from **28 s** to **213 s** per question at **1M** while judge score falls to **80.0**. **Limited-Ingestion** targets a different point on the quality–cost curve: lightweight FTS indexing plus the same typed-set workspace yields **~13–16 s** inference from **100K through 5M**, and at **1M** it reaches **84.5** judge vs **80.0** for A-RAG at **~16×** lower latency and **~0.5%** of A-RAG’s ingestion time (paper). **Zero-Ingestion** avoids indexing entirely but pays repeated full-corpus scans online (**515 s** at 500K, **1076 s** at 1M). At **5M**, within the paper’s 48-hour budget, Limited-Ingestion is the only system above **80** judge; dense Vanilla RAG finishes quickly but stays near **16** judge. Cells below are GPT-4.1 judge (%) and mean inference seconds per question (judging excluded); see the [paper](https://arxiv.org/pdf/2607.25135) for Gemini scores, EM/F1, ingestion, and confidence intervals.

| System | 100K judge | 100K infer (s) | 500K judge | 500K infer (s) | 1M judge | 1M infer (s) | 5M judge | 5M infer (s) |
|--------|------------|----------------|------------|----------------|----------|--------------|----------|--------------|
| **Limited-Ingestion ScalableRAG** | 80.50 | **16.45** | 79.00 | **13.07** | **84.50** | **13.53** | 81.50 | **13.64** |
| **Zero-Ingestion ScalableRAG** | 80.50 | 100.58 | 82.50 | 515.12 | 82.50 | 1076.25 | —† | 4746.79 |
| Vanilla RAG (top-100) | 28.50 | 5.72 | 21.00 | 8.20 | 24.00 | 6.64 | 16.30 | 10.58 |
| A-RAG | **88.50** | 27.93 | **84.00** | 104.25 | 80.00 | 213.31 | —† | — |

† Did not finish within the 48-hour cutoff at 5M (Zero-Ingestion and A-RAG). A-RAG’s 5M index build also did not finish in time.

Together, the tables show a single architecture at an operating point the baselines do not jointly occupy: **agentic-grade** conventional QA, **dominant** performance whenever the task is aggregative or corpus-level, and **million-document** access with bounded latency once a lightweight index is added—without paying per-document LLM ingestion or giving up named sets as first-class operands.

Concretely, the agent maintains a registry of named document sets and typed value sets (numbers, dates, lists, and strings). Tools create sets from regex or indexed literals, extract values, intersect and union sets, and run aggregations with parent-relative diagnostics that flag implausible narrowing. That design is what lets ScalableRAG match strong conventional QA without giving up exact corpus-level reasoning. See the [paper](https://arxiv.org/pdf/2607.25135) for the full method and ablations.

---

## Install

```bash
pip install -r requirements.txt
```

Direct dependencies are pinned for reproducible installs; transitive packages (e.g. PyTorch) are resolved by pip at install time.

Set API keys via environment variables:

```bash
# OpenAI (default)
export OPENAI_API_KEY=sk-...
export SR_LLM_PROVIDER=openai
export SR_LLM_MODEL=gpt-4.1

# Azure OpenAI — same OPENAI_API_KEY env var holds your Azure API key
export OPENAI_API_KEY=...
export SR_LLM_PROVIDER=azure
export SR_AZURE_ENDPOINT=https://your-resource.openai.azure.com
export SR_LLM_MODEL=your-deployment-name

# Gemini
export GEMINI_KEY=...
export SR_LLM_PROVIDER=gemini
```

OpenAI/Azure embeddings (`--embed-provider openai` or `azure`) also use `OPENAI_API_KEY`.

For a local GPU server instead, see "Other LLM backends" at the bottom.

---

## 1. Prepare a store (Zero-Ingestion)

**Zero-Ingestion** means no LLM calls and no vector database at prepare time. Put plain `.txt` files in one folder and run `ingest.py` with `--no-lexical-index`: it **copies your corpus into a store directory** (plus lightweight metadata) in the layout the agent reads from. Despite the script name, this step is file staging—not knowledge-graph or embedding ingestion.

```bash
python scripts/ingest.py \
  --source /path/to/my_corpus \
  --store  /path/to/my_store \
  --no-lexical-index
```

Your source folder looks like:

```
my_corpus/
  doc_001.txt
  doc_002.txt
  ...
```

Useful filenames help (the agent can filter on them), but are not required.

The store directory looks like:

```
my_store/
  docs/          # copied text + metadata
```

### Limited-Ingestion (optional)

Same copy step, **plus** a lexical index (built by default). Add vector embeddings only if you want semantic search (`--with-embeddings` at ingest and again at query/eval):

```bash
# Lexical index only (default limited-ingestion store)
python scripts/ingest.py \
  --source /path/to/my_corpus \
  --store  /path/to/my_store

# Optional: also build embeddings (use the same provider/model at query/eval)
python scripts/ingest.py \
  --source /path/to/my_corpus \
  --store  /path/to/my_store \
  --with-embeddings \
  --embed-provider openai \
  --embed-model text-embedding-3-small
```

Limited-ingestion adds:

```
my_store/
  docs/
  lexical.sqlite3 # local SQLite FTS5 trigram index for scalable literal filtering
  embeddings/     # optional; only with --with-embeddings at ingest
```

The lexical index is embedded and serverless: it uses SQLite FTS5 from
Python's standard library, with no external vector or search service. Ingestion
builds it by default for Limited-Ingestion stores. To add or rebuild it on an
existing store:

```bash
python scripts/build_lexical_index.py --store /path/to/my_store
```

---

## 2. Ask a question

```bash
# Zero-Ingestion
python scripts/query.py \
  --regime zero \
  --store /path/to/my_store \
  --question "How many documents mention clause 4.2?"

# Limited-Ingestion (store needs lexical.sqlite3 from step 1)
python scripts/query.py \
  --regime limited \
  --store /path/to/my_store \
  --question "How many documents mention clause 4.2?"

# Limited-Ingestion with semantic search (store must include embeddings)
python scripts/query.py \
  --regime limited \
  --with-embeddings \
  --store /path/to/my_store \
  --question "How many documents mention clause 4.2?"
```

Limited-Ingestion uses two filtering tiers. On a large active set, literal
keywords and literal alternations are served by the local index; arbitrary
regex is enabled after the active set falls below both 1,000 documents and
10 MB. This keeps corpus-wide inference work bounded while preserving exact
regex behavior on the narrowed set. Override the thresholds with
`--tier-max-docs` and `--tier-max-bytes`, or use `--no-tiered-filters` for the
legacy unrestricted behavior.

Add `--quiet` to print only the final answer.

---

## 3. Run a benchmark (optional)

Questions file: JSONL with `question_id`, `question`, and `golden_answer` per line.

```bash
# Zero-Ingestion
python scripts/eval.py \
  --regime zero \
  --store /path/to/my_store \
  --questions /path/to/questions.jsonl \
  --output /path/to/results.jsonl

# Limited-Ingestion
python scripts/eval.py \
  --regime limited \
  --store /path/to/my_store \
  --questions /path/to/questions.jsonl \
  --output /path/to/results.jsonl
```

**Outputs:** `results.jsonl` (one line per question: `golden_answer`, `answer_raw`, compact trace) and `results_traces.jsonl` (full step-by-step tool trace).

Eval resumes automatically if you re-run the same command; it skips questions already in the output file. Use `--start`, `--limit`, or `--max-steps` if you need finer control.

Both regimes enable **aggregation tools** and **Branch-and-Select** by default. `--regime limited` additionally turns on **tiered lexical filtering** and **pattern discovery/curation** during the run (embeddings only if you pass `--with-embeddings`). You can also pre-build patterns offline:

```bash
python scripts/discover_patterns.py \
  --store /path/to/my_store \
  --output /path/to/my_store/patterns.json
python scripts/eval.py --regime limited --store /path/to/my_store \
  --questions questions.jsonl --output results.jsonl \
  --patterns /path/to/my_store/patterns.json
```

---

## 4. Score results

After eval, run LLM-as-judge scoring:

```bash
python scripts/judge.py \
  --in-jsonl /path/to/results.jsonl \
  --out-dir /path/to/judge_out \
  --llm-provider openai \
  --llm-model gpt-4.1
```

Uses `SR_LLM_PROVIDER` and `SR_LLM_MODEL` when `--llm-provider` / `--llm-model` are omitted (same as query/eval). Azure: set `SR_LLM_PROVIDER=azure`, `SR_AZURE_ENDPOINT`, and `OPENAI_API_KEY`.

Writes `{stem}_judged.jsonl` (adds `answer_extracted` and a `judge` block with verdict, score, and SQuAD EM/F1) and `summary_{stem}_judged.json` under `--out-dir`.

---

## Zero-Ingestion vs Limited-Ingestion

`--regime` is a preset. **`--regime zero`** and **`--regime limited`** set the flags below; anything you pass explicitly overrides the preset.

**Ingest**

| | Zero-ingestion store | Limited-ingestion store |
|---|----------------------|-------------------------|
| Command | `ingest.py ... --no-lexical-index` | `ingest.py ...` (add `--with-embeddings` for vectors) |

**Query / eval**

| Flag | `--regime zero` | `--regime limited` |
|------|-----------------|---------------------|
| `--with-embeddings` | off | off (pass flag to enable) |
| `--discover-patterns` | off | on |
| `--curate-patterns` | off | on |
| `--tiered-filters` | off | on (requires `lexical.sqlite3`) |
| `--agg-tools` | on | on |
| `--branch-select` | on | on |
| Embed defaults when `--with-embeddings` | — | `openai` / `text-embedding-3-small` |

Use the same regime (or equivalent flags) at prepare, query, and eval time. Limited-ingestion requires a store with `lexical.sqlite3`; use `--with-embeddings` only when the store has embeddings.

**Finer control:** skip `--regime` and set flags directly, for example:

```bash
python scripts/eval.py \
  --store /path/to/my_store \
  --questions questions.jsonl \
  --output results.jsonl \
  --tiered-filters \
  --agg-tools \
  --no-discover-patterns \
  --patterns /path/to/my_store/patterns.json
```

Same boolean flags work on `query.py`. Ingest uses `--with-embeddings` (opt-in) plus `--embed-provider` and `--embed-model`.

---

## Other LLM backends

**vLLM (local Llama):** start a server, then:

```bash
export SR_LLM_PROVIDER=vllm
export SR_VLLM_MODEL=meta-llama/Llama-3.3-70B-Instruct
export SR_VLLM_BASE_URL=http://127.0.0.1:8000/v1

python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.3-70B-Instruct --host 127.0.0.1 --port 8000
```

**Azure OpenAI:** set `SR_LLM_PROVIDER=azure`, `SR_AZURE_ENDPOINT`, `OPENAI_API_KEY`, and use your deployment name as `SR_LLM_MODEL`.

**Local HTTP endpoints:** Ollama and vLLM defaults use loopback (`127.0.0.1`) over HTTP, which is appropriate for same-machine inference. If you point `SR_OLLAMA_URL` or `SR_VLLM_BASE_URL` at a remote host, use HTTPS or a trusted private network — plain HTTP sends prompts and model output in cleartext.

Provider details live in `scalablerag/llm/factory.py`; regime and tool flags in `scalablerag/run_config.py`.

---

## Scripts

| Script | What it does |
|--------|----------------|
| `scripts/ingest.py` | Stage a corpus into a store (copy docs; optional embeddings) |
| `scripts/query.py` | One question, interactive |
| `scripts/eval.py` | Batch run over JSONL questions |
| `scripts/judge.py` | Score eval output |
| `scripts/discover_patterns.py` | Pre-build patterns for limited-ingestion |
| `scripts/build_embeddings.py` | Add embeddings to an existing store |
| `scripts/build_lexical_index.py` | Build/rebuild the local SQLite literal index |

## Legal Notices

Every product name, logo, brand, and service name referenced in this software belongs to its respective owner and appears solely for identification. Cohesity is a registered trademark of Cohesity, Inc., and the open-source license agreement grants no right to use Cohesity's trademark.

Disclaimer of Warranty: Cohesity furnishes no support or maintenance services for this software. Cohesity supplies the software (and each contributor supplies its respective contribution) on an "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, either expressed or implied, including but not limited to any warranties of TITLE, NONINFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A PARTICULAR PURPOSE. Deciding whether it is suitable to use or redistribute the software rests entirely with you. You bear any risks tied to your exercise of the permissions granted under the open-source license.
