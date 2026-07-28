# ScalableRAG

ScalableRAG (paper link) is a high-quality agentic retrieval system for question answering over text corpora that is capable of scaling to enormous corpora by limiting ingestion costs. The two main modes are **Zero-Ingestion ScalableRAG**, which has 0 ingestion (no LLM calls at ingestion and no vector database creation); and **Limited-Ingestion ScalableRAG**, which has a vector database and a number of LLM calls at ingestion that is constant as a function of the number of documents in the corpus.

On six diverse QA corpora, both ScalableRAG variants match or beat high-ingestion RAG systems, including knowledge-graph and schema-based approaches (GraphRAG, HippoRAG2, SRAG, AutoSchemaKG), despite requiring no preprocessing cost whatsoever. Zero-Ingestion ScalableRAG wins outright on three datasets (MuSiQue, Transcripts, Hotels) and just narrowly misses being the winner on the remaining three; across all six datasets it leads the next-best system (A-RAG) by **7.36 points** in LLM-as-judge accuracy (in %, using GPT-4.1):

| System | MuSiQue | 2Wiki | Transcripts | FinanceBench | ComplexTR | Hotels | Average Across Datasets |
|--------|---------|-------|-------------|--------------|-----------|--------|--------------|
| **Limited-Ingestion ScalableRAG** | **58.91** | 86.76 | **83.50** | 75.58 | 75.57 | **73.15** | **75.58** |
| **Zero-Ingestion ScalableRAG** | **58.09** | 85.74 | **82.00** | 76.31 | 78.84 | **71.74** | **75.45** |
| A-RAG | 57.00 | **87.05** | 78.50 | **77.41** | 77.09 | 31.52 | 68.09 |
| HippoRAG2 | 50.43 | 73.99 | 76.00 | 61.08 | **81.34** | 32.97 | 62.63 |
| SRAG | 3.33 | 14.75 | 74.00 | 18.25 | 8.33 | 35.51 | 25.70 |
| GraphRAG | 15.47 | 35.55 | 35.50 | 23.54 | 13.75 | 1.45 | 20.88 |
| Vanilla RAG (top-100) | 38.49 | 43.80 | 71.00 | 32.09 | 77.59 | 30.80 | 48.96 |
| AutoSchemaKG | — | — | — | — | 80.40 | 24.64 | — |

ScalableRAG works by mimicking the logic an agent may apply to a structured data system such as a knowledge graph, but entirely at inference time. In order to do this it creates a workspace of persistent sets that it is able to interact with. This includes sets of filenames, sets of values, sets of strings, sets of dates, and sets of lists. The agent can create new sets from other sets through a series of tools: sets of files can be created through regex hits, sets of values extracted through regex value extraction, and so on. Each time a set is created, the agent sees a series of statistics; as well as scope checks to make sure it had created the correct set. Once the agent has created the relevant sets and validated them, aggregations are done automatically and shown to the agent. It is also capable of reading through docs in the created sets, doing set algebra, etc. It is this interaction of the agent with a workspace of agentic sets that makes ScalableRAG capable of inference-time aggregative logic as if it had access to a knowledge graph. See (paper link) for details.

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

## 1. Add your documents

Plain text files in one folder:

```
my_corpus/
  doc_001.txt
  doc_002.txt
  ...
```

Useful filenames help (the agent can filter on them), but not necessary.

---

## 2. Build an index

### Zero-Ingestion (recommended starting point)

This copies your files into an index directory the agent reads from.

```bash
python scripts/ingest.py \
  --source /path/to/my_corpus \
  --store  /path/to/my_index \
  --no-embeddings
```

### Limited-Ingestion

Build embeddings at ingest (use the same provider/model when you query or eval):

```bash
# Local embeddings (free, default model)
python scripts/ingest.py \
  --source /path/to/my_corpus \
  --store  /path/to/my_index

# Or OpenAI embeddings (matches paper limited-ingestion setup)
python scripts/ingest.py \
  --source /path/to/my_corpus \
  --store  /path/to/my_index \
  --embed-provider openai \
  --embed-model text-embedding-3-small
```

You get a directory like:

```
my_index/
  docs/          # copied text + metadata
  embeddings/    # only for limited-ingestion
```

---

## 3. Ask a question

```bash
# Zero-Ingestion
python scripts/query.py \
  --regime zero \
  --store /path/to/my_index \
  --question "How many documents mention clause 4.2?"

# Limited-Ingestion (index must have embeddings from step 2)
python scripts/query.py \
  --regime limited \
  --store /path/to/my_index \
  --question "How many documents mention clause 4.2?"
```

Add `--quiet` to print only the final answer.

---

## 4. Run a benchmark (optional)

Questions file: JSONL with `question_id`, `question`, and `golden_answer` per line.

```bash
# Zero-Ingestion
python scripts/eval.py \
  --regime zero \
  --store /path/to/my_index \
  --questions /path/to/questions.jsonl \
  --output /path/to/results.jsonl

# Limited-Ingestion
python scripts/eval.py \
  --regime limited \
  --store /path/to/my_index \
  --questions /path/to/questions.jsonl \
  --output /path/to/results.jsonl
```

**Outputs:** `results.jsonl` (one line per question: `golden_answer`, `answer_raw`, compact trace) and `results_traces.jsonl` (full step-by-step tool trace).

Eval resumes automatically if you re-run the same command; it skips questions already in the output file. Use `--start`, `--limit`, or `--max-steps` if you need finer control.

Both regimes enable **aggregation tools** and **Branch-and-Select** by default. `--regime limited` additionally turns on **embeddings** (semantic search) and **pattern discovery/curation** during the run. You can also pre-build patterns offline:

```bash
python scripts/discover_patterns.py \
  --store /path/to/my_index \
  --output /path/to/my_index/patterns.json
python scripts/eval.py --regime limited --store /path/to/my_index \
  --questions questions.jsonl --output results.jsonl \
  --patterns /path/to/my_index/patterns.json
```

---

## 5. Score results

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

| | `--regime zero` | `--regime limited` |
|---|-----------------|---------------------|
| Command | `ingest.py ... --no-embeddings` | `ingest.py ...` (with `--embed-provider openai --embed-model text-embedding-3-small` to match the preset defaults) |

**Query / eval**

| Flag | `--regime zero` | `--regime limited` |
|------|-----------------|---------------------|
| `--embeddings` | off (`--no-embeddings`) | on |
| `--discover-patterns` | off | on |
| `--curate-patterns` | off | on |
| `--agg-tools` | on | on |
| `--branch-select` | on | on |
| Default embed model | — | `openai` / `text-embedding-3-small` |

Use the same regime (or equivalent flags) at ingest, query, and eval time. If the index has no embeddings, query/eval must use `--regime zero` or `--no-embeddings`.

**Finer control:** skip `--regime` and set flags directly, for example:

```bash
python scripts/eval.py \
  --store /path/to/my_index \
  --questions questions.jsonl \
  --output results.jsonl \
  --no-embeddings \
  --agg-tools \
  --no-discover-patterns \
  --patterns /path/to/my_index/patterns.json
```

Same boolean flags work on `query.py`. Ingest uses `--embeddings` / `--no-embeddings` plus `--embed-provider` and `--embed-model`.

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
| `scripts/ingest.py` | Build index from a document folder |
| `scripts/query.py` | One question, interactive |
| `scripts/eval.py` | Batch run over JSONL questions |
| `scripts/judge.py` | Score eval output |
| `scripts/discover_patterns.py` | Pre-build patterns for limited-ingestion |
| `scripts/build_embeddings.py` | Add embeddings to an existing index |

## Legal Notices

Every product name, logo, brand, and service name referenced in this software belongs to its respective owner and appears solely for identification. Cohesity is a registered trademark of Cohesity, Inc., and the open-source license agreement grants no right to use Cohesity's trademark.

Disclaimer of Warranty: Cohesity furnishes no support or maintenance services for this software. Cohesity supplies the software (and each contributor supplies its respective contribution) on an "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, either expressed or implied, including but not limited to any warranties of TITLE, NONINFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A PARTICULAR PURPOSE. Deciding whether it is suitable to use or redistribute the software rests entirely with you. You bear any risks tied to your exercise of the permissions granted under the open-source license.