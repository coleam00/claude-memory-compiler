# AGENTS.md - Personal Knowledge Base Schema

> Adapted from [Andrej Karpathy's LLM Knowledge Base](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) architecture.
> Instead of ingesting external articles, this system compiles knowledge from your own AI conversations.

## The Compiler Analogy

```text
records/captures/ = source code   (Capture Records - one per logical session)
LLM               = compiler      (extracts and organizes knowledge)
knowledge/        = executable    (structured, queryable knowledge base)
lint              = test suite    (health checks for consistency)
queries           = runtime       (using the knowledge)
```

You don't manually organize your knowledge. You have conversations, and the LLM handles the synthesis, cross-referencing, and maintenance.

---

## Architecture

### Layer 1: `records/captures/YYYY-MM-DD/` - Capture Records (Immutable Source)

Each logical conversation session becomes exactly one direct **Capture Record**: a markdown file named `<day>--direct--<project-slug>--<identity>.md`. The write path is fixed:

1. A SessionEnd or PreCompact hook reads the transcript locally and extracts conversation context. The hook makes no API calls.
2. The hook spawns `flush.py` as a detached background process, passing the context file path, session id, and source cwd.
3. flush.py makes one Claude Agent SDK call to decide what is worth saving, then writes exactly one direct Capture Record under `records/captures/<day>/`.

Nothing appends to `daily/*.md`. Hooks never write records themselves; only flush.py does.

Direct records carry this frontmatter:

```yaml
---
record_type: "capture_record"
record_id: "direct:[\"YYYY-MM-DD\",\"<source_cwd>\",\"<identity>\"]"
record_day: YYYY-MM-DD
capture_kind: memory_flush
capture_status: narrative | no_op | failure
source_timestamp: "2026-08-25T09:30:00-05:00"
source_sid: "<Claude Code session id>"
source_project: "</path/to/source/cwd>"
payload_sha256: "<sha256 of the body below frontmatter>"
record_origin: direct
tags: ["memory-compiler/capture-record"]
---
```

Identity is the session id when one exists, otherwise the first 16 hex characters of `payload_sha256`. The record id encodes `(record_day, source_project, identity)` and the file name mirrors it, so overlapping events for the same day, project, and session overwrite the same path instead of piling up duplicates. PreCompact and SessionEnd for one session converge on a single record this way. Writes take an interprocess lock and replace files atomically, so concurrent flushes cannot interleave or corrupt a record.

Materialized legacy notes use an older field set such as `source_file`, `source_aggregate_sha256`, and `captured_at`, and never carry `record_origin: direct`. That field is how tooling tells direct records apart from projected ones.

`daily/*.md` aggregates are read-only legacy inputs. They remain readable for history, but nothing appends to them and compilation never selects them.

### Layer 2: `raw/` - Raw Inputs

Unprocessed Markdown notes dropped into the vault-root `raw/` directory (one level above this repo) are first-class compile inputs. See "Raw Input Handling" under compile.py for the same-day and unprocessed rules.

### Layer 3: `knowledge/` - Compiled Knowledge (LLM-Owned)

The LLM owns this directory entirely. Humans read it but rarely edit it directly.

```text
knowledge/
├── index.md              # Master catalog - every article with one-line summary
├── log.md                # Append-only chronological build log
├── concepts/             # Atomic knowledge articles
├── connections/          # Cross-cutting insights linking 2+ concepts
└── qa/                   # Filed query answers

### Layer 4: This File (AGENTS.md)

The schema that tells the LLM how to compile and maintain the knowledge base. This is the "compiler specification."

---

## Structural Files

### `knowledge/index.md` - Master Catalog

A table listing every knowledge article. This is the primary retrieval mechanism - the LLM reads this FIRST when answering any query or compiling a record, then selects relevant articles to read in full.

Format:

```markdown
# Knowledge Base Index

| Article | Summary | Compiled From | Updated |
|---------|---------|---------------|---------|
| [[concepts/supabase-auth]] | Row-level security patterns and JWT gotchas | capture 2026-04-02 | 2026-04-02 |
| [[connections/auth-and-webhooks]] | Token verification shared across auth and webhooks | captures 2026-04-02, 2026-04-04 | 2026-04-04 |
```

### `knowledge/log.md` - Build Log

Append-only chronological record of every compile, query, and lint operation:

```markdown
## [2026-04-01T14:30:00] compile | Capture Record 2026-04-01--direct--myapp--a1b2c3d4e5f6
- Source: records/captures/2026-04-01/2026-04-01--direct--myapp--a1b2c3d4e5f6.md
- Articles created: [[concepts/nextjs-project-structure]], [[concepts/tailwind-setup]]
- Articles updated: (none)
```

---

## Article Formats

### Concept Articles (`knowledge/concepts/`)

One article per atomic piece of knowledge. These are facts, patterns, decisions, preferences, and lessons extracted from your conversations.

```markdown
---
title: "Concept Name"
aliases: [alternate-name, abbreviation]
tags: [domain, topic]
summary: "1-2 sentence overview that conveys what this concept is and why it matters. This is the same content as the lead paragraph below, duplicated into frontmatter so Bases/Dataview can query it without parsing the body."
sources:
  - "records/captures/2026-04-01/<capture-file>.md"
created: 2026-04-01
updated: 2026-04-03
---

# Concept Name

[2-4 sentence core explanation, matching `summary:` in spirit]

## Key Points

- [Bullet points, each self-contained]

## Details

[Deeper explanation, encyclopedia-style paragraphs]

## Related Concepts

- [[concepts/related-concept]] - How it connects
```

### Connection Articles (`knowledge/connections/`)

Cross-cutting synthesis linking 2+ concepts. Created when a conversation reveals a non-obvious relationship.

```markdown
---
title: "Connection: X and Y"
summary: "1-2 sentence overview of the non-obvious relationship between X and Y."
connects:
  - "concepts/concept-x"
  - "concepts/concept-y"
sources:
  - "records/captures/2026-04-04/<capture-file>.md"
created: 2026-04-04
updated: 2026-04-04
---

# Connection: X and Y

## The Connection

[What links these concepts]

## Key Insight

[The non-obvious relationship discovered]

## Related Concepts

- [[concepts/concept-x]]
- [[concepts/concept-y]]
```

### Q&A Articles (`knowledge/qa/`)

Filed answers from queries. Every complex question answered by the system can be permanently stored, making future queries smarter.

```markdown
---
title: "Q: Original Question"
question: "The exact question asked"
consulted:
  - "concepts/article-1"
  - "concepts/article-2"
filed: 2026-04-05
---

# Q: Original Question

## Answer

[The synthesized answer with [[wikilinks]] to sources]

## Follow-Up Questions

- What about edge case X?
```

---

## Core Operations

### 1. Compile (Capture Records -> knowledge/)

`compile.py` runs one OMP agent per pending substantive Capture Record - invocation, model balancing, locking, and state updates are specified under **compile.py** in Script Details. Per record, the agent should:

1. Read the named Capture Record (and its attached Raw Inputs, if any)
2. Read `knowledge/index.md` to understand current knowledge state
3. Read existing articles that may need updating
4. For each piece of knowledge found in the record:
   - If an existing concept article covers this topic: UPDATE it with new information, add the record as a source
   - If it's a new topic: CREATE a new `concepts/` article
5. If the record reveals a non-obvious connection between 2+ existing concepts: CREATE a `connections/` article
6. UPDATE `knowledge/index.md` with new/modified entries
7. APPEND to `knowledge/log.md`

**Important guidelines:**

- A single Capture Record may touch 3-10 knowledge articles
- Prefer updating existing articles over creating near-duplicates
- Use Obsidian-style `[[wikilinks]]` with full relative paths from knowledge/
- Write in encyclopedia style - factual, concise, self-contained (the `/skill:unslop` prefix enforces this)
- Every article must have YAML frontmatter
- Every article must link back to its source Capture Records

#### Historical Bootstrap

On first run after the migration, historical Capture Records materialized from legacy daily aggregates are bootstrapped as already compiled whenever their `source_aggregate_sha256` starts with the matching legacy `state['ingested'][<daily filename>]['hash']`. This prevents replaying already compiled history. The legacy `ingested` map stays readable but is never used for ongoing selection.

### 2. Query (Ask the Knowledge Base)

1. Read `knowledge/index.md` (the master catalog)
2. Based on the question, identify 3-10 relevant articles from the index
3. Read those articles in full
4. Synthesize an answer with `[[wikilink]]` citations
5. If `--file-back` is specified: create a `knowledge/qa/` article and update index.md and log.md

At personal scale (50-500 articles), the LLM reading a structured index outperforms cosine similarity - no RAG needed. See "Scaling Beyond Index-Guided Retrieval" for when that changes.

---

## Conventions

- **Wikilinks:** Use Obsidian-style `[[path/to/article]]` without `.md` extension
- **Writing style:** Encyclopedia-style, factual, third-person where appropriate
- **Dates:** ISO 8601 (YYYY-MM-DD for dates, full ISO for timestamps in log.md)
- **File naming:** lowercase, hyphens for spaces (e.g., `supabase-row-level-security.md`)
- **Frontmatter:** Every article must have YAML frontmatter with at minimum: title, summary, sources, created, updated. The `summary:` field is a 1-2 sentence overview, identical in intent to the lead paragraph after the H1.
- **Sources:** Always link back to the Capture Record(s) that contributed to an article; legacy daily-log references remain valid in historical articles

---

## Full Project Structure

```text
obsidian/                            # Vault root
|-- raw/                             # Markdown Raw Inputs live at the vault root, not inside the repo
|-- claude-memory-compiler/
|   |-- .claude/
|   |   |-- settings.json            # Hook configuration (auto-activates in Claude Code)
|   |-- CONTEXT.md                   # Domain vocabulary: Raw Input, Capture Record, Compilation Run, Knowledge Article
|   |-- AGENTS.md                    # This file - schema + full technical reference
|   |-- README.md                    # Concise overview + quick start
|   |-- Memory Compiler.base         # Unified Obsidian Base: nine views over captures, raw inputs, runs, and knowledge
|   |-- Memory Compiler.md           # Dashboard note that embeds the Base
|   |-- docs/
|   |   |-- architecture.html        # Architecture diagram (data flow from hook to dashboard)
|   |-- plans/                       # Approved work plans
|   |-- pyproject.toml               # Dependencies (at root so hooks can find it)
|   |-- tests/                       # Unit tests for capture records, compilation, and flush writes
|   |-- daily/                       # LEGACY read-only aggregate logs (never written or selected)
|   |-- records/
|   |   |-- captures/                # Direct Capture Records plus materialized legacy projections
|   |   |-- runs/                    # Compilation Run projections from knowledge/log.md
|   |-- knowledge/                   # Compiled knowledge (LLM-owned)
|   |   |-- index.md                 #   Master catalog - THE retrieval mechanism
|   |   |-- log.md                   #   Append-only build log; the authoritative run log
|   |   |-- concepts/                #   Atomic knowledge articles
|   |   |-- connections/             #   Cross-cutting insights linking 2+ concepts
|   |   |-- qa/                      #   Filed query answers
|   |-- scripts/                     # CLI tools
|   |   |-- capture_records.py       #   Capture Record locking, atomic writes, parsing, raw-input discovery + tagging
|   |   |-- compile.py               #   One OMP agent per pending Capture Record (dual-model round-robin)
|   |   |-- query.py                 #   Ask questions (index-guided, no RAG)
|   |   |-- lint.py                  #   8 health checks (capture-record-keyed staleness)
|   |   |-- flush.py                 #   Extracts knowledge from context and writes one direct Capture Record
|   |   |-- materialize_records.py   #   Legacy projection into records/ (--check reports drift)
|   |   |-- backfill.py              #   Backfill historical transcripts
|   |   |-- backfill_summary.py      #   Backfill summary frontmatter into existing articles
|   |   |-- cron-backstop.sh         #   Nightly backstop for missed SessionEnd hooks
|   |   |-- config.py                #   Path constants and model defaults
|   |   |-- utils.py                 #   Shared helpers
|   |-- hooks/                       # Claude Code hooks
|   |   |-- session-start.py         #   Injects index + recent Capture Record excerpts every session
|   |   |-- session-end.py           #   Extracts context and spawns flush.py
|   |   |-- pre-compact.py           #   Same chain before compaction
|   |-- reports/                     # Lint reports (gitignored)
```

`raw/` sits at the vault root because Raw Inputs can belong to any project in the vault, not only this repo.

---

## Hook System (Automatic Capture)

Hooks are configured in `.claude/settings.json` and fire automatically when you use Claude Code in this project.

### `.claude/settings.json` Format

```json
{
  "hooks": {
    "SessionStart": [{ "matcher": "", "hooks": [{ "type": "command", "command": "uv run python hooks/session-start.py", "timeout": 15 }] }],
    "PreCompact": [{ "matcher": "", "hooks": [{ "type": "command", "command": "uv run python hooks/pre-compact.py", "timeout": 10 }] }],
    "SessionEnd": [{ "matcher": "", "hooks": [{ "type": "command", "command": "uv run python hooks/session-end.py", "timeout": 10 }] }]
  }
}
```

Commands use simple relative paths from the project root. Empty `matcher` catches all events.

### Hook Details

**`session-start.py`** (SessionStart)

- Pure local I/O, no API calls, runs in under 1 second
- Reads `knowledge/index.md` plus recent **substantive** Capture Records (`capture_status: narrative`) - bounded excerpts, newest first; never reads `daily/`
- Outputs JSON to stdout: `{"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": "..."}}`; max context 20,000 characters
- Tuning env vars: `KB_INDEX_MAX_ROWS` (default 10), `KB_RECENT_CAPTURES` (default 4), `KB_CAPTURE_EXCERPT_CHARS` (default 900), `KB_INDEX_DAYS`

**`session-end.py`** (SessionEnd)

- Reads hook input from stdin (JSON with `session_id`, `transcript_path`, `cwd`)
- Extracts conversation context locally, then spawns flush.py as a detached background process with the context file path, session id, and source cwd. The hook makes no API calls.
- Recursion guard: exits immediately if `CLAUDE_INVOKED_BY` env var is set

**`pre-compact.py`** (PreCompact)

- Same architecture as session-end.py; fires before Claude Code auto-compacts the context window
- Guards against empty `transcript_path` (known Claude Code bug #13668)

**Why both PreCompact and SessionEnd?** Long-running sessions may auto-compact several times before closing. Both hooks feed the same detached flush, and stable record identity `(record_day, source_project, session id)` converges every write on one Capture Record path, so intermediate context survives summarization without duplication.

**`flush.py`** (background extraction)

Spawned by the capture hooks as a detached background process (Windows `CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS`, Mac/Linux `start_new_session=True`); stderr lands in `scripts/flush-stderr.log`. It sets `CLAUDE_INVOKED_BY=memory_flush`, runs one Claude Agent SDK call with `claude-sonnet-5` by default (`CMC_MODEL` overrides it) to decide what is worth saving, and writes exactly one direct Capture Record with `capture_kind: memory_flush` and a `capture_status` of `narrative`, `no_op`, or `failure`. An identical extraction for the same session within the last 60 seconds is skipped; identity collapse at capture_records.py handles everything else.

---

## Script Details

### capture_records.py - Capture Record I/O

Shared module consumed by flush.py, the compiler, and lint:

- `write_capture_record(...)` - write one Capture Record under `records/captures/<day>/` (interprocess lock + atomic replace)
- `read_capture(path)` / `list_capture_records()` - parse records into frozen `CaptureRecord` objects exposing `path`, `record_id`, `record_day`, `capture_status`, `payload_sha256`, `source_sid`, and `body`
- `discover_unprocessed_raw(record_day, state)` / `tag_raw_inputs(paths, *, compiled, capture_id)` - Raw Input selection and tag flips around compilation
- `capture_state_key(record)` - the key used in `state['compiled_captures']`

Historical Capture Records parse unchanged; parsing is forward-compatible.

### compile.py - The Compiler

One OMP agent per pending substantive Capture Record:

```text
omp -p --no-session --auto-approve --cwd <vault-root> --model <model>
```

with `CLAUDE_INVOKED_BY=knowledge_compiler` exported. Pending means the record is narrative and its id is absent from `state['compiled_captures']`, or its payload hash changed since the recorded compile. Pending records are stable-sorted; models alternate round-robin between `anthropic/claude-sonnet-5` and `openai-codex/gpt-5.6-terra` so model counts differ by at most one. Each prompt begins with the exact text `/skill:unslop`, names exactly one Capture Record plus its same-day Raw Input paths, and directs the agent through the index-first workflow under Core Operations. Article bodies never enter argv or stdin.

One global `flock` makes compilation sequential: two compiles can never load a state snapshot and replay work against stale state.

On success compile.py stores `compiled_captures[record_id] = {payload_sha256, compiled_at, model}` and any Raw Input entries atomically, normalizes `knowledge/index.md`, and calls materialize_records.refresh_projection() so the records projection stays current. A failed agent leaves its record out of `compiled_captures` and its Raw Inputs untagged as compiled, so both retry on the next run; processing continues with the remaining records, and the command exits 1 if any assignment failed.

#### Raw Input Handling

Before invoking the agent for a record, compile.py attaches that day's unprocessed Raw Inputs to the **first** sorted pending Capture Record for that day:

1. A Raw Input is same-day when its YAML `created` date equals the record's `record_day`, and unprocessed when its current SHA-256 is absent from `state['compiled_raw_inputs']`
2. Before invocation the file gets the `memory-compiler/raw-input` tag
3. Only after successful compilation it additionally gets `memory-compiler/compiled` plus a `compiled_in` marker, and `state['compiled_raw_inputs'][relative_path] = {sha256, compiled_at, capture_id}` is recorded

Failed compilations leave Raw Inputs tagged `memory-compiler/raw-input` only, so they retry with the next pending record for that day.

**CLI:**

```bash
uv run python scripts/compile.py                     # compile pending Capture Records
uv run python scripts/compile.py --all               # recompile every narrative record
uv run python scripts/compile.py --file <record>.md  # compile one specific Capture Record
uv run python scripts/compile.py --dry-run           # print assignments under the lock; write nothing
```

`--all` re-selects every narrative record, including already compiled ones; no_op and failure records are never selected. `--file` compiles one record without Raw Input planning. `--dry-run` shares the locked snapshot but mutates nothing.

### query.py - Index-Guided Retrieval

Loads the index plus all articles into context. No RAG - see Core Operations.

**CLI:**

```bash
uv run python scripts/query.py "What auth patterns do I use?"
uv run python scripts/query.py "What's my error handling strategy?" --file-back
```

With `--file-back`, query.py writes a Q&A article in `knowledge/qa/` plus matching index and log entries, then refreshes the records projection. Every filed answer gives future queries more to draw on.

### lint.py - Health Checks

Eight checks:

| Check | Type | Catches |
|-------|------|---------|
| Broken links | Structural | `[[wikilinks]]` to non-existent articles |
| Orphan pages | Structural | Articles with zero inbound links |
| Orphan sources | Structural | Capture Records not yet in `compiled_captures` |
| Stale articles | Structural | Capture Record payload hashes changed since compilation |
| Missing backlinks | Structural | A links to B but B doesn't link back |
| Sparse articles | Structural | Under 200 words |
| Contradictions | LLM | Conflicting claims across articles |
| Index table | Structural | Rows with wrong column count, unescaped alias pipes, or Dataview-triggering code spans |

Orphan-source and staleness checks key on Capture Record IDs and `compiled_captures` payload hashes; they never compare whole-file hashes of daily aggregates. Reports saved to `reports/lint-YYYY-MM-DD.md`.

**CLI:**

```bash
uv run python scripts/lint.py                    # all checks
uv run python scripts/lint.py --structural-only  # skip LLM check (free)
```

### backfill.py - Historical Transcript Importer

Walks `~/.claude/projects/*/*.jsonl`, groups messages by `(local-date, session_id)`, and imports historical conversations as Capture Records for the correct day. Re-runs are safe: stable record identity deduplicates previously imported sessions. Cost estimate is printed before running; `--max-cost` (default $20) aborts if the estimate exceeds the threshold.

```bash
uv run python scripts/backfill.py --date 2026-04-20 --yes  # skip confirmation
```

### materialize_records.py - Legacy Projection

Projects two legacy corpora into `records/`: capture entries in `daily/*.md` become generated notes under `records/captures/`, and compile or filed-query run entries in `knowledge/log.md` become run notes under `records/runs/`. Each entry becomes one note, written with atomic replace.

Notes written directly by flush.py carry `record_origin: direct`. The projector detects that field and excludes those paths from every missing, changed, and stale computation; it never overwrites or deletes a direct record. Only notes it generated itself are refresh candidates, so a refresh is idempotent and hand edits to direct records always survive.

compile.py calls `refresh_projection()` after every successful compile, and query.py does the same after a filed query. Run the script manually to repair projection drift left by hand edits.

```bash
uv run python scripts/materialize_records.py          # rebuild the projection
uv run python scripts/materialize_records.py --check  # report drift without writing; exit 1 when stale
```

`daily/*.md` stays read-only legacy history. `knowledge/log.md` remains the authoritative run log.

### cron-backstop.sh - Nightly Hook Safety Net

Runs nightly to catch sessions whose SessionEnd hook didn't fire (crashed shell, `kill -9`, hook regression) by re-running the import path; stable record identity makes re-runs free. Register as a launchd job for hands-off operation; the script derives `ROOT` from its own location so it survives vault relocations.

---

## State Tracking

`scripts/state.json` tracks:

- `compiled_captures` - map of Capture Record ID to `{payload_sha256, compiled_at, model}`; the sole authority for pending/stale selection
- `compiled_raw_inputs` - map of vault-relative Raw Input path to `{sha256, compiled_at, capture_id}`
- `ingested` - **legacy** map of daily log filenames to whole-file hashes, kept readable for bootstrap and history; never used for ongoing selection
- `query_count`, `last_lint`, `total_cost` - usage counters

`scripts/last-flush.json` tracks flush deduplication (session id, timestamp, context hash). Both state files are gitignored and regenerated automatically.

---

## Dependencies

`pyproject.toml` (at project root):

- `claude-agent-sdk>=0.1.29` - LLM calls for memory flush extraction
- `python-dotenv>=1.0.0` - Environment variable management
- `tzdata>=2024.1` - Timezone data
- Python 3.12+, managed by [uv](https://docs.astral.sh/uv/)
- `omp` CLI on PATH - drives per-record compilation agents

Compilation uses your existing agent credentials; no separate API key is required.

---

## Costs

| Operation | Cost |
|-----------|------|
| Compile one Capture Record | one OMP agent run (varies by record size and assigned model) |
| Query (no file-back) | ~$0.15-0.25 |
| Query (with file-back) | ~$0.25-0.40 |
| Full lint (with contradictions) | ~$0.15-0.25 |
| Structural lint only | $0.00 |
| Memory flush (per session) | ~$0.02-0.05 |
| Materialize records | $0.00 (local I/O only) |

---

## Customization

### Additional Article Types

Add directories like `people/`, `projects/`, `tools/` to `knowledge/`. Define the article format in this file (AGENTS.md) and update `utils.py`'s `list_wiki_articles()` to include them.

### Obsidian Integration

The knowledge base is pure markdown with `[[wikilinks]]`, so Obsidian renders it natively with graph view, backlinks, and search.

`Memory Compiler.base` at the repo root defines one unified dashboard with nine views:

1. Knowledge Catalog
2. Connections
3. Raw Inputs
4. Web Clippings
5. Other Raw
6. Capture Narratives
7. Capture Failures
8. Capture No-ops
9. Compilation Runs

`Memory Compiler.md` embeds the Base. Direct captures and knowledge articles are the primary files in it. Historical daily capture entries and `knowledge/log.md` run entries appear through the materialize_records.py projection, which compile runs and filed queries refresh automatically; run that script manually to repair hand edits. The Base selects Raw Inputs with bare `file.inFolder("raw")` because `raw/` sits at the vault root, one level above this repo; repo-local content needs the `claude-memory-compiler/` prefix in its filters instead. See `docs/architecture.html` for the full data flow.

### Scaling Beyond Index-Guided Retrieval

At ~2,000+ articles / ~2M+ tokens, the index becomes too large for the context window. At that point, add hybrid RAG (keyword + semantic search) as a retrieval layer before the LLM. See Karpathy's recommendation of `qmd` by Tobi Lutke for search at scale.
