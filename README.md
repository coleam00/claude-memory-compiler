# LLM Personal Knowledge Base

Your AI conversations compile themselves into a searchable knowledge base.

Adapted from [Karpathy's LLM Knowledge Base](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f). Instead of clipping web articles, the raw data is your own conversations with Claude Code. When a session ends, or when auto-compaction fires mid-session, Claude Code hooks call `scripts/flush.py`. It uses the Claude Agent SDK to extract knowledge from the conversation and writes one direct Capture Record per logical session into `records/captures/<YYYY-MM-DD>/`, a markdown file with YAML frontmatter identifying the session. You then compile those Capture Records into structured, cross-referenced knowledge articles organized by concept. Retrieval uses a simple index file instead of RAG: no vector database, no embeddings, just markdown.

## Quick start

Tell your AI coding agent:

> "Clone <https://github.com/coleam00/claude-memory-compiler> into this project. Set up the Claude Code hooks so my conversations automatically get captured as Capture Records, compiled into a knowledge base, and injected back into future sessions. Read the AGENTS.md for the full technical reference on how everything works."

The agent will:

1. Clone the repo and run `uv sync` to install dependencies
2. Copy `.claude/settings.json` into your project (or merge the hooks into your existing settings)
3. The hooks activate automatically next time you open Claude Code

From there, your conversations start accumulating as Capture Records. Compilation turns each pending Capture Record into or onto knowledge articles; run `uv run python scripts/compile.py` manually at any time.

## How it works

```text
Conversation -> SessionEnd/PreCompact hooks pass context to scripts/flush.py
    -> flush.py extracts knowledge via Claude Agent SDK
    -> writes one direct memory_flush Capture Record
    -> records/captures/YYYY-MM-DD/*.md
    -> compile.py spawns one OMP agent per pending record (dual models, round-robin)
    -> knowledge/concepts/, connections/, qa/
        -> SessionStart hook injects index + recent Capture Record excerpts -> cycle repeats
```

- The SessionEnd and PreCompact hooks never write records themselves. They pass the context path, session ID, and source cwd to flush.py, which does the SDK extraction and writes the record. Because both events converge on the same identity, an overlap overwrites one path instead of producing two records.
- Capture Records are immutable per-session inputs with stable identity across PreCompact and SessionEnd (`record_day`, `source_project`, `session_id`; payload hash when no session ID exists).
- Conversations produce Capture Records under `records/captures/`. Raw Inputs are a separate thing: unprocessed markdown files under the vault-root `raw/` directory. Same-day files whose YAML `created` date matches a record's day join that day's first pending compilation and receive `memory-compiler/raw-input` / `memory-compiler/compiled` tags.
- compile.py processes pending substantive records with one OMP agent per record, balancing two models (`anthropic/claude-sonnet-5`, `openai-codex/gpt-5.6-terra`) so model counts differ by at most one. Every prompt starts with `/skill:unslop`, names exactly one record plus its Raw Input paths, and directs the agent to read the knowledge index and only relevant articles. Article bodies never travel through argv or stdin.
- Locking and state stay simple. flush.py writes records atomically under a capture lock, so concurrent hook events cannot tear a file. compile.py takes one global lock, making compilation sequential. state.json stores each record's payload hash; a changed hash means retry, an unchanged hash already in state means the record is done.
- query.py answers questions using index-guided retrieval (no RAG needed at personal scale).
- lint.py runs 8 health checks (broken links, orphans, contradictions, staleness keyed on Capture Record hashes).
- Historical `daily/*.md` aggregates remain readable as read-only legacy inputs; nothing appends to them or selects them for compilation anymore.

## Key commands

```bash
uv run python scripts/compile.py                      # compile pending Capture Records
uv run python scripts/query.py "question"             # ask the knowledge base
uv run python scripts/query.py "question" --file-back # ask + save answer back
uv run python scripts/lint.py                         # run health checks
uv run python scripts/lint.py --structural-only       # free structural checks only
uv run python scripts/materialize_records.py          # refresh legacy projection
uv run python scripts/materialize_records.py --check  # report drift without writing
```

materialize_records.py projects two legacy sources into `records/`: historical capture entries from `daily/*.md` and compilation run entries from `knowledge/log.md`. It detects direct records and never overwrites or deletes them. Run it manually if you hand-edit those sources; compile.py and file-backed queries refresh it automatically.

## Why no RAG?

Karpathy's insight holds at personal scale. With 50 to 500 articles, an LLM reading a structured `index.md` beats vector similarity. The LLM understands what you are actually asking; cosine similarity just finds similar words. RAG starts earning its keep around 2,000+ articles, when the index no longer fits in the context window.

## Browse in Obsidian

Open `Memory Compiler.md` in the vault. It embeds `Memory Compiler.base`, a dashboard with nine filtered views over the records and knowledge directories, such as Knowledge Catalog, Raw Inputs, and Capture Failures. See [docs/architecture.html](docs/architecture.html) for how the pieces fit together.

## Technical reference

See [AGENTS.md](AGENTS.md) for the complete technical reference: article formats, hook architecture, script internals, state keys, costs, and customization options. AGENTS.md gives an AI agent everything it needs to understand, modify, or rebuild the system.
