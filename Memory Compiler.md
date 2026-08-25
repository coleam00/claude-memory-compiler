# Memory Compiler

This dashboard projects raw inputs, capture records, compilation runs, and knowledge articles.

Direct Capture Records and Knowledge Articles are the primary files. `scripts/materialize_records.py` projects historical daily capture entries and knowledge/log run entries into records/captures and records/runs. Successful compiles and filed queries refresh that projection; run `uv run python scripts/materialize_records.py` to refresh manually and append `--check` to detect drift after hand edits.

![[Memory Compiler.base]]
