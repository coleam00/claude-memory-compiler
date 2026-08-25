# Memory Compiler

The Memory Compiler preserves working context and saved source material, then turns selected material into durable knowledge.

## Language

**Raw Input**:
Uncurated saved material that is eligible to inform compilation.
_Avoid_: Bookmark, clipping

**Capture Record**:
One preserved memory-capture event, whether it contains a substantive session narrative, a no-op, or a failure. Direct records are written by flush; legacy daily aggregate entries are projected copies created by `materialize_records.py`.
_Avoid_: Session Record, daily aggregate entry

**Compilation Run**:
One attempt to compile selected source material into durable knowledge, including retries and no-op runs.
_Avoid_: Build-log entry

**Knowledge Article**:
A durable concept, connection, or question-and-answer note maintained by compilation.
_Avoid_: Index row, page
