# Agent Guide


## Working Modes

- Explore mode: inspect files, commands, tests, and data without changing anything.
- Planning mode: prepare a scoped plan before changes, especially for medium- and high-risk work.
- Patching mode: edit only the files needed for the approved change and keep unrelated refactors out.
- Review mode: check the change for architecture, data-contract, prompt, and report-format risks.
- Test mode: run the smallest relevant checks first; broaden tests when shared behavior or pipeline contracts changed.

For high-risk tasks, use this sequence: Explore -> Planning -> Review -> Patching -> Test.

Always reread this `AGENTS.md` at the start of a new coding task or after context compaction. If it conflicts with older assumptions, this file wins unless the user explicitly says otherwise.

## Engineering Style

- Prefer the simplest implementation that preserves the contract and is easy to inspect in runtime artifacts.
- Follow a Ponytail-style bias: keep the code tied back, neat, and out of the way. Do not add abstractions, frameworks, generic engines, or multi-layer configuration unless they remove real complexity now.
- Do not over-engineer speculative future needs. Leave clear extension points, but implement only the behavior required by the current document types and fixtures.
- Make parsers boring and explicit: small functions, readable names, deterministic transformations, and visible warnings when confidence is low.
- Prefer structured intermediate artifacts over clever prompt logic. If a table can be parsed deterministically, parse it before involving an LLM.
- Do not hide messy input behind broad catch-all logic. Preserve debug evidence so bad parsing can be inspected quickly.