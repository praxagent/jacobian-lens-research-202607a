# shared/researchers — researcher handle registry

A reusable, machine-readable registry of active researchers in LLM-behavior /
interpretability / neuro-AI, so we can track the field and (later) automate.

## Files

- [`handles.yaml`](handles.yaml) — the registry. One entry per researcher:
  handle, name, affiliation, focus, tags, source, and a `verify` flag for
  attributions we haven't confirmed. **Never fabricate an affiliation** — mark
  `verify: true` if unsure.

## Intended future use (not built yet)

A **daily research scan**: poll these handles' recent posts, filter to ones that
link papers / announce results, and produce a daily digest ("new from the people
we follow"). Design notes for whoever builds it:

- The registry is the input; `tags` let a digest be topic-scoped (e.g. only
  `brain-alignment` + `interpretability`).
- X/Twitter API access (or an ethical scraper / Nitter-style feed) is the
  transport; keep any token in a gitignored `.env`, never here.
- Output could feed a Prax channel or a markdown digest — but the scan itself is
  research tooling and lives in THIS repo, not the Prax product harness.

## Adding researchers

When a source (a thread, a paper, a talk) surfaces relevant researchers, add
them here with the `source` field pointing back. Growing this registry over time
is how praxagent stays a well-connected member of the global LLM-behavior
research community.
