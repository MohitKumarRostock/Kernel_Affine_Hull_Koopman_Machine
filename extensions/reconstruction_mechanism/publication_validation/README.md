# Final publication validation

This directory records verification performed when the completed results were
prepared for public release. It does not replace the original execution,
scientific freeze, selected-state replay or independent audit in `results/`.

- `independent_recheck.json`: a fresh run of the independent implementation
  audit against the preserved evidence, without refitting a forecasting model.
- `contract_tests.log`: all 38 numerical, split and analysis tests pass.
- `transport_verify.json`: all 77 lossless parts and 1,631 archived members
  match their recorded hashes.
- `release_check.json`: checks the standalone release archives after extraction.
  The editable source compiles to 99 pages with identical layout text and no
  unresolved references or overflow warnings. The extracted reproduction
  package replays all 1,920 selected prediction arrays exactly, without
  refitting, reselection or additional statistical testing. Archive hashes
  identify the checked packages; local paths record the verification environment.
- `LOCAL_EXECUTION_PROVENANCE.json`: the original local commit objects and
  their identities, including the prospective mechanism freeze. The final
  public snapshot has the preceding public research commit as its parent;
  these local objects are provenance records, not a public preregistration
  or a claim about the public branch's ancestry.

The original source and input hashes, raw results, analysis decision and
recorded costs remain unchanged. Final packaging separately verifies the
compiled manuscript, complete editable source and expanded reproduction
archive. The final manuscript reports completed research and its fixed
stopping decision; no roadmap-only revision is published as a result.
