# Repository working rules

- Read the relevant research specification and implementation before changing a
  mechanism. Distinguish new implementations from reproduced experimental results.
- Preserve unrelated edits, existing checkpoints and experiment artifacts. Keep
  fixes small and verify action, timestep, mask, reward and gradient contracts.
- Run `python -m unittest discover -s tests -v` for changes to the `merl` core.
  GPU/distributed changes also need appropriate runtime validation; report blockers.
- For each commit, use an English subject `Type(module): One-sentence summary`
  (for example `Feat(trust): Add frozen no-oracle residual estimation`), followed
  by a blank line and detailed English bullet points.
- The user has requested that every local commit be followed by a push to its
  corresponding remote branch. Inspect the branch/upstream and working tree first,
  stage only intended files, then commit and push. Do not force-push or rewrite
  shared history. If push fails, report the local commit and the failure explicitly.
- Keep private paper/rebuttal materials, credentials, datasets, model weights and
  generated outputs out of commits. Do not infer a root license from vendored code.
