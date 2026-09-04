# Incident: `git status --short` disclosed deleted legacy filenames

## Summary

While starting execution of `PLANS.md`, Codex ran `git status --short` to inspect the working tree. Although this command does not read commit history or deleted file contents, it listed paths of tracked files that had been deleted from the current working tree. This exposed the names and prior structure of the failed implementation that `PLANS.md` explicitly instructed Codex not to inspect.

## Impact

- The command revealed deleted legacy filenames and directory structure.
- No deleted file contents, commits, diffs, or Git history were read.
- The exposed filenames entered the model context and could bias subsequent implementation decisions.
- Work on executing `PLANS.md` was stopped immediately after the user identified the issue.

## Timeline

1. Codex read `PLANS.md`.
2. Codex ran `git status --short` together with a current-file listing to determine the repository state.
3. `git status --short` printed deleted tracked paths from the prior implementation.
4. The user pointed out that this leaked which files had existed.
5. Codex acknowledged the violation and stopped execution work.

## Root cause

Codex treated `git status --short` as a harmless working-tree inspection because it does not show commit history. That interpretation was too narrow. In a repository where deleted tracked paths belong to a failed implementation that must not influence the new work, the Git index itself is historical information: status output reveals names and structure that are absent from the current filesystem.

## Contributing factors

- The repository workflow normally calls for checking untracked paths before broad review or diff operations.
- Codex applied that general workflow without reconciling it with the more specific instruction not to inspect the failed implementation.
- Codex focused on avoiding `git log` and `git show`, but failed to classify deleted-path output from `git status` as information about the old implementation.

## Corrective actions

- Do not run repository-wide `git status`, `git diff`, `git ls-files`, `git log`, `git show`, or similar Git inspection commands while executing this clean-room rebuild unless the user explicitly authorizes them.
- Discover the current implementation surface only from files that physically exist in the current working tree, using filesystem-scoped commands such as `rg --files` with generated-data exclusions.
- When committing an intentional edit, stage only the exact edited path and commit it without first querying repository-wide Git state.
- Do not inspect, restore, infer, or reproduce any deleted legacy path revealed by this incident unless the user explicitly directs it.

## Prevention rule

For this branch, “do not view Git history or failed prior implementations” includes Git metadata that reveals deleted tracked paths. A command is disallowed if its output can expose absent legacy files, even when it does not display commits or file contents.
