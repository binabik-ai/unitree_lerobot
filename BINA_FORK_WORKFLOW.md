# Fork Workflow

How Binabik manages our fork of [unitree_lerobot](https://github.com/unitreerobotics/unitree_lerobot).

## Branch structure

- **`main`** — Clean mirror of `upstream/main`. Never commit directly.
- **`binabik/main`** — Our long-lived working branch. Default branch on the fork.
- **`upstream/<name>`** — Short-lived branches for fixes/features intended for upstream PRs. Branch from `main`.
- **`binabik/<name>`** — Short-lived branches for internal-only work. Branch from `binabik/main`.

## Remotes

```
origin    → github.com/binabik/unitree_lerobot     (our fork)
upstream  → github.com/unitreerobotics/unitree_lerobot
```

## One-time setup

```bash
git clone git@github.com:binabik/unitree_lerobot.git
cd unitree_lerobot

git remote add upstream https://github.com/unitreerobotics/unitree_lerobot.git
git remote set-url --push upstream DISABLED   # prevent accidental pushes upstream

git fetch upstream
git checkout main
git merge --ff-only upstream/main

git checkout -b binabik/main
git push -u origin binabik/main
```

On GitHub: Settings → Branches → set `binabik/main` as default.

## Workflow: internal feature

```bash
git checkout binabik/main
git pull
git checkout -b binabik/add-custom-gripper-support
# ...work, commit, push...
git push -u origin binabik/add-custom-gripper-support
```

Open PR: `binabik/add-custom-gripper-support` → `binabik/main`.

## Workflow: upstream-bound fix

**Branch from `main`, not `binabik/main`** — this keeps the change free of internal code.

```bash
git fetch upstream
git checkout main
git merge --ff-only upstream/main

git checkout -b upstream/fix-dataset-loader-race
# ...fix, commit, push...
git push -u origin upstream/fix-dataset-loader-race
```

Open two PRs from the same branch:

1. `upstream/fix-dataset-loader-race` → `unitreerobotics:main` (upstream PR)
2. `upstream/fix-dataset-loader-race` → `binabik:binabik/main` (so we use the fix now without waiting)

## Weekly upstream sync

```bash
# Update main from upstream
git fetch upstream
git checkout main
git merge --ff-only upstream/main
git push origin main

# Bring upstream changes into binabik/main via a PR
git checkout -b chore/sync-upstream-$(date +%Y-%m-%d)
git merge main
# ...resolve conflicts, commit...
git push -u origin chore/sync-upstream-$(date +%Y-%m-%d)
```

Open PR → `binabik/main`. Never push merge commits straight to `binabik/main`.

## Releases

```bash
git checkout binabik/main
git pull
git tag -a binabik-v1.0.0 -m "Internal release 1.0.0"
git push origin binabik-v1.0.0
```

## Optional: git aliases

Add to `~/.gitconfig`:

```ini
[alias]
    sync-upstream = !git fetch upstream && git checkout main && git merge --ff-only upstream/main && git push origin main
    new-upstream-fix = !f() { git fetch upstream && git checkout main && git merge --ff-only upstream/main && git checkout -b upstream/$1; }; f
    new-binabik-feat = !f() { git checkout binabik/main && git pull && git checkout -b binabik/$1; }; f
```

## Rules of thumb

- Upstream-bound branches contain **no Binabik-specific code**. Ever.
- Long-lived branches (`main`, `binabik/main`) get changes via PR, not direct push.
- Use `merge` (not `rebase`) for long-lived branches; rebase is fine on short-lived feature branches before merging.
- Prefer adding new files / extension points over modifying upstream files — reduces merge conflicts.
- `unitree_lerobot` is Apache 2.0 (built on Hugging Face `lerobot`). Check NOTICE and attribution requirements before distributing anything derived from it.