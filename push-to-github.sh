#!/usr/bin/env bash
#
# Push the fraud-detection project to GitHub.
#
#   chmod +x push-to-github.sh
#   ./push-to-github.sh            # run from inside your project folder
#   ./push-to-github.sh ~/path/to/project
#
# Creates the repo with the `gh` CLI if available, otherwise expects you to have
# already created an EMPTY (no README, no .gitignore) repo named fraud-detection
# on github.com/new - the script then just wires up the remote and pushes.
#
# Prerequisites: git, and either `gh auth login` or an SSH key added to GitHub.

set -euo pipefail

REPO_NAME="fraud-detection"
PROJECT_DIR="${1:-$PWD}"
cd "$PROJECT_DIR"

echo "==> project: $PROJECT_DIR"

# ---------------------------------------------------------------- sanity check
for f in streamlit_app.py inference.py requirements-cloud.txt models/fraud_model.bin; do
  [ -f "$f" ] || { echo "ERROR: $f not found - are you in the project folder?"; exit 1; }
done
echo "==> required files present"

# ------------------------------------------------------------------ git setup
[ -d .git ] || git init -q -b main
git branch -m main 2>/dev/null || true

# git config, only if unset (never overwrite an existing identity)
git config user.name  >/dev/null 2>&1 || { printf 'Your git name : '; read -r GN; git config user.name  "$GN"; }
git config user.email >/dev/null 2>&1 || { printf 'Your git email: '; read -r GE; git config user.email "$GE"; }

# ------------------------------------------------------------- stage the files
git add -A

# models/fraud_model.bin MUST be committed - Streamlit Cloud cannot run train.py.
# `git add -f` overrides any stale ignore rule on your machine.
git add -f models/fraud_model.bin

# --------------------------------------------------- refuse to commit big data
if git diff --cached --name-only | grep -qE '^(data/|outputs/)'; then
  echo "ERROR: data/ or outputs/ about to be committed."
  echo "       Fix .gitignore, then: git rm -r --cached data outputs"
  exit 1
fi

# ------------------------------------------------------------------- commit
if git diff --cached --quiet; then
  echo "==> nothing new to commit"
else
  git commit -q -m "PaySim fraud detection: CRISP-DM pipeline, FastAPI + Streamlit, Docker, cloud-ready"
  echo "==> committed"
fi

# ------------------------------------------------------------------- remote
if git remote get-url origin >/dev/null 2>&1; then
  echo "==> origin already set: $(git remote get-url origin)"
else
  if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
    echo "==> creating repo '$REPO_NAME' on GitHub (public - required by Streamlit Cloud free tier)"
    gh repo create "$REPO_NAME" --public --source=. --remote=origin --push
    echo
    echo "DONE -> $(gh repo view "$REPO_NAME" --json url -q .url)"
    echo "Next: share.streamlit.io -> New app -> main file: streamlit_app.py"
    echo "      Advanced settings -> Python 3.12, requirements-cloud.txt"
    exit 0
  else
    echo
    echo "No 'gh' CLI (or not logged in). Create the repo yourself first:"
    echo "  1. Open https://github.com/new"
    echo "  2. Name: $REPO_NAME   Visibility: Public"
    echo "  3. Leave EVERY checkbox unticked (no README, no .gitignore, no licence)"
    echo "  4. Create repository"
    echo
    printf 'GitHub username: '; read -r USERNAME
    git remote add origin "https://github.com/$USERNAME/$REPO_NAME.git"
  fi
fi

# --------------------------------------------------------------------- push
echo "==> pushing to $(git remote get-url origin)"
git push -u origin main

echo
echo "DONE -> https://github.com/$(git remote get-url origin | sed -E 's#.*github.com[:/](.*/.*)(\.git)?#\1#')"
echo "Next: share.streamlit.io -> New app -> main file: streamlit_app.py"
echo "      Advanced settings -> Python 3.12, requirements-cloud.txt"
