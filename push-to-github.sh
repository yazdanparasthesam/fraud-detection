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

# Wait for Enter. Reads from /dev/tty when stdin is not a terminal (piped
# input), and never aborts the script on EOF.
press_enter() {
  local msg="$1"
  if [ -t 0 ]; then
    read -r -p "$msg" _ || true
  elif [ -r /dev/tty ] && printf '%s' "$msg" >/dev/tty 2>/dev/null; then
    read -r _ < /dev/tty 2>/dev/null || true
  else
    echo "$msg (no terminal to wait on - continuing)"
  fi
}

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
git config user.name  >/dev/null 2>&1 || { printf 'Your git name : '; read -r GN || true; git config user.name  "$GN"; }
git config user.email >/dev/null 2>&1 || { printf 'Your git email: '; read -r GE || true; git config user.email "$GE"; }

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
    printf 'GitHub username: '; read -r USERNAME || true
    git remote add origin "https://github.com/$USERNAME/$REPO_NAME.git"
  fi
fi

# ---------------------------------------------------------- auth pre-flight
# GitHub has refused account passwords over HTTPS since 2021: "Invalid username
# or token. Password authentication is not supported." Catch that before the
# user hits it, and offer to switch the remote to SSH, which needs a key rather
# than a token and then works permanently.
URL="$(git remote get-url origin)"

if [ "${URL#https://}" != "$URL" ] || [ "${URL#http://}" != "$URL" ]; then
  # `--dry-run` exercises the real push auth path without sending anything.
  # (ls-remote would only prove read access, which works anonymously on a
  #  public repo even when pushing is impossible.)
  PUSHTEST="$(git push --dry-run origin main 2>&1 || true)"
  case "$PUSHTEST" in
    *"could not read Username"*|*"Authentication failed"*|*"Invalid username or token"*|*"Permission denied"*|*403*)
      echo
    echo "HTTPS remote, but authentication is failing."
    echo "(GitHub does not accept your account password over HTTPS - it needs a"
    echo " Personal Access Token or an SSH key.)"
    echo
    echo "  1) Switch to SSH  (recommended: one-time key, then it just works)"
    echo "  2) Keep HTTPS     (you will be asked for a Personal Access Token)"
    echo "  3) Abort"
    echo
    printf 'Choose [1/2/3]: '; read -r CHOICE || true
    case "$CHOICE" in
      1)
        REPO_PATH="${URL#https://github.com/}"; REPO_PATH="${REPO_PATH%.git}"
        [ -f "$HOME/.ssh/id_ed25519.pub" ] || ssh-keygen -t ed25519 -N '' -f "$HOME/.ssh/id_ed25519"
        eval "$(ssh-agent -s)" >/dev/null && ssh-add "$HOME/.ssh/id_ed25519" 2>/dev/null
        echo
        echo "Add this public key at https://github.com/settings/keys (New SSH key):"
        echo
        cat "$HOME/.ssh/id_ed25519.pub"
        echo
        press_enter "Press Enter once the key is added to GitHub: "
        # Trust github.com's host key up front, so the first push is not
        # blocked by an interactive "authenticity of host" prompt.
        mkdir -p "$HOME/.ssh" && chmod 700 "$HOME/.ssh"
        touch "$HOME/.ssh/known_hosts" && chmod 600 "$HOME/.ssh/known_hosts"
        if command -v ssh-keyscan >/dev/null 2>&1; then
          ssh-keyscan -t ed25519,rsa github.com >>"$HOME/.ssh/known_hosts" 2>/dev/null || true
        fi

        git remote set-url origin "git@github.com:$REPO_PATH.git"
        echo "==> remote is now git@github.com:$REPO_PATH.git"
        ssh -T git@github.com 2>&1 | head -2 || true
        ;;
      2)
        echo
        echo "Create a token at https://github.com/settings/tokens"
        echo "  Fine-grained token -> repository access: $REPO_NAME"
        echo "  Permissions -> Contents: Read and write"
        echo
        echo "When git asks for a password, PASTE THE TOKEN (not your password)."
        echo
        press_enter "Press Enter to continue: "
        ;;
      *)
        echo "Aborted."; exit 1 ;;
    esac
  ;;
  esac
fi

# --------------------------------------------------------------------- push
echo "==> pushing to $(git remote get-url origin)"
git push -u origin main

echo
echo "DONE -> https://github.com/$(git remote get-url origin | sed -E 's#.*github.com[:/](.*/.*)(\.git)?#\1#')"
echo "Next: share.streamlit.io -> New app -> main file: streamlit_app.py"
echo "      Advanced settings -> Python 3.12, requirements-cloud.txt"
