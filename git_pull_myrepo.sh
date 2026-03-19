#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

REMOTE_NAME="${REMOTE_NAME:-myrepo}"
REMOTE_BRANCH="${REMOTE_BRANCH:-main}"
LOCAL_BRANCH="${LOCAL_BRANCH:-myrepo_main}"
FORCE_RESET="${FORCE_RESET:-true}"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Error: ${SCRIPT_DIR} is not a git repository."
  exit 1
fi

if ! git remote get-url "${REMOTE_NAME}" >/dev/null 2>&1; then
  echo "Error: git remote '${REMOTE_NAME}' does not exist."
  exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Error: working tree is not clean. Please commit or stash local changes first."
  git status --short
  exit 1
fi

echo "[1/5] Fetching ${REMOTE_NAME}..."
git fetch "${REMOTE_NAME}" --prune

if ! git show-ref --verify --quiet "refs/remotes/${REMOTE_NAME}/${REMOTE_BRANCH}"; then
  echo "Error: remote branch ${REMOTE_NAME}/${REMOTE_BRANCH} does not exist."
  exit 1
fi

if git show-ref --verify --quiet "refs/heads/${LOCAL_BRANCH}"; then
  echo "[2/5] Switching to existing local branch ${LOCAL_BRANCH}..."
  git checkout "${LOCAL_BRANCH}"
else
  echo "[2/5] Creating local branch ${LOCAL_BRANCH} from ${REMOTE_NAME}/${REMOTE_BRANCH}..."
  git checkout -b "${LOCAL_BRANCH}" "${REMOTE_NAME}/${REMOTE_BRANCH}"
fi

if [[ "${FORCE_RESET}" == "true" ]]; then
  echo "[3/5] Resetting ${LOCAL_BRANCH} to ${REMOTE_NAME}/${REMOTE_BRANCH}..."
  git reset --hard "${REMOTE_NAME}/${REMOTE_BRANCH}"
else
  echo "[3/5] Pulling ${REMOTE_NAME}/${REMOTE_BRANCH} into ${LOCAL_BRANCH}..."
  git pull --ff-only "${REMOTE_NAME}" "${REMOTE_BRANCH}"
fi

echo "[4/5] Current branch: $(git branch --show-current)"
echo "[5/5] Current commit: $(git rev-parse --short HEAD)"

echo
echo "Done. Local branch '${LOCAL_BRANCH}' now matches ${REMOTE_NAME}/${REMOTE_BRANCH}."
