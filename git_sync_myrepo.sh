#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

REMOTE_NAME="${REMOTE_NAME:-myrepo}"
TARGET_BRANCH="${TARGET_BRANCH:-main}"
SNAPSHOT_BRANCH="${SNAPSHOT_BRANCH:-myrepo_sync}"
COMMIT_MESSAGE="${COMMIT_MESSAGE:-Sync lerobot snapshot $(date '+%Y-%m-%d %H:%M:%S')}"
TMP_DIR=""

cleanup() {
  if [[ -n "${TMP_DIR}" ]]; then
    git worktree remove --force "${TMP_DIR}" >/dev/null 2>&1 || true
    rm -rf "${TMP_DIR}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Error: ${SCRIPT_DIR} is not a git repository."
  exit 1
fi

if ! git remote get-url "${REMOTE_NAME}" >/dev/null 2>&1; then
  echo "Error: git remote '${REMOTE_NAME}' does not exist."
  exit 1
fi

if [[ -n "$(git status --porcelain --untracked-files=no | grep '^UU' || true)" ]]; then
  echo "Error: unresolved merge conflicts detected."
  exit 1
fi

echo "[1/7] Repository: ${SCRIPT_DIR}"
echo "[2/7] Remote: ${REMOTE_NAME} -> $(git remote get-url "${REMOTE_NAME}")"

echo "[3/7] Staging tracked and untracked files (respects .gitignore)..."
git add -A

if git diff --cached --quiet; then
  echo "[4/7] No new local source commit needed."
else
  echo "[4/7] Creating local source commit..."
  git commit -m "${COMMIT_MESSAGE}"
fi

SOURCE_COMMIT="$(git rev-parse HEAD)"

echo "[5/7] Fetching ${REMOTE_NAME}/${TARGET_BRANCH}..."
git fetch "${REMOTE_NAME}" --prune

TMP_DIR="$(mktemp -d /tmp/lerobot_sync.XXXXXX)"
git worktree add --detach "${TMP_DIR}" "${SOURCE_COMMIT}" >/dev/null

cd "${TMP_DIR}"
if git show-ref --verify --quiet "refs/remotes/${REMOTE_NAME}/${TARGET_BRANCH}"; then
  echo "Remote branch exists. Building snapshot on top of ${REMOTE_NAME}/${TARGET_BRANCH}..."
  git switch -C "${SNAPSHOT_BRANCH}" "${REMOTE_NAME}/${TARGET_BRANCH}" >/dev/null
else
  echo "Remote branch does not exist yet. Creating orphan snapshot branch..."
  git checkout --orphan "${SNAPSHOT_BRANCH}" >/dev/null
fi

git rm -rf . --ignore-unmatch >/dev/null 2>&1 || true
git clean -fdx >/dev/null 2>&1 || true
git checkout "${SOURCE_COMMIT}" -- .
git add -A

if git diff --cached --quiet; then
  echo "[6/7] Snapshot branch already matches source tree."
else
  echo "[6/7] Creating snapshot commit for ${REMOTE_NAME}/${TARGET_BRANCH}..."
  git commit -m "${COMMIT_MESSAGE}"
fi

echo "[7/7] Pushing snapshot branch to ${REMOTE_NAME}/${TARGET_BRANCH}..."
git push "${REMOTE_NAME}" "${SNAPSHOT_BRANCH}:${TARGET_BRANCH}"

echo
echo "Done."
echo "Local source branch: $(git -C "${SCRIPT_DIR}" branch --show-current)"
echo "Remote updated: ${REMOTE_NAME}/${TARGET_BRANCH}"
