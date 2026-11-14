#!/usr/bin/env bash
# Polls NIFC for changed perimeters every POLL_SECONDS for LOOP_SECONDS and pushes them to the
# perimeter-archive branch. Run by .github/workflows/perimeter-capture.yml; also runs on any machine
# that may push to the repository:
#   ARCHIVE_URL=https://github.com/GT-Fire-Research-and-Innovation/WildfireRepo1 bash live/capture_loop.sh
# A push happens when a poll saved something, and at least once an hour for the poll log.
set -euo pipefail
: "${ARCHIVE_URL:?repository to push the archive to}"
POLL_SECONDS=${POLL_SECONDS:-300}
LOOP_SECONDS=${LOOP_SECONDS:-19800}          # 5.5 h: GitHub stops a job at 6 h
PUSH_SECONDS=${PUSH_SECONDS:-3300}
BRANCH=${BRANCH:-perimeter-archive}
WORK=${WORK:-${RUNNER_TEMP:-/tmp}/arch}
POLLER="$(cd "$(dirname "$0")" && pwd)/wfigs_poller.py"

# Only the index, the list of what has been seen and the poll log are checked out: the archive grows all
# year and a poll just adds files to it.
rc=0; git ls-remote --exit-code --heads "$ARCHIVE_URL" "$BRANCH" >/dev/null 2>&1 || rc=$?
rm -rf "$WORK"
if [ "$rc" -eq 0 ]; then
  git clone -q --depth 1 --filter=blob:none --sparse --branch "$BRANCH" "$ARCHIVE_URL" "$WORK"
  git -C "$WORK" sparse-checkout set --no-cone /perimeters/index.csv /perimeters/seen.csv /perimeters/polls/
elif [ "$rc" -eq 2 ]; then                   # no such branch yet: start the archive
  mkdir -p "$WORK" && git -C "$WORK" init -q -b "$BRANCH" && git -C "$WORK" remote add origin "$ARCHIVE_URL"
else                                         # never start a second, empty archive because the network blinked
  echo "cannot reach the repository (git ls-remote exit $rc)"; exit 1
fi
cd "$WORK"

save() {
  git add --sparse -A
  git diff --cached --quiet || git -c user.name="gt-fire-bot" -c user.email="actions@users.noreply.github.com" \
      commit -qm "NIFC perimeters $(date -u +%Y-%m-%dT%H:%MZ)"
  for _ in 1 2 3; do
    git push -q origin "$BRANCH" && return 0
    git pull -q --rebase origin "$BRANCH" || true
  done
  return 1
}

end=$(( $(date +%s) + LOOP_SECONDS )); pushed=$(date +%s); unsaved=0
while :; do
  t0=$(date +%s)
  python "$POLLER" --out . --once || true    # a failed poll is in the log; the next one catches up
  git add --sparse -A
  if ! git diff --cached --quiet -- perimeters/index.csv || [ "$unsaved" -eq 1 ] || [ $(( t0 - pushed )) -ge "$PUSH_SECONDS" ]; then
    if save; then pushed=$t0; unsaved=0; else unsaved=1; echo "push failed; trying again after the next poll"; fi
  fi
  wait_s=$(( POLL_SECONDS - ($(date +%s) - t0) ))
  [ $(( $(date +%s) + wait_s )) -ge "$end" ] && break
  [ "$wait_s" -gt 0 ] && sleep "$wait_s"
done
save
