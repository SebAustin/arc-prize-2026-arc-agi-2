#!/usr/bin/env bash
# Install/uninstall a macOS launchd agent that fires the ARC-AGI-2 daily
# autopilot (scripts/daily_autopilot.py --once) twice a day.
#
# WHY launchd (not a cloud scheduler): the Kaggle credentials live in local
# files (~/.kaggle/credentials.json + tokens) and the repo is on this machine,
# so only a LOCAL job can authenticate and push. Caveat: the Mac must be awake
# for a scheduled tick to fire; missed ticks are harmless — each tick is
# idempotent and simply resumes the state machine on the next fire.
#
# Usage:
#   scripts/install_autopilot.sh install     # write plist + load it
#   scripts/install_autopilot.sh uninstall   # unload + remove plist
#   scripts/install_autopilot.sh status      # show agent + last log lines
#   scripts/install_autopilot.sh run-once     # fire a single tick right now
set -euo pipefail

LABEL="com.arcagi2.autopilot"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"
LOG="$REPO/artifacts/autopilot.log"

# Prefer the repo venv python; fall back to python3 on PATH.
PY="$REPO/.venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3)"

# The autopilot shells out to `kaggle`; capture its dir so launchd's minimal
# PATH can still find it.
KAGGLE_BIN="$(command -v kaggle || true)"
KAGGLE_DIR="$(dirname "${KAGGLE_BIN:-/usr/local/bin}")"
RUN_PATH="$KAGGLE_DIR:/usr/local/bin:/usr/bin:/bin:$(dirname "$PY")"

write_plist() {
  mkdir -p "$(dirname "$PLIST")" "$REPO/artifacts"
  cat > "$PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>${LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>${PY}</string>
    <string>${REPO}/scripts/daily_autopilot.py</string>
    <string>--once</string>
  </array>
  <key>WorkingDirectory</key><string>${REPO}</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key><string>${HOME}</string>
    <key>PATH</key><string>${RUN_PATH}</string>
  </dict>
  <key>StartCalendarInterval</key>
  <array>
    <dict><key>Hour</key><integer>8</integer><key>Minute</key><integer>0</integer></dict>
    <dict><key>Hour</key><integer>20</integer><key>Minute</key><integer>0</integer></dict>
  </array>
  <key>StandardOutPath</key><string>${LOG}</string>
  <key>StandardErrorPath</key><string>${LOG}</string>
  <key>RunAtLoad</key><false/>
</dict>
</plist>
PLIST
}

case "${1:-status}" in
  install)
    write_plist
    launchctl bootout "gui/$(id -u)/${LABEL}" 2>/dev/null || true
    launchctl bootstrap "gui/$(id -u)" "$PLIST"
    echo "installed: ${LABEL} (fires 08:00 & 20:00 local)"
    echo "plist: $PLIST"
    echo "log:   $LOG"
    ;;
  uninstall)
    launchctl bootout "gui/$(id -u)/${LABEL}" 2>/dev/null || true
    rm -f "$PLIST"
    echo "uninstalled: ${LABEL}"
    ;;
  run-once)
    cd "$REPO" && exec "$PY" scripts/daily_autopilot.py --once
    ;;
  status)
    echo "== launchd =="; launchctl print "gui/$(id -u)/${LABEL}" 2>/dev/null | grep -E "state|program|Label" || echo "(not loaded)"
    echo "== autopilot state =="; cd "$REPO" && "$PY" scripts/daily_autopilot.py --status 2>/dev/null || echo "(no state yet)"
    echo "== last log =="; tail -n 15 "$LOG" 2>/dev/null || echo "(no log yet)"
    ;;
  *) echo "usage: $0 {install|uninstall|run-once|status}"; exit 1 ;;
esac
