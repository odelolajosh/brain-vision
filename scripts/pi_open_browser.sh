#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  pi_open_browser.sh — Open brainvision demo in Chromium on Pi display      ║
# ║                                                                              ║
# ║  Usage: ./scripts/pi_open_browser.sh [options]                             ║
# ║                                                                              ║
# ║  Options:                                                                    ║
# ║    --host HOST      Pi hostname or IP                                        ║
# ║    --user USER      Pi username                                              ║
# ║    --port PORT      SSH port                                                 ║
# ║    --kiosk          Open in kiosk mode (fullscreen, no browser chrome)      ║
# ║    --close          Close the browser instead of opening it                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/pi_config.sh"

KIOSK=false
CLOSE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)   PI_HOST="$2";  SSH_TARGET="${PI_USER}@${PI_HOST}"; shift 2 ;;
        --user)   PI_USER="$2";  SSH_TARGET="${PI_USER}@${PI_HOST}"; shift 2 ;;
        --port)   PI_PORT="$2";  shift 2 ;;
        --kiosk)  KIOSK=true;    shift ;;
        --close)  CLOSE=true;    shift ;;
        *) error "Unknown option: $1" ;;
    esac
done

rebuild_ssh_cmd

APP_URL="http://localhost:${STREAMLIT_PORT}"

# ── Close browser ─────────────────────────────────────────────────────────────
if $CLOSE; then
    step "Closing Chromium on Pi"
    $SSH_CMD $SSH_OPTS "$SSH_TARGET" \
        "DISPLAY=:0 pkill -f chromium-browser || pkill -f chromium || true"
    success "Browser closed"
    exit 0
fi

# ── Check Streamlit is running ────────────────────────────────────────────────
step "Checking Streamlit is running"
PID_CHECK=$($SSH_CMD $SSH_OPTS "$SSH_TARGET" \
    "[ -f '${STREAMLIT_PID_FILE}' ] && \
     kill -0 \$(cat '${STREAMLIT_PID_FILE}') 2>/dev/null && \
     echo running || echo stopped")

if [[ "$PID_CHECK" != "running" ]]; then
    warn "Streamlit does not appear to be running."
    echo "  Start it first: ./scripts/pi_launch.sh start"
    echo "  Then retry:     ./scripts/pi_open_browser.sh"
    exit 1
fi
success "Streamlit is running"

# ── Check DISPLAY is available ────────────────────────────────────────────────
step "Checking Pi display"
DISPLAY_CHECK=$($SSH_CMD $SSH_OPTS "$SSH_TARGET" \
    "DISPLAY=:0 xdpyinfo &>/dev/null && echo ok || echo none" 2>/dev/null || echo none)

if [[ "$DISPLAY_CHECK" != "ok" ]]; then
    warn "No display found on the Pi (DISPLAY=:0)."
    echo "  Ensure the Pi is booted to desktop (not headless)."
    echo "  If running headless, connect a monitor or enable VNC."
    exit 1
fi
success "Display available"

# ── Close any existing Chromium ───────────────────────────────────────────────
step "Launching Chromium"
$SSH_CMD $SSH_OPTS "$SSH_TARGET" \
    "DISPLAY=:0 pkill -f chromium 2>/dev/null || true; sleep 1"

# ── Build Chromium flags ──────────────────────────────────────────────────────
CHROMIUM_FLAGS=(
    "--no-first-run"
    "--disable-infobars"
    "--disable-session-crashed-bubble"
    "--disable-restore-session-state"
    "--noerrdialogs"
    "--start-maximized"
)

$KIOSK && CHROMIUM_FLAGS+=("--kiosk")

FLAGS_STR="${CHROMIUM_FLAGS[*]}"

# ── Launch Chromium on Pi display ─────────────────────────────────────────────
$SSH_CMD $SSH_OPTS "$SSH_TARGET" "
DISPLAY=:0 nohup chromium-browser ${FLAGS_STR} '${APP_URL}' \
    > /tmp/chromium.log 2>&1 &
sleep 2
if pgrep -f chromium > /dev/null; then
    echo 'OK'
else
    echo 'FAILED'
    cat /tmp/chromium.log
fi
" | grep -q "OK" && success "Chromium launched at ${APP_URL}" \
                 || { warn "Chromium may not have launched — check /tmp/chromium.log on Pi"; }

echo
echo -e "  ${BOLD}brainvision demo is live on the Pi display${NC}"
$KIOSK && echo "  Mode: kiosk (fullscreen)" || echo "  Mode: maximised window"
echo "  URL:  ${APP_URL}"
echo
echo -e "  To close:  ${CYAN}./scripts/pi_open_browser.sh --close${NC}"