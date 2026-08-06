#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  pi_launch.sh — Manage the Streamlit process on Raspberry Pi               ║
# ║                                                                              ║
# ║  Usage: ./scripts/pi_launch.sh <command> [options]                         ║
# ║                                                                              ║
# ║  Commands:                                                                   ║
# ║    start    Start Streamlit (if not already running)                        ║
# ║    stop     Stop Streamlit                                                   ║
# ║    restart  Stop then start                                                  ║
# ║    status   Show whether Streamlit is running and its URL                   ║
# ║    logs     Tail the Streamlit log file                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/pi_config.sh"
export PI_PASSWORD="${PI_PASSWORD:-}"

COMMAND="${1:-status}"
shift || true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host) PI_HOST="$2"; SSH_TARGET="${PI_USER}@${PI_HOST}"; shift 2 ;;
        --user) PI_USER="$2"; SSH_TARGET="${PI_USER}@${PI_HOST}"; shift 2 ;;
        --port) PI_PORT="$2"; SSH_OPTS="-p ${PI_PORT} -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new "; shift 2 ;;
        *) error "Unknown option: $1" ;;
    esac
done

rebuild_ssh_cmd

REMOTE_LOG="${REMOTE_DIR}/streamlit.log"

# ── SSH check ────────────────────────────────────────────────────────────────
$SSH_CMD $SSH_OPTS "$SSH_TARGET" "echo ok" > /dev/null \
    || error "Cannot reach ${SSH_TARGET}"

# ── Remote command helpers ────────────────────────────────────────────────────
remote() { $SSH_CMD $SSH_OPTS "$SSH_TARGET" "$@"; }

pi_status() {
    remote "
PID_FILE='${STREAMLIT_PID_FILE}'
if [ -f \"\${PID_FILE}\" ]; then
    PID=\$(cat \"\${PID_FILE}\")
    if kill -0 \"\${PID}\" 2>/dev/null; then
        IP=\$(hostname -I | awk '{print \$1}')
        echo \"RUNNING \${PID} http://\${IP}:${STREAMLIT_PORT}\"
    else
        echo 'STALE'
        rm -f \"\${PID_FILE}\"
    fi
else
    echo 'STOPPED'
fi"
}

pi_stop() {
    remote "
PID_FILE='${STREAMLIT_PID_FILE}'
if [ -f \"\${PID_FILE}\" ]; then
    PID=\$(cat \"\${PID_FILE}\")
    if kill -0 \"\${PID}\" 2>/dev/null; then
        kill \"\${PID}\"
        sleep 1
        kill -0 \"\${PID}\" 2>/dev/null && kill -9 \"\${PID}\" || true
        echo \"Stopped PID \${PID}\"
    else
        echo 'Process not running (stale PID file)'
    fi
    rm -f \"\${PID_FILE}\"
else
    echo 'Streamlit is not running'
fi"
}

pi_start() {
    remote "
set -e
PID_FILE='${STREAMLIT_PID_FILE}'
REMOTE_DIR='${REMOTE_DIR}'
VENV_DIR='${VENV_DIR}'
LOG_FILE='${REMOTE_LOG}'
PORT='${STREAMLIT_PORT}'

# Check not already running
if [ -f \"\${PID_FILE}\" ]; then
    PID=\$(cat \"\${PID_FILE}\")
    if kill -0 \"\${PID}\" 2>/dev/null; then
        IP=\$(hostname -I | awk '{print \$1}')
        echo \"Already running (PID \${PID}) at http://\${IP}:\${PORT}\"
        exit 0
    else
        rm -f \"\${PID_FILE}\"
    fi
fi

# Check venv
if [ ! -f \"\${VENV_DIR}/bin/activate\" ]; then
    echo 'Virtual environment not found. Run pi_env.sh first.'
    exit 1
fi

# Check app exists
if [ ! -f \"\${REMOTE_DIR}/demo/app.py\" ]; then
    echo 'demo/app.py not found. Run pi_sync_code.sh first.'
    exit 1
fi

cd \"\${REMOTE_DIR}\"
source \"\${VENV_DIR}/bin/activate\"
export PYTHONPATH=\"\${REMOTE_DIR}:\${PYTHONPATH:-}\"

# Start detached
nohup streamlit run \"\${REMOTE_DIR}/demo/app.py\" \\
    --server.address 0.0.0.0 \\
    --server.port \"\${PORT}\" \\
    --server.headless true \\
    --browser.gatherUsageStats false \\
    > \"\${LOG_FILE}\" 2>&1 &

PID=\$!
echo \$PID > \"\${PID_FILE}\"
sleep 2

if kill -0 \"\${PID}\" 2>/dev/null; then
    IP=\$(hostname -I | awk '{print \$1}')
    echo \"Started (PID \${PID})\"
    echo \"URL: http://\${IP}:\${PORT}\"
    echo \"Log: \${LOG_FILE}\"
else
    echo 'Streamlit failed to start — check the log:'
    tail -20 \"\${LOG_FILE}\"
    exit 1
fi"
}

# ── Command dispatch ──────────────────────────────────────────────────────────
case "$COMMAND" in

    status)
        step "Streamlit status on ${SSH_TARGET}"
        STATUS=$(pi_status)
        if [[ "$STATUS" == RUNNING* ]]; then
            PID=$(echo "$STATUS" | awk '{print $2}')
            URL=$(echo "$STATUS" | awk '{print $3}')
            success "Streamlit is running (PID ${PID})"
            echo -e "  URL: ${BOLD}${URL}${NC}"
        elif [[ "$STATUS" == "STALE" ]]; then
            warn "Stale PID file removed — Streamlit is not running"
        else
            info "Streamlit is not running"
        fi
        ;;

    start)
        step "Starting Streamlit on ${SSH_TARGET}"
        pi_start
        echo
        echo -e "  ${GREEN}${BOLD}Streamlit started.${NC}"
        echo -e "  Open the URL above in your browser."
        ;;

    stop)
        step "Stopping Streamlit on ${SSH_TARGET}"
        pi_stop
        success "Done"
        ;;

    restart)
        step "Restarting Streamlit on ${SSH_TARGET}"
        pi_stop
        sleep 1
        pi_start
        echo
        success "Streamlit restarted"
        ;;

    logs)
        step "Streaming logs from ${SSH_TARGET}"
        info "Press Ctrl+C to stop"
        echo
        $SSH_CMD $SSH_OPTS "$SSH_TARGET" "tail -f ${REMOTE_LOG}" || true
        ;;

    *)
        error "Unknown command: ${COMMAND}. Use: start | stop | restart | status | logs"
        ;;
esac