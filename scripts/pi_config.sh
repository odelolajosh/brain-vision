#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  pi_config.sh — Shared configuration for all brainvision Pi scripts        ║
# ║                                                                              ║
# ║  This file is sourced by every script. Edit defaults here, or override      ║
# ║  per-script via flags. This file is gitignored.                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── Connection ─────────────────────────────────────────────────────────────────
PI_HOST="${PI_HOST:-raspberrypi.local}"
PI_USER="${PI_USER:-joshua}"
PI_PORT="${PI_PORT:-22}"

# ── Remote paths ───────────────────────────────────────────────────────────────
REMOTE_DIR="${REMOTE_DIR:-/home/${PI_USER}/brainvision_demo}"
VENV_DIR="${VENV_DIR:-${REMOTE_DIR}/.venv}"
REMOTE_CKPT_DIR="${REMOTE_CKPT_DIR:-${REMOTE_DIR}/checkpoints}"
REMOTE_PROCESSED_DIR="${REMOTE_PROCESSED_DIR:-${REMOTE_DIR}/processed}"   # preprocessed .npz files
REMOTE_DATASETS_DIR="${REMOTE_DATASETS_DIR:-${REMOTE_DIR}/datasets}"      # raw .npz files

# ── Streamlit ──────────────────────────────────────────────────────────────────
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"
STREAMLIT_PID_FILE="${STREAMLIT_PID_FILE:-${REMOTE_DIR}/.streamlit.pid}"

# ── Deployment mode ────────────────────────────────────────────────────────────
DEPLOY_MODE="${DEPLOY_MODE:-fast}"

# ── Python ─────────────────────────────────────────────────────────────────────
MIN_PYTHON_MINOR=10
MIN_DISK_MB=512

# ── Auth ───────────────────────────────────────────────────────────────────────
# Option A — SSH key (recommended, no setup needed here):
#   ssh-keygen -t ed25519
#   ssh-copy-id joshua@raspberrypi.local
#
# Option B — Password via sshpass:
#   export PI_PASSWORD="yourpassword"
#   brew install hudochenkov/sshpass/sshpass   # macOS
#   sudo apt install sshpass                   # Linux
#
# Option C — Interactive password prompt (no sshpass, no key):
#   Leave PI_PASSWORD unset. Scripts will prompt for password each time.
PI_PASSWORD="${PI_PASSWORD:-}"

# ── Derived: SSH target ────────────────────────────────────────────────────────
SSH_TARGET="${PI_USER}@${PI_HOST}"
SSH_OPTS="-p ${PI_PORT} -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new"
SCP_OPTS="-P ${PI_PORT} -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new"

# ── Derived: command wrappers (sshpass if password set, plain otherwise) ───────
if [[ -n "${PI_PASSWORD}" ]]; then
    if command -v sshpass &>/dev/null; then
        SSH_CMD="sshpass -p ${PI_PASSWORD} ssh"
        SCP_CMD="sshpass -p ${PI_PASSWORD} scp"
        RSYNC_CMD="sshpass -p ${PI_PASSWORD} rsync"
    else
        echo -e "\033[1;33m[WARN]\033[0m  PI_PASSWORD set but sshpass not found."
        echo "         macOS : brew install hudochenkov/sshpass/sshpass"
        echo "         Linux : sudo apt install sshpass"
        echo "  Falling back to interactive password prompts (you will be asked each time)."
        SSH_CMD="ssh"
        SCP_CMD="scp"
        RSYNC_CMD="rsync"
    fi
else
    SSH_CMD="ssh"
    SCP_CMD="scp"
    RSYNC_CMD="rsync"
fi

# ── Colours ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }
step()    { echo -e "\n${BOLD}── $* ──${NC}"; }

# ── Call this after argument parsing to rebuild SSH_CMD with current PI_PASSWORD ──
rebuild_ssh_cmd() {
    SSH_TARGET="${PI_USER}@${PI_HOST}"
    # ControlMaster multiplexes all connections over one authenticated session.
    # Without it, dozens of rapid scp/ssh calls trip sshd's MaxStartups limit
    # and start getting "Permission denied".
    _CM="-o ControlMaster=auto -o ControlPath=/tmp/bv-%r@%h:%p -o ControlPersist=120"
    SSH_OPTS="-p ${PI_PORT} -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new ${_CM}"
    SCP_OPTS="-P ${PI_PORT} -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new ${_CM}"
    if [[ -n "${PI_PASSWORD}" ]]; then
        if command -v sshpass &>/dev/null; then
            SSH_CMD="sshpass -p ${PI_PASSWORD} ssh"
            SCP_CMD="sshpass -p ${PI_PASSWORD} scp"
            RSYNC_CMD="sshpass -p ${PI_PASSWORD} rsync"
        else
            SSH_CMD="ssh"
            SCP_CMD="scp"
            RSYNC_CMD="rsync"
        fi
    else
        SSH_CMD="ssh"
        SCP_CMD="scp"
        RSYNC_CMD="rsync"
    fi
}