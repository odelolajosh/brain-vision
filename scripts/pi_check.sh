#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  pi_check.sh — Pre-flight checks before deploying to Raspberry Pi          ║
# ║                                                                              ║
# ║  Usage: ./scripts/pi_check.sh [--host HOST] [--user USER] [--port PORT]    ║
# ║                                                                              ║
# ║  Checks:                                                                     ║
# ║    1. SSH reachability                                                       ║
# ║    2. Python version (>= 3.10)                                               ║
# ║    3. Available disk space (>= 512 MB)                                       ║
# ║    4. Port 8501 availability                                                 ║
# ║    5. rsync availability on Pi                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/pi_config.sh"
export PI_PASSWORD="${PI_PASSWORD:-}"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --host) PI_HOST="$2"; SSH_TARGET="${PI_USER}@${PI_HOST}"; shift 2 ;;
        --user) PI_USER="$2"; SSH_TARGET="${PI_USER}@${PI_HOST}"; shift 2 ;;
        --port) PI_PORT="$2"; SSH_OPTS="-p ${PI_PORT} -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new "; shift 2 ;;
        *) error "Unknown option: $1" ;;
    esac
done

rebuild_ssh_cmd

PASS=0; FAIL=0

check_pass() { success "$1"; ((PASS++)) || true; }
check_fail() { warn    "$1"; ((FAIL++)) || true; }

echo -e "${BOLD}"
echo "╔══════════════════════════════════════════════════════╗"
echo "║         brainvision — Pi Pre-flight Checks          ║"
echo "╚══════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo -e "  Target : ${BOLD}${SSH_TARGET}${NC} (port ${PI_PORT})"
echo

# ── 1. Local SSH key or password ─────────────────────────────────────────────
step "Check 1 — Auth method"
if [[ -n "${PI_PASSWORD}" ]]; then
    check_pass "Password auth configured via PI_PASSWORD"
elif ls ~/.ssh/id_*.pub &>/dev/null; then
    check_pass "SSH key found: $(ls ~/.ssh/id_*.pub | head -1)"
else
    check_fail "No SSH key and PI_PASSWORD not set"
    echo "         Option A (recommended): ssh-keygen -t ed25519 && ssh-copy-id ${SSH_TARGET}"
    echo "         Option B             : export PI_PASSWORD=yourpassword"
fi

# ── 2. SSH reachability ───────────────────────────────────────────────────────
step "Check 2 — SSH connection"
if $SSH_CMD $SSH_OPTS "$SSH_TARGET" "echo ok" &>/dev/null; then
    check_pass "SSH connection successful"
else
    check_fail "Cannot connect to ${SSH_TARGET}"
    echo "         Ensure the Pi is on, SSH is enabled, and your key is copied:"
    echo "         ssh-copy-id -i ~/.ssh/id_ed25519.pub ${SSH_TARGET}"
    echo
    echo -e "  ${YELLOW}Remaining checks skipped — fix SSH first.${NC}"
    echo
    echo -e "  ${RED}${FAIL} check(s) failed.${NC}"
    exit 1
fi

# ── 3. Python version ─────────────────────────────────────────────────────────
step "Check 3 — Python version (need >= 3.${MIN_PYTHON_MINOR})"
PY_VERSION=$($SSH_CMD $SSH_OPTS "$SSH_TARGET" \
    "python3 -c 'import sys; print(sys.version_info.minor)'")
if [[ "$PY_VERSION" -ge "$MIN_PYTHON_MINOR" ]]; then
    FULL_VER=$($SSH_CMD $SSH_OPTS "$SSH_TARGET" "python3 --version")
    check_pass "Python version OK: ${FULL_VER}"
else
    check_fail "Python 3.${MIN_PYTHON_MINOR}+ required, found Python 3.${PY_VERSION}"
fi

# ── 4. Disk space ─────────────────────────────────────────────────────────────
step "Check 4 — Disk space (need >= ${MIN_DISK_MB} MB free)"
FREE_MB=$($SSH_CMD $SSH_OPTS "$SSH_TARGET" \
    "df -m ~ | tail -1 | awk '{print \$4}'")
if [[ "$FREE_MB" -ge "$MIN_DISK_MB" ]]; then
    check_pass "Disk space OK: ${FREE_MB} MB free"
else
    check_fail "Low disk space: ${FREE_MB} MB free (need ${MIN_DISK_MB} MB)"
    echo "         PyTorch CPU wheel is ~200 MB. Free up space on the Pi."
fi

# ── 5. Port 8501 ──────────────────────────────────────────────────────────────
step "Check 5 — Streamlit port ${STREAMLIT_PORT}"
PORT_IN_USE=$($SSH_CMD $SSH_OPTS "$SSH_TARGET" \
    "ss -tlnp 2>/dev/null | grep ':${STREAMLIT_PORT}' || echo ''")
if [[ -z "$PORT_IN_USE" ]]; then
    check_pass "Port ${STREAMLIT_PORT} is free"
else
    check_fail "Port ${STREAMLIT_PORT} is already in use"
    echo "         An existing Streamlit process may be running."
    echo "         Run: ./scripts/pi_launch.sh stop"
fi

# ── 6. rsync on Pi ────────────────────────────────────────────────────────────
step "Check 6 — rsync on Pi"
if $SSH_CMD $SSH_OPTS "$SSH_TARGET" "command -v rsync" &>/dev/null; then
    check_pass "rsync available on Pi"
else
    check_fail "rsync not found on Pi — installing..."
    $SSH_CMD $SSH_OPTS "$SSH_TARGET" "sudo apt-get install -y rsync -q" \
        && check_pass "rsync installed" \
        || check_fail "Could not install rsync — install manually: sudo apt install rsync"
fi

# ── 7. pip on Pi ─────────────────────────────────────────────────────────────
step "Check 7 — pip on Pi"
if $SSH_CMD $SSH_OPTS "$SSH_TARGET" "python3 -m pip --version" &>/dev/null; then
    check_pass "pip available"
else
    check_fail "pip not found — installing..."
    $SSH_CMD $SSH_OPTS "$SSH_TARGET" "sudo apt-get install -y python3-pip -q" \
        && check_pass "pip installed" \
        || check_fail "Could not install pip"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo
echo "──────────────────────────────────────────────────────"
echo -e "  ${GREEN}${PASS} check(s) passed${NC}   ${RED}${FAIL} check(s) failed${NC}"
echo "──────────────────────────────────────────────────────"
echo

if [[ "$FAIL" -gt 0 ]]; then
    echo -e "  ${YELLOW}Fix the failed checks before deploying.${NC}"
    exit 1
else
    echo -e "  ${GREEN}Pi is ready for deployment.${NC}"
    exit 0
fi
