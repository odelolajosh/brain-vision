#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  pi_env.sh — Set up Python virtual environment on Raspberry Pi             ║
# ║                                                                              ║
# ║  Usage: ./scripts/pi_env.sh [--host HOST] [--user USER] [--force]          ║
# ║                                                                              ║
# ║  Idempotent — safe to re-run. Skips steps already completed.                ║
# ║  --force recreates the venv from scratch.                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/pi_config.sh"
export PI_PASSWORD="${PI_PASSWORD:-}"

FORCE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)  PI_HOST="$2";  SSH_TARGET="${PI_USER}@${PI_HOST}"; shift 2 ;;
        --user)  PI_USER="$2";  SSH_TARGET="${PI_USER}@${PI_HOST}"; shift 2 ;;
        --port)  PI_PORT="$2";  SSH_OPTS="-p ${PI_PORT} -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new "; shift 2 ;;
        --force) FORCE=true; shift ;;
        *) error "Unknown option: $1" ;;
    esac
done

rebuild_ssh_cmd

echo -e "${BOLD}"
echo "╔══════════════════════════════════════════════════════╗"
echo "║         brainvision — Pi Environment Setup          ║"
echo "╚══════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo -e "  Target : ${BOLD}${SSH_TARGET}${NC}"
echo -e "  Venv   : ${BOLD}${VENV_DIR}${NC}"
echo -e "  Force  : ${FORCE}"
echo

# ── Verify SSH ────────────────────────────────────────────────────────────────
step "Verifying SSH connection"
$SSH_CMD $SSH_OPTS "$SSH_TARGET" "echo ok" > /dev/null \
    || error "Cannot reach ${SSH_TARGET} — run pi_check.sh first."
success "Connected"

# ── Build the remote setup script ────────────────────────────────────────────
step "Running environment setup on Pi"

REMOTE_SETUP=$(cat <<REMOTE
set -e

REMOTE_DIR="${REMOTE_DIR}"
VENV_DIR="${VENV_DIR}"
FORCE="${FORCE}"

echo ""
echo "── Python version ──────────────────────────────────"
python3 --version

# ── Create directories ────────────────────────────────────────────────────────
mkdir -p "\${REMOTE_DIR}/checkpoints"
mkdir -p "\${REMOTE_DIR}/data"
mkdir -p "\${REMOTE_DIR}/demo"
mkdir -p "\${REMOTE_DIR}/brainvision/models"
mkdir -p "\${REMOTE_DIR}/brainvision/data"
echo "Directories ready"

# ── Virtual environment ───────────────────────────────────────────────────────
echo ""
echo "── Virtual environment ─────────────────────────────"
if [ "\${FORCE}" = "true" ] && [ -d "\${VENV_DIR}" ]; then
    echo "Force flag set — removing existing venv..."
    rm -rf "\${VENV_DIR}"
fi

if [ ! -d "\${VENV_DIR}" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "\${VENV_DIR}"
    echo "Created: \${VENV_DIR}"
else
    echo "Venv already exists — skipping creation"
fi

source "\${VENV_DIR}/bin/activate"
pip install --upgrade pip --quiet
echo "pip: \$(pip --version)"

# ── PyTorch CPU wheel ─────────────────────────────────────────────────────────
echo ""
echo "── PyTorch (CPU) ───────────────────────────────────"
if python3 -c "import torch; print('torch', torch.__version__)" 2>/dev/null; then
    echo "Already installed — skipping"
else
    echo "Downloading CPU-only PyTorch wheel (this may take a few minutes)..."
    pip install torch \
        --index-url https://download.pytorch.org/whl/cpu \
        --quiet
    python3 -c "import torch; print('Installed torch', torch.__version__)"
fi

# ── Demo requirements ─────────────────────────────────────────────────────────
echo ""
echo "── Demo requirements ───────────────────────────────"
REQ_FILE="\${REMOTE_DIR}/demo/requirements_demo.txt"
if [ -f "\${REQ_FILE}" ]; then
    pip install -r "\${REQ_FILE}" --quiet
    echo "Requirements installed from \${REQ_FILE}"
else
    echo "requirements_demo.txt not found — installing core packages directly"
    echo "(this happens on a first deploy, before code sync has run)"
    pip install streamlit numpy matplotlib einops --quiet
fi

# ── Verify imports ────────────────────────────────────────────────────────────
echo ""
echo "── Verifying imports ───────────────────────────────"
python3 -c "
import torch, streamlit, numpy, matplotlib
print('  torch      :', torch.__version__)
print('  streamlit  :', streamlit.__version__)
print('  numpy      :', numpy.__version__)
print('  matplotlib :', matplotlib.__version__)
"

# ── Write launch helper ───────────────────────────────────────────────────────
echo ""
echo "── Writing launch script ───────────────────────────"
cat > "\${REMOTE_DIR}/launch.sh" << 'LAUNCH'
#!/usr/bin/env bash
cd "\$(dirname "\$0")"
source .venv/bin/activate
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
IP=\$(hostname -I | awk '{print \$1}')
echo ""
echo "  brainvision demo starting..."
echo "  Open: http://\${IP}:8501"
echo ""
streamlit run demo/app.py \\
    --server.address 0.0.0.0 \\
    --server.port 8501 \\
    --server.headless true \\
    --browser.gatherUsageStats false
LAUNCH
chmod +x "\${REMOTE_DIR}/launch.sh"
echo "launch.sh written"

echo ""
echo "✅ Environment setup complete"
echo ""
echo "  To start the demo:"
echo "    cd \${REMOTE_DIR} && source .venv/bin/activate && ./launch.sh"
REMOTE
)

$SSH_CMD $SSH_OPTS "$SSH_TARGET" "bash -s" <<< "$REMOTE_SETUP"

success "Environment ready on Pi"
echo
echo -e "  Next: ${CYAN}./scripts/pi_sync_code.sh${NC} to push the codebase"