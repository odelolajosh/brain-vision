#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  pi_sync_code.sh — Sync brainvision package and demo app to Pi             ║
# ║                                                                              ║
# ║  Usage: ./scripts/pi_sync_code.sh [options]                                ║
# ║                                                                              ║
# ║  Options:                                                                    ║
# ║    --host HOST      Pi hostname or IP                                        ║
# ║    --user USER      Pi username                                              ║
# ║    --port PORT      SSH port                                                 ║
# ║    --mode MODE      fast (1D only) | full (all models) [default: fast]      ║
# ║    --dry-run        Show what would be synced without doing it               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/pi_config.sh"
export PI_PASSWORD="${PI_PASSWORD:-}"

DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)    PI_HOST="$2";       SSH_TARGET="${PI_USER}@${PI_HOST}"; shift 2 ;;
        --user)    PI_USER="$2";       SSH_TARGET="${PI_USER}@${PI_HOST}"; shift 2 ;;
        --port)    PI_PORT="$2";       SSH_OPTS="-p ${PI_PORT} -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new "; shift 2 ;;
        --mode)    DEPLOY_MODE="$2";   shift 2 ;;
        --dry-run) DRY_RUN=true;       shift ;;
        *) error "Unknown option: $1" ;;
    esac
done

rebuild_ssh_cmd

[[ "$DEPLOY_MODE" == "fast" || "$DEPLOY_MODE" == "full" ]] \
    || error "Invalid mode '$DEPLOY_MODE'. Use 'fast' or 'full'."

RSYNC_DRY=""
$DRY_RUN && RSYNC_DRY="--dry-run"

echo -e "${BOLD}"
echo "╔══════════════════════════════════════════════════════╗"
echo "║           brainvision — Sync Code to Pi             ║"
echo "╚══════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo -e "  Target  : ${BOLD}${SSH_TARGET}${NC}"
echo -e "  Mode    : ${BOLD}${DEPLOY_MODE}${NC}"
echo -e "  Dry run : ${DRY_RUN}"
echo

# ── Verify we are at repo root ────────────────────────────────────────────────
[[ -d "brainvision" ]] || error "brainvision/ not found. Run from repo root."
[[ -f "demo/app.py" ]] || error "demo/app.py not found. Run from repo root."

# ── SSH check ────────────────────────────────────────────────────────────────
$SSH_CMD $SSH_OPTS "$SSH_TARGET" "echo ok" > /dev/null \
    || error "Cannot reach ${SSH_TARGET}"

# ── Ensure remote dirs exist ──────────────────────────────────────────────────
step "Preparing remote directories"
$SSH_CMD $SSH_OPTS "$SSH_TARGET" \
    "mkdir -p ${REMOTE_DIR}/brainvision/models \
              ${REMOTE_DIR}/brainvision/data \
              ${REMOTE_DIR}/demo"
success "Remote directories ready"

# ── Build exclude list for rsync ─────────────────────────────────────────────
# Always exclude training-only files and __pycache__
EXCLUDES=(
    "--exclude=__pycache__/"
    "--exclude=*.pyc"
    "--exclude=*.pyo"
    "--exclude=.DS_Store"
    "--exclude=notebooks/"
    "--exclude=results/"
    "--exclude=checkpoints/"   # handled by pi_sync_assets.sh
    "--exclude=processed/"     # handled by pi_sync_assets.sh
)

# In fast mode, exclude 2D, 3D, and transformer models
if [[ "$DEPLOY_MODE" == "fast" ]]; then
    EXCLUDES+=(
        "--exclude=brainvision/models/fabelo_2dcnn.py"
        "--exclude=brainvision/models/simple_2dcnn.py"
        "--exclude=brainvision/models/lee_2dcnn.py"
        "--exclude=brainvision/models/hamida_3dcnn.py"
        "--exclude=brainvision/models/hybridsn.py"
        "--exclude=brainvision/models/spectralformer.py"
    )
    info "Fast mode: excluding 2D/3D/transformer model files"
fi

# ── Sync brainvision package ──────────────────────────────────────────────────
step "Syncing brainvision package"
$RSYNC_CMD -avz --progress $RSYNC_DRY \
    "${EXCLUDES[@]}" \
    -e "$SSH_CMD $SSH_OPTS" \
    brainvision/ \
    "${SSH_TARGET}:${REMOTE_DIR}/brainvision/"

success "brainvision package synced"

# ── Sync demo app ─────────────────────────────────────────────────────────────
step "Syncing demo/"
$RSYNC_CMD -avz --progress $RSYNC_DRY \
    "${EXCLUDES[@]}" \
    -e "$SSH_CMD $SSH_OPTS" \
    demo/ \
    "${SSH_TARGET}:${REMOTE_DIR}/demo/"

success "demo/ synced"



# ── Replace brainvision/__init__.py with demo-safe stub ──────────────────────
# The full __init__.py imports sklearn which is not installed on Pi.
step "Patching brainvision/__init__.py for demo"
$SCP_CMD -p $SCP_OPTS \
    "${SCRIPT_DIR}/_pi_init_stub.py" \
    "${SSH_TARGET}:${REMOTE_DIR}/brainvision/__init__.py"
success "brainvision/__init__.py patched (sklearn imports removed)"

# ── Replace brainvision/models/__init__.py with demo-safe stub ───────────────
step "Patching brainvision/models/__init__.py for demo"
$SCP_CMD -p $SCP_OPTS \
    "${SCRIPT_DIR}/_pi_models_init_stub.py" \
    "${SSH_TARGET}:${REMOTE_DIR}/brainvision/models/__init__.py"
success "brainvision/models/__init__.py patched (einops/heavy imports removed)"


# ── Summary ───────────────────────────────────────────────────────────────────
echo
if $DRY_RUN; then
    warn "Dry run — no files were transferred."
else
    success "Code sync complete"
    echo
    echo -e "  Next: ${CYAN}./scripts/pi_sync_assets.sh${NC} to push checkpoints/images"
fi