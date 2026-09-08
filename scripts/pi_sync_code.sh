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
# ║    --no-stubs       Never stub package __init__ files (assume all deps      ║
# ║                     installed on the Pi)                                     ║
# ║    --force-stubs    Always stub, regardless of what is installed             ║
# ║    --dry-run        Show what would be synced without doing it               ║
# ║                                                                              ║
# ║  Stubbing policy:                                                            ║
# ║    The real brainvision/__init__.py imports the training stack (sklearn,     ║
# ║    scipy); models/__init__.py imports every architecture (einops). On a      ║
# ║    minimal Pi those imports fail, so this script can replace them with       ║
# ║    empty stubs. That is a fallback, not the goal — a stubbed                 ║
# ║    models/__init__.py hides SpectralFormer and friends from anything that    ║
# ║    imports the package normally.                                             ║
# ║                                                                              ║
# ║    By default the Pi is PROBED for einops / sklearn / scipy and each         ║
# ║    __init__ is stubbed only if its dependencies are actually missing. In     ║
# ║    fast mode the models stub is always applied, because the model files      ║
# ║    themselves are not synced.                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/pi_config.sh"
export PI_PASSWORD="${PI_PASSWORD:-}"

DRY_RUN=false
NO_STUBS=false
FORCE_STUBS=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)        PI_HOST="$2";     SSH_TARGET="${PI_USER}@${PI_HOST}"; shift 2 ;;
        --user)        PI_USER="$2";     SSH_TARGET="${PI_USER}@${PI_HOST}"; shift 2 ;;
        --port)        PI_PORT="$2";     shift 2 ;;
        --mode)        DEPLOY_MODE="$2"; shift 2 ;;
        --no-stubs)    NO_STUBS=true;    shift ;;
        --force-stubs) FORCE_STUBS=true; shift ;;
        --dry-run)     DRY_RUN=true;     shift ;;
        *) error "Unknown option: $1" ;;
    esac
done

rebuild_ssh_cmd

[[ "$DEPLOY_MODE" == "fast" || "$DEPLOY_MODE" == "full" ]] \
    || error "Invalid mode '$DEPLOY_MODE'. Use 'fast' or 'full'."

$NO_STUBS && $FORCE_STUBS \
    && error "--no-stubs and --force-stubs are mutually exclusive."

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

# In fast mode, exclude the heavier architectures
if [[ "$DEPLOY_MODE" == "fast" ]]; then
    EXCLUDES+=(
        "--exclude=brainvision/models/fabelo_2dcnn.py"
        "--exclude=brainvision/models/simple_2dcnn.py"
        "--exclude=brainvision/models/lee_2dcnn.py"
        "--exclude=brainvision/models/hamida_3dcnn.py"
        "--exclude=brainvision/models/hybridsn.py"
        "--exclude=brainvision/models/spectralformer.py"
    )
    info "Fast mode: 1D models only (2D/3D/transformer excluded)"
else
    info "Full mode: all model files synced"
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

# ── Decide which __init__ files need stubbing ────────────────────────────────
# The sync above has just restored the REAL __init__.py files. Only replace
# them where the Pi genuinely cannot satisfy their imports.
step "Checking Pi dependencies"

if $DRY_RUN; then
    info "[dry-run] Skipping dependency probe and stub decision"
    STUB_PACKAGE=false
    STUB_MODELS=false
else
    DEPS=$($SSH_CMD $SSH_OPTS "$SSH_TARGET" "
        source '${VENV_DIR}/bin/activate' 2>/dev/null || true
        for m in einops sklearn scipy; do
            python3 -c \"import \$m\" 2>/dev/null && echo \"\$m=yes\" || echo \"\$m=no\"
        done" 2>/dev/null || echo "")

    HAS_EINOPS=$(echo "$DEPS"  | grep '^einops=' | cut -d= -f2)
    HAS_SKLEARN=$(echo "$DEPS" | grep '^sklearn=' | cut -d= -f2)
    HAS_SCIPY=$(echo "$DEPS"   | grep '^scipy='  | cut -d= -f2)
    HAS_EINOPS="${HAS_EINOPS:-no}"
    HAS_SKLEARN="${HAS_SKLEARN:-no}"
    HAS_SCIPY="${HAS_SCIPY:-no}"

    echo "    einops  : ${HAS_EINOPS}   (needed by spectralformer.py)"
    echo "    sklearn : ${HAS_SKLEARN}   (needed by metrics.py)"
    echo "    scipy   : ${HAS_SCIPY}   (needed by preprocessing.py)"

    # brainvision/__init__.py pulls in metrics (sklearn) and preprocessing (scipy)
    if $NO_STUBS; then
        STUB_PACKAGE=false
    elif $FORCE_STUBS; then
        STUB_PACKAGE=true
    elif [[ "$HAS_SKLEARN" == "yes" && "$HAS_SCIPY" == "yes" ]]; then
        STUB_PACKAGE=false
    else
        STUB_PACKAGE=true
    fi

    # models/__init__.py imports every architecture, so it needs both einops
    # AND every model file present — which fast mode deliberately breaks.
    if $NO_STUBS; then
        STUB_MODELS=false
    elif $FORCE_STUBS; then
        STUB_MODELS=true
    elif [[ "$DEPLOY_MODE" == "fast" ]]; then
        STUB_MODELS=true
    elif [[ "$HAS_EINOPS" == "yes" ]]; then
        STUB_MODELS=false
    else
        STUB_MODELS=true
    fi
fi

# ── Apply (or skip) the package stub ─────────────────────────────────────────
if $DRY_RUN; then
    :
elif $STUB_PACKAGE; then
    step "Stubbing brainvision/__init__.py"
    $SCP_CMD -p $SCP_OPTS \
        "${SCRIPT_DIR}/_pi_init_stub.py" \
        "${SSH_TARGET}:${REMOTE_DIR}/brainvision/__init__.py"
    warn "brainvision/__init__.py replaced with a stub"
    echo "       Reason: sklearn=${HAS_SKLEARN}, scipy=${HAS_SCIPY} on the Pi."
    echo "       Submodules still import directly "
    echo "       (from brainvision.constants import ...), which is what the"
    echo "       demo and latency scripts do. To keep the real file instead:"
    echo "         pip install scikit-learn scipy   # on the Pi"
    echo "         ./scripts/pi_sync_code.sh --mode full"
else
    success "brainvision/__init__.py kept (dependencies satisfied)"
fi

# ── Apply (or skip) the models stub ──────────────────────────────────────────
if $DRY_RUN; then
    :
elif $STUB_MODELS; then
    step "Stubbing brainvision/models/__init__.py"
    $SCP_CMD -p $SCP_OPTS \
        "${SCRIPT_DIR}/_pi_models_init_stub.py" \
        "${SSH_TARGET}:${REMOTE_DIR}/brainvision/models/__init__.py"
    warn "brainvision/models/__init__.py replaced with a stub"
    if [[ "$DEPLOY_MODE" == "fast" ]]; then
        echo "       Reason: fast mode does not sync the 2D/3D/transformer"
        echo "       model files, so the real __init__ would fail to import."
        echo "       Use --mode full to deploy and expose every architecture."
    else
        echo "       Reason: einops is not installed on the Pi, so"
        echo "       spectralformer.py cannot be imported."
        echo "       Fix:  pip install einops   # on the Pi, then re-sync"
    fi
else
    success "brainvision/models/__init__.py kept — all architectures importable"
fi

# ── Verify the package actually imports on the Pi ────────────────────────────
if ! $DRY_RUN; then
    step "Verifying imports on the Pi"
    IMPORT_CHECK=$($SSH_CMD $SSH_OPTS "$SSH_TARGET" "
        cd '${REMOTE_DIR}'
        source '${VENV_DIR}/bin/activate' 2>/dev/null || true
        PYTHONPATH='${REMOTE_DIR}' python3 - <<'PYEOF'
mods = [
    ('constants',            'brainvision.constants'),
    ('fabelo_dnn',           'brainvision.models.fabelo_dnn'),
    ('baseline_dnn',         'brainvision.models.baseline_dnn'),
    ('hu_1dcnn',             'brainvision.models.hu_1dcnn'),
    ('fabelo_2dcnn',         'brainvision.models.fabelo_2dcnn'),
    ('simple_2dcnn',         'brainvision.models.simple_2dcnn'),
    ('lee_2dcnn',            'brainvision.models.lee_2dcnn'),
    ('hybridsn',             'brainvision.models.hybridsn'),
    ('spectralformer',       'brainvision.models.spectralformer'),
]
for name, mod in mods:
    try:
        __import__(mod)
        print('  OK      ' + name)
    except Exception as e:
        print('  MISSING ' + name + '  (' + type(e).__name__ + ': ' + str(e)[:60] + ')')
PYEOF" 2>/dev/null || echo "  (verification could not run)")
    echo "$IMPORT_CHECK"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo
if $DRY_RUN; then
    warn "Dry run — no files were transferred."
else
    success "Code sync complete"
    echo
    echo -e "  Next: ${CYAN}./scripts/pi_sync_assets.sh${NC} to push checkpoints/images"
fi