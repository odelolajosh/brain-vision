#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  deploy.sh — Full brainvision demo deployment orchestrator                 ║
# ║                                                                              ║
# ║  Usage: ./scripts/deploy.sh [options]                                      ║
# ║                                                                              ║
# ║  Options:                                                                    ║
# ║    --host HOST          Pi hostname or IP  [default: joshua.local]          ║
# ║    --user USER          Pi username        [default: joshua]                ║
# ║    --port PORT          SSH port           [default: 22]                    ║
# ║    --mode MODE          fast | full        [default: fast]                  ║
# ║    --checkpoint FILE    .pt file to deploy (repeatable)                     ║
# ║    --image FILE         preprocessed .npz to deploy (repeatable)           ║
# ║    --raw-image FILE     raw .npz to deploy → datasets/ (repeatable)        ║
# ║    --all-checkpoints    Deploy all .pt files from checkpoints/              ║
# ║    --force              Overwrite existing assets without prompting         ║
# ║    --skip-check         Skip pre-flight checks                              ║
# ║    --skip-env           Skip environment setup (already done)               ║
# ║    --skip-code          Skip code sync                                      ║
# ║    --skip-assets        Skip asset sync                                     ║
# ║    --skip-launch        Don't start Streamlit after deploy                  ║
# ║    --restart            Restart Streamlit even if already running           ║
# ║    --dry-run            Preview actions without executing                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/pi_config.sh"
export PI_PASSWORD="${PI_PASSWORD:-}"

# ── Defaults ──────────────────────────────────────────────────────────────────
CHECKPOINTS=()
IMAGES=()
RAW_IMAGES=()
ALL_CHECKPOINTS=false
FORCE=false
SKIP_CHECK=false
SKIP_ENV=false
SKIP_CODE=false
SKIP_ASSETS=false
SKIP_LAUNCH=false
RESTART=false
DRY_RUN=false

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)             PI_HOST="$2";        SSH_TARGET="${PI_USER}@${PI_HOST}"; shift 2 ;;
        --user)             PI_USER="$2";        SSH_TARGET="${PI_USER}@${PI_HOST}"; shift 2 ;;
        --port)             PI_PORT="$2";        shift 2 ;;
        --mode)             DEPLOY_MODE="$2";    shift 2 ;;
        --checkpoint)       CHECKPOINTS+=("$2"); shift 2 ;;
        --image)            IMAGES+=("$2");       shift 2 ;;
        --raw-image)        RAW_IMAGES+=("$2");   shift 2 ;;
        --all-checkpoints)  ALL_CHECKPOINTS=true; shift ;;
        --force)            FORCE=true;           shift ;;
        --skip-check)       SKIP_CHECK=true;      shift ;;
        --skip-env)         SKIP_ENV=true;        shift ;;
        --skip-code)        SKIP_CODE=true;       shift ;;
        --skip-assets)      SKIP_ASSETS=true;     shift ;;
        --skip-launch)      SKIP_LAUNCH=true;     shift ;;
        --restart)          RESTART=true;         shift ;;
        --dry-run)          DRY_RUN=true;         shift ;;
        -h|--help)
            sed -n '2,28p' "$0" | sed 's/^# *//'
            exit 0 ;;
        *) error "Unknown option: $1 — run with --help for usage" ;;
    esac
done

rebuild_ssh_cmd

# ── Banner ────────────────────────────────────────────────────────────────────
echo -e "${BOLD}"
echo "╔══════════════════════════════════════════════════════╗"
echo "║        brainvision — Full Deployment to Pi 5        ║"
echo "╚══════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo -e "  Target       : ${BOLD}${PI_USER}@${PI_HOST}${NC} (port ${PI_PORT})"
echo -e "  Mode         : ${BOLD}${DEPLOY_MODE}${NC}"
echo -e "  Checkpoints  : ${#CHECKPOINTS[@]} specified"
echo -e "  Images       : ${#IMAGES[@]} specified"
echo -e "  Dry run      : ${DRY_RUN}"
echo
echo -e "  Steps:"
echo -e "    Pre-flight checks  : $( $SKIP_CHECK && echo 'SKIP' || echo 'YES')"
echo -e "    Environment setup  : $( $SKIP_ENV   && echo 'SKIP' || echo 'YES')"
echo -e "    Code sync          : $( $SKIP_CODE  && echo 'SKIP' || echo 'YES')"
echo -e "    Asset sync         : $( $SKIP_ASSETS && echo 'SKIP' || echo 'YES')"
echo -e "    Launch Streamlit   : $( $SKIP_LAUNCH && echo 'SKIP' || echo 'YES')"
echo

# ── Verify local repo structure ───────────────────────────────────────────────
[[ -d "brainvision" ]] || error "brainvision/ not found. Run from repo root."
[[ -f "demo/app.py" ]] || error "demo/app.py not found."

# ── Build pass-through argument lists ────────────────────────────────────────
COMMON_ARGS=(--host "$PI_HOST" --user "$PI_USER" --port "$PI_PORT")
DRY_FLAG=(); $DRY_RUN && DRY_FLAG=(--dry-run)
FORCE_FLAG=(); $FORCE && FORCE_FLAG=(--force)

CKPT_ARGS=()
for c in ${CHECKPOINTS[@]+"${CHECKPOINTS[@]}"}; do CKPT_ARGS+=(--checkpoint "$c"); done
$ALL_CHECKPOINTS && CKPT_ARGS+=(--all-checkpoints)

IMG_ARGS=()
for i in ${IMAGES[@]+"${IMAGES[@]}"}; do IMG_ARGS+=(--image "$i"); done
for i in ${RAW_IMAGES[@]+"${RAW_IMAGES[@]}"}; do IMG_ARGS+=(--raw-image "$i"); done

STARTED_AT=$(date +%s)

# ── Step 1: Pre-flight checks ─────────────────────────────────────────────────
if ! $SKIP_CHECK; then
    step "Step 1 — Pre-flight checks"
    "${SCRIPT_DIR}/pi_check.sh" "${COMMON_ARGS[@]}" \
        || error "Pre-flight checks failed. Fix issues before deploying."
else
    warn "Skipping pre-flight checks (--skip-check)"
fi

# ── Step 2: Environment setup ─────────────────────────────────────────────────
if ! $SKIP_ENV; then
    step "Step 2 — Environment setup"
    "${SCRIPT_DIR}/pi_env.sh" "${COMMON_ARGS[@]}"
else
    warn "Skipping environment setup (--skip-env)"
fi

# ── Step 3: Sync code ─────────────────────────────────────────────────────────
if ! $SKIP_CODE; then
    step "Step 3 — Sync code (mode: ${DEPLOY_MODE})"
    "${SCRIPT_DIR}/pi_sync_code.sh" \
        "${COMMON_ARGS[@]}" \
        --mode "$DEPLOY_MODE" \
        ${DRY_FLAG[@]+"${DRY_FLAG[@]}"}
else
    warn "Skipping code sync (--skip-code)"
fi

# ── Step 4: Sync assets ───────────────────────────────────────────────────────
if ! $SKIP_ASSETS; then
    if [[ ${#CKPT_ARGS[@]} -gt 0 || ${#IMG_ARGS[@]} -gt 0 ]]; then
        step "Step 4 — Sync assets"
        "${SCRIPT_DIR}/pi_sync_assets.sh" \
            "${COMMON_ARGS[@]}" \
            ${CKPT_ARGS[@]+"${CKPT_ARGS[@]}"} \
            ${IMG_ARGS[@]+"${IMG_ARGS[@]}"} \
            ${DRY_FLAG[@]+"${DRY_FLAG[@]}"} \
            ${FORCE_FLAG[@]+"${FORCE_FLAG[@]}"}
    else
        warn "No assets specified — skipping asset sync"
        warn "Use --checkpoint FILE or --image FILE to deploy assets"
    fi
else
    warn "Skipping asset sync (--skip-assets)"
fi

# ── Step 5: Launch Streamlit ──────────────────────────────────────────────────
if ! $SKIP_LAUNCH && ! $DRY_RUN; then
    step "Step 5 — Launch Streamlit"
    if $RESTART; then
        "${SCRIPT_DIR}/pi_launch.sh" restart "${COMMON_ARGS[@]}"
    else
        "${SCRIPT_DIR}/pi_launch.sh" start "${COMMON_ARGS[@]}"
    fi
else
    $DRY_RUN && warn "Dry run — skipping launch" \
             || warn "Skipping launch (--skip-launch)"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
ELAPSED=$(( $(date +%s) - STARTED_AT ))

echo
echo -e "${GREEN}${BOLD}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}${BOLD}║              Deployment complete ✅                  ║${NC}"
echo -e "${GREEN}${BOLD}╚══════════════════════════════════════════════════════╝${NC}"
echo
echo -e "  Time elapsed : ${ELAPSED}s"
if ! $SKIP_LAUNCH && ! $DRY_RUN; then
    PI_IP=$($SSH_CMD $SSH_OPTS "${PI_USER}@${PI_HOST}" "hostname -I | awk '{print \$1}'" 2>/dev/null || echo "${PI_HOST}")
    echo -e "  Demo URL     : ${BOLD}http://${PI_IP}:${STREAMLIT_PORT}${NC}"
    echo
    echo -e "  Useful commands:"
    echo -e "    ${CYAN}./scripts/pi_launch.sh status${NC}   — check if running"
    echo -e "    ${CYAN}./scripts/pi_launch.sh logs${NC}     — stream live logs"
    echo -e "    ${CYAN}./scripts/pi_launch.sh restart${NC}  — restart Streamlit"
fi
echo