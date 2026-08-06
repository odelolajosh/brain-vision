#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  pi_sync_assets.sh — Copy model checkpoints and HSI images to Pi           ║
# ║                                                                              ║
# ║  Usage: ./scripts/pi_sync_assets.sh [options]                              ║
# ║                                                                              ║
# ║  Options:                                                                    ║
# ║    --host HOST            Pi hostname or IP                                  ║
# ║    --user USER            Pi username                                        ║
# ║    --port PORT            SSH port                                           ║
# ║    --checkpoint FILE      .pt checkpoint to copy (repeatable)               ║
# ║    --image FILE           .npz processed cube to copy (repeatable)          ║
# ║    --all-checkpoints      Copy all .pt files from checkpoints/              ║
# ║    --force                Overwrite existing files without prompting        ║
# ║    --dry-run              Show what would be copied without doing it        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/pi_config.sh"
export PI_PASSWORD="${PI_PASSWORD:-}"

CHECKPOINTS=()
PROCESSED_IMAGES=()
RAW_IMAGES=()
ALL_CHECKPOINTS=false
FORCE=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)             PI_HOST="$2";       SSH_TARGET="${PI_USER}@${PI_HOST}"; shift 2 ;;
        --user)             PI_USER="$2";       SSH_TARGET="${PI_USER}@${PI_HOST}"; shift 2 ;;
        --port)             PI_PORT="$2";       SSH_OPTS="-p ${PI_PORT} -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new "; shift 2 ;;
        --checkpoint)       CHECKPOINTS+=("$2"); shift 2 ;;
        --image)            PROCESSED_IMAGES+=("$2"); shift 2 ;;
        --raw-image)        RAW_IMAGES+=("$2");       shift 2 ;;
        --all-checkpoints)  ALL_CHECKPOINTS=true; shift ;;
        --force)            FORCE=true;          shift ;;
        --dry-run)          DRY_RUN=true;        shift ;;
        *) error "Unknown option: $1" ;;
    esac
done

rebuild_ssh_cmd

echo -e "${BOLD}"
echo "╔══════════════════════════════════════════════════════╗"
echo "║          brainvision — Sync Assets to Pi            ║"
echo "╚══════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo -e "  Target  : ${BOLD}${SSH_TARGET}${NC}"
echo -e "  Dry run : ${DRY_RUN}"
echo

# ── Collect all checkpoints if requested ──────────────────────────────────────
if $ALL_CHECKPOINTS; then
    [[ -d "checkpoints" ]] || error "checkpoints/ directory not found."
    mapfile -t EXTRA < <(find checkpoints -name "*.pt" | sort)
    CHECKPOINTS+=(${EXTRA[@]+"${EXTRA[@]}"})
    info "Found ${#EXTRA[@]} checkpoint(s) in checkpoints/"
fi

# ── Nothing to do? ────────────────────────────────────────────────────────────
if [[ ${#CHECKPOINTS[@]} -eq 0 && ${#PROCESSED_IMAGES[@]} -eq 0 && ${#RAW_IMAGES[@]} -eq 0 ]]; then
    warn "Nothing to sync. Specify assets with --checkpoint, --image, or --raw-image."
    echo
    echo "  Examples:"
    echo "    --checkpoint checkpoints/1dnnfabelo_ce_bal_fold2_vpfabelo.pt"
    echo "    --image processed/first_campaign/008-02.npz       (preprocessed)"
    echo "    --raw-image datasets/first_campaign/008-02.npz    (raw)"
    echo "    --all-checkpoints"
    exit 0
fi

# ── SSH check ────────────────────────────────────────────────────────────────
$SSH_CMD $SSH_OPTS "$SSH_TARGET" "echo ok" > /dev/null \
    || error "Cannot reach ${SSH_TARGET}"

# ── Ensure remote asset directories exist ────────────────────────────────────
step "Preparing remote asset directories"
$SSH_CMD $SSH_OPTS "$SSH_TARGET" \
    "mkdir -p ${REMOTE_CKPT_DIR} ${REMOTE_PROCESSED_DIR} ${REMOTE_DATASETS_DIR}"
success "Remote asset directories ready"

# ── Helper: safe copy with size check and overwrite guard ────────────────────
copy_asset() {
    local local_path="$1"
    local remote_dir="$2"
    local label="$3"

    [[ -f "$local_path" ]] || { warn "File not found: $local_path — skipping"; return; }

    local filename
    filename="$(basename "$local_path")"
    local size_mb
    size_mb=$(du -m "$local_path" | cut -f1)

    # Check if file already exists on Pi
    local exists
    exists=$($SSH_CMD $SSH_OPTS "$SSH_TARGET" \
        "[ -f '${remote_dir}/${filename}' ] && echo yes || echo no")

    if [[ "$exists" == "yes" ]] && ! $FORCE && ! $DRY_RUN; then
        echo -ne "  ${YELLOW}${filename}${NC} already exists on Pi. Overwrite? [y/N] "
        read -r answer
        [[ "$answer" =~ ^[Yy]$ ]] || { info "Skipping ${filename}"; return; }
    fi

    if $DRY_RUN; then
        info "[dry-run] Would copy ${label}: ${local_path} → ${remote_dir}/ (${size_mb} MB)"
        return
    fi

    info "Copying ${label}: ${filename} (${size_mb} MB)…"
    $SCP_CMD -p $SCP_OPTS "$local_path" "${SSH_TARGET}:${remote_dir}/${filename}"
    success "${filename} → ${remote_dir}/"
}

# ── Copy checkpoints ──────────────────────────────────────────────────────────
if [[ ${#CHECKPOINTS[@]} -gt 0 ]]; then
    step "Copying ${#CHECKPOINTS[@]} checkpoint(s)"
    for ckpt in "${CHECKPOINTS[@]}"; do
        copy_asset "$ckpt" "$REMOTE_CKPT_DIR" "checkpoint"
    done
fi

# ── Helper: resolve campaign subdir ──────────────────────────────────────────
campaign_subdir() {
    local path="$1"
    local base="$2"
    if [[ "$path" == *"first_campaign"* ]];  then echo "${base}/first_campaign"
    elif [[ "$path" == *"second_campaign"* ]]; then echo "${base}/second_campaign"
    elif [[ "$path" == *"third_campaign"* ]];  then echo "${base}/third_campaign"
    else echo "${base}"
    fi
}

# ── Copy preprocessed images (.npz from processed/) ──────────────────────────
if [[ ${#PROCESSED_IMAGES[@]} -gt 0 ]]; then
    step "Copying ${#PROCESSED_IMAGES[@]} preprocessed image(s) → processed/"
    for img in ${PROCESSED_IMAGES[@]+"${PROCESSED_IMAGES[@]}"}; do
        remote_subdir=$(campaign_subdir "$img" "${REMOTE_PROCESSED_DIR}")
        if ! $DRY_RUN; then
            $SSH_CMD $SSH_OPTS "$SSH_TARGET" "mkdir -p ${remote_subdir}"
        fi
        copy_asset "$img" "$remote_subdir" "preprocessed image"
    done
fi

# ── Copy raw images (.npz from datasets/) ────────────────────────────────────
if [[ ${#RAW_IMAGES[@]} -gt 0 ]]; then
    step "Copying ${#RAW_IMAGES[@]} raw image(s) → datasets/"
    for img in ${RAW_IMAGES[@]+"${RAW_IMAGES[@]}"}; do
        remote_subdir=$(campaign_subdir "$img" "${REMOTE_DATASETS_DIR}")
        if ! $DRY_RUN; then
            $SSH_CMD $SSH_OPTS "$SSH_TARGET" "mkdir -p ${remote_subdir}"
        fi
        copy_asset "$img" "$remote_subdir" "raw image"
    done
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo
if $DRY_RUN; then
    warn "Dry run — no files were transferred."
else
    success "Asset sync complete"
    echo
    # List what's on Pi
    info "Checkpoints on Pi:"
    $SSH_CMD $SSH_OPTS "$SSH_TARGET" \
        "ls -lh ${REMOTE_CKPT_DIR}/*.pt 2>/dev/null | awk '{print \"    \" \$5 \"  \" \$9}' || echo '    (none)'"
    info "Preprocessed images on Pi:"
    $SSH_CMD $SSH_OPTS "$SSH_TARGET" \
        "find ${REMOTE_PROCESSED_DIR} -name '*.npz' 2>/dev/null | sed 's|^|    |' || echo '    (none)'"
    info "Raw images on Pi:"
    $SSH_CMD $SSH_OPTS "$SSH_TARGET" \
        "find ${REMOTE_DATASETS_DIR} -name '*.npz' 2>/dev/null | sed 's|^|    |' || echo '    (none)'"
    echo
    echo -e "  Next: ${CYAN}./scripts/pi_launch.sh start${NC}"
fi