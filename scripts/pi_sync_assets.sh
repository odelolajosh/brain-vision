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
    # Portable alternative to `mapfile` (bash 4+ only; macOS ships bash 3.2)
    EXTRA=()
    while IFS= read -r _ckpt; do
        [[ -n "$_ckpt" ]] && EXTRA+=("$_ckpt")
    done < <(find checkpoints -name "*.pt" | sort)
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

# ── Helper: batch copy many files in ONE rsync connection ────────────────────
# Copying files one-by-one with scp opens a new SSH connection per file, which
# trips sshd's MaxStartups limit after ~10 rapid connections ("Permission
# denied"). rsync moves the whole set over a single connection instead.
copy_batch() {
    local remote_dir="$1"; shift
    local label="$1";      shift
    local files=("$@")

    [[ ${#files[@]} -gt 0 ]] || return

    # Filter out anything missing locally
    local present=()
    local f
    for f in "${files[@]}"; do
        if [[ -f "$f" ]]; then present+=("$f")
        else warn "File not found: $f — skipping"; fi
    done
    [[ ${#present[@]} -gt 0 ]] || return

    local total_mb
    total_mb=$(du -ch "${present[@]}" 2>/dev/null | tail -1 | cut -f1)

    if $DRY_RUN; then
        info "[dry-run] Would copy ${#present[@]} ${label}(s) (${total_mb}) → ${remote_dir}/"
        return
    fi

    local update_flag=()
    if ! $FORCE; then
        # Without --force, skip files already present and newer/identical on Pi
        update_flag=(--ignore-existing)
        info "Existing files on the Pi will be kept (use --force to overwrite)"
    fi

    info "Copying ${#present[@]} ${label}(s), ${total_mb} total…"
    # NOTE: --progress (not --info=progress2) for compatibility with the
    # rsync 2.6.9 / openrsync that ships with macOS.
    $RSYNC_CMD -az --progress \
        ${update_flag[@]+"${update_flag[@]}"} \
        -e "$SSH_CMD $SSH_OPTS" \
        "${present[@]}" \
        "${SSH_TARGET}:${remote_dir}/"
    success "${#present[@]} ${label}(s) → ${remote_dir}/"
}

# ── Copy checkpoints ──────────────────────────────────────────────────────────
if [[ ${#CHECKPOINTS[@]} -gt 0 ]]; then
    step "Copying ${#CHECKPOINTS[@]} checkpoint(s)"
    copy_batch "$REMOTE_CKPT_DIR" "checkpoint" "${CHECKPOINTS[@]}"
fi

# ── Helper: batch-copy images, grouped by campaign ───────────────────────────
copy_images() {
    local base_dir="$1"; shift
    local label="$1";    shift
    local images=("$@")

    [[ ${#images[@]} -gt 0 ]] || return

    # Group by campaign so each campaign is a single rsync transfer
    local campaign sub group img
    for campaign in first_campaign second_campaign third_campaign _other; do
        group=()
        for img in "${images[@]}"; do
            if [[ "$campaign" == "_other" ]]; then
                [[ "$img" != *first_campaign*   && \
                   "$img" != *second_campaign*  && \
                   "$img" != *third_campaign*   ]] && group+=("$img")
            elif [[ "$img" == *"${campaign}"* ]]; then
                group+=("$img")
            fi
        done
        [[ ${#group[@]} -gt 0 ]] || continue

        if [[ "$campaign" == "_other" ]]; then
            sub="${base_dir}"
        else
            sub="${base_dir}/${campaign}"
        fi

        if ! $DRY_RUN; then
            $SSH_CMD $SSH_OPTS "$SSH_TARGET" "mkdir -p ${sub}"
        fi
        copy_batch "$sub" "$label" "${group[@]}"
    done
}

# ── Copy preprocessed images (.npz from processed/) ──────────────────────────
if [[ ${#PROCESSED_IMAGES[@]} -gt 0 ]]; then
    step "Copying ${#PROCESSED_IMAGES[@]} preprocessed image(s) → processed/"
    copy_images "$REMOTE_PROCESSED_DIR" "preprocessed image" \
        "${PROCESSED_IMAGES[@]}"
fi

# ── Copy raw images (.npz from datasets/) ────────────────────────────────────
if [[ ${#RAW_IMAGES[@]} -gt 0 ]]; then
    step "Copying ${#RAW_IMAGES[@]} raw image(s) → datasets/"
    copy_images "$REMOTE_DATASETS_DIR" "raw image" \
        "${RAW_IMAGES[@]}"
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