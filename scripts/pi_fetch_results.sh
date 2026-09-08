#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  pi_fetch_results.sh — Copy measurement outputs from the Pi to the Mac     ║
# ║                                                                              ║
# ║  Usage: ./scripts/pi_fetch_results.sh [options]                            ║
# ║                                                                              ║
# ║  Options:                                                                    ║
# ║    --host HOST      Pi hostname or IP                                        ║
# ║    --user USER      Pi username                                              ║
# ║    --port PORT      SSH port                                                 ║
# ║    --dest DIR       Local destination [default: results/pi]                  ║
# ║    --pattern GLOB   Only fetch matching files [default: all]                 ║
# ║    --flat           Merge into results/ instead of results/pi/               ║
# ║    --dry-run        Show what would be fetched without doing it              ║
# ║                                                                              ║
# ║  Pi outputs land in a separate directory by default so that                 ║
# ║  latency_pi.* does not sit alongside latency_mps.* under the same name      ║
# ║  and get confused during write-up.                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/pi_config.sh"
export PI_PASSWORD="${PI_PASSWORD:-}"

DEST="results/pi"
PATTERN=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)    PI_HOST="$2"; SSH_TARGET="${PI_USER}@${PI_HOST}"; shift 2 ;;
        --user)    PI_USER="$2"; SSH_TARGET="${PI_USER}@${PI_HOST}"; shift 2 ;;
        --port)    PI_PORT="$2"; shift 2 ;;
        --dest)    DEST="$2";    shift 2 ;;
        --pattern) PATTERN="$2"; shift 2 ;;
        --flat)    DEST="results"; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) error "Unknown option: $1" ;;
    esac
done

rebuild_ssh_cmd

REMOTE_RESULTS="${REMOTE_DIR}/results"

echo -e "${BOLD}"
echo "╔══════════════════════════════════════════════════════╗"
echo "║        brainvision — Fetch Results from Pi          ║"
echo "╚══════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo -e "  Source  : ${BOLD}${SSH_TARGET}:${REMOTE_RESULTS}${NC}"
echo -e "  Dest    : ${BOLD}${DEST}${NC}"
[[ -n "$PATTERN" ]] && echo -e "  Pattern : ${BOLD}${PATTERN}${NC}"
echo -e "  Dry run : ${DRY_RUN}"
echo

# ── SSH check ────────────────────────────────────────────────────────────────
$SSH_CMD $SSH_OPTS "$SSH_TARGET" "echo ok" > /dev/null \
    || error "Cannot reach ${SSH_TARGET}"

# ── Does the remote results directory exist and hold anything? ───────────────
step "Checking remote results"
REMOTE_LIST=$($SSH_CMD $SSH_OPTS "$SSH_TARGET" \
    "ls -1 ${REMOTE_RESULTS} 2>/dev/null || true")

if [[ -z "$REMOTE_LIST" ]]; then
    warn "No files found in ${REMOTE_RESULTS}"
    echo "  Run a measurement on the Pi first, e.g.:"
    echo "    PYTHONPATH=. python demo/measure_latency.py \\"
    echo "        --image processed/first_campaign/008-02.npz"
    exit 0
fi

echo "$REMOTE_LIST" | sed 's|^|    |'
N_REMOTE=$(echo "$REMOTE_LIST" | wc -l | tr -d ' ')
success "${N_REMOTE} file(s) on the Pi"

# ── Fetch ────────────────────────────────────────────────────────────────────
mkdir -p "$DEST"

RSYNC_ARGS=(-avz --progress)
$DRY_RUN && RSYNC_ARGS+=(--dry-run)
[[ -n "$PATTERN" ]] && RSYNC_ARGS+=(--include="$PATTERN" --include="*/" --exclude="*")

step "Fetching to ${DEST}/"
$RSYNC_CMD "${RSYNC_ARGS[@]}" \
    -e "$SSH_CMD $SSH_OPTS" \
    "${SSH_TARGET}:${REMOTE_RESULTS}/" \
    "${DEST}/"

# ── Summary ──────────────────────────────────────────────────────────────────
echo
if $DRY_RUN; then
    warn "Dry run — nothing was copied."
else
    success "Fetched to ${DEST}/"
    echo
    info "Local contents:"
    ls -lh "$DEST" | tail -n +2 | awk '{print "    " $5 "  " $9}'
    echo
    echo -e "  Pi latency data is now at ${CYAN}${DEST}/latency_pi.json${NC}"
    echo -e "  Point plot_pareto.py at it, or update the CONFIGS block there."
fi