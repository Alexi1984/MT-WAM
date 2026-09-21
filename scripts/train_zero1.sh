set -euo pipefail

NPROC_PER_NODE="${1:?Usage: bash scripts/train_zero1.sh <processes> --config-name=train_libero}"
shift
EXTRA_ARGS=("$@")
NUM_MACHINES="${MTWAM_NNODES:-1}"
MACHINE_RANK="${MTWAM_NODE_RANK:-0}"
MAIN_PROCESS_IP="${MASTER_ADDR:-127.0.0.1}"
MAIN_PROCESS_PORT="${MASTER_PORT:-29500}"
CONFIG_NAME="train_libero"

for value in "$NPROC_PER_NODE" "$NUM_MACHINES" "$MACHINE_RANK"; do
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "Process, machine and rank values must be integers." >&2
    exit 1
  fi
done
if (( NPROC_PER_NODE < 1 || NUM_MACHINES < 1 || MACHINE_RANK >= NUM_MACHINES )); then
  echo "Invalid process count, machine count or machine rank." >&2
  exit 1
fi

for ((i = 0; i < ${#EXTRA_ARGS[@]}; i++)); do
  case "${EXTRA_ARGS[$i]}" in
    --config-name)
      if (( i + 1 >= ${#EXTRA_ARGS[@]} )); then
        echo "--config-name requires a value." >&2
        exit 1
      fi
      CONFIG_NAME="${EXTRA_ARGS[$((i + 1))]%.yaml}"
      ;;
    --config-name=*)
      CONFIG_NAME="${EXTRA_ARGS[$i]#--config-name=}"
      CONFIG_NAME="${CONFIG_NAME%.yaml}"
      ;;
  esac
done
if [[ "$CONFIG_NAME" != "train_libero" && "$CONFIG_NAME" != "train_robotwin" ]]; then
  echo "Select train_libero or train_robotwin." >&2
  exit 1
fi
if (( NUM_MACHINES > 1 )) && [[ -z "${MTWAM_RUN_ID:-}" ]]; then
  echo "Set the same MTWAM_RUN_ID on every machine." >&2
  exit 1
fi

RUN_ID="${MTWAM_RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
ACCEL_CONFIG="${MTWAM_ACCEL_CONFIG:-scripts/accelerate_configs/accelerate_zero1_ds.yaml}"

accelerate launch \
  --config_file "$ACCEL_CONFIG" \
  --num_processes "$((NPROC_PER_NODE * NUM_MACHINES))" \
  --num_machines "$NUM_MACHINES" \
  --machine_rank "$MACHINE_RANK" \
  --main_process_ip "$MAIN_PROCESS_IP" \
  --main_process_port "$MAIN_PROCESS_PORT" \
  scripts/train.py \
  "output_dir=./runs/${CONFIG_NAME}/${RUN_ID}" \
  "wandb.name=${CONFIG_NAME}" \
  "${EXTRA_ARGS[@]}"
