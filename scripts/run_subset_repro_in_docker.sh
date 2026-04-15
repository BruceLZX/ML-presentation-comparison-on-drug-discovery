#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME=${IMAGE_NAME:-admet-ai-repro}
ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
DATASETS=${DATASETS:-"caco2_wang hia_hou"}

cd "$ROOT_DIR"

mkdir -p repro_data/raw repro_data/prepared repro_results

docker build -f Dockerfile.repro -t "$IMAGE_NAME" .

docker run --rm -v "$PWD":/opt/admet_ai -w /opt/admet_ai "$IMAGE_NAME" \
  python scripts/prepare_tdc_admet_subset.py \
  --raw_data_dir repro_data/raw \
  --save_dir repro_data/prepared \
  --datasets $DATASETS

docker run --rm -v "$PWD":/opt/admet_ai -w /opt/admet_ai "$IMAGE_NAME" \
  python scripts/train_tdc_admet_group.py \
  --data_dir repro_data/prepared \
  --save_dir repro_results/models \
  --model_type chemprop

docker run --rm -v "$PWD":/opt/admet_ai -w /opt/admet_ai "$IMAGE_NAME" \
  python scripts/evaluate_tdc_admet_group.py \
  --data_dir repro_data/raw \
  --preds_dir repro_results/models/chemprop
