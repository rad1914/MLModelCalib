#!/usr/bin/env bash
# @path: run.sh
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
abspath() { "$PYTHON" -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$1"; }

download_if_missing() {
  local url="$1"
  local out="$2"
  if [[ -s "$out" ]]; then
    echo "  ↳ skip: $out exists"
  else
    echo "  ↳ downloading: $out"
    wget -O "$out" "$url"
  fi
}

ENC="$(abspath "${1:-msd_musicnn.onnx}")"
HEAD="$(abspath "${2:-deam_head.onnx}")"
CALIB_DIR="$(abspath "${3:-calib_wavs}")"
AUDIO="$(abspath "${4:-test.wav}")"
OUTDIR="$(abspath "${OUTDIR:-out}")"
PREFIX="${PREFIX:-h_}"
TANH="${TANH:-1}"
mkdir -p "$OUTDIR"
MEAN="$OUTDIR/emb_mean.npy"
STD="$OUTDIR/emb_std.npy"
HEAD_CALIB_DIR="$OUTDIR/head_calib"
HEAD_Q="$OUTDIR/deam_head_q.onnx"
MERGED="$OUTDIR/final_merged.onnx"
MERGED_FIX="$OUTDIR/final_merged_fixed.onnx"
DEBUG="$OUTDIR/final_debug.onnx"

echo "[0/6] Download Model (skip if exists):"

download_if_missing \
  "https://essentia.upf.edu/models/feature-extractors/musicnn/msd-musicnn-1.onnx" \
  "msd_musicnn.onnx"

download_if_missing \
  "https://essentia.upf.edu/models/feature-extractors/musicnn/msd-musicnn-1.json" \
  "msd_musicnn.json"

download_if_missing \
  "https://essentia.upf.edu/models/classification-heads/deam/deam-msd-musicnn-2.onnx" \
  "deam_head.onnx"

download_if_missing \
  "https://essentia.upf.edu/models/classification-heads/deam/deam-msd-musicnn-2.json" \
  "deam_head.json"

echo "[1/6] Compute encoder calibration stats"
echo "Running compute_calib_stats.py..."
"$PYTHON" "$SCRIPT_DIR/compute_calib_stats.py" \
  -m "$ENC" \
  -c "$CALIB_DIR" \
  --out-mean "$MEAN" \
  --out-std "$STD" || { echo "[FAIL] calibration step"; exit 1; }
echo "[2/6] Build head calibration embeddings"
rm -rf "$HEAD_CALIB_DIR"
mkdir -p "$HEAD_CALIB_DIR"
(
  cd "$OUTDIR"
  "$PYTHON" "$SCRIPT_DIR/gen_head_calib.py" \
    "$ENC" \
    "$CALIB_DIR" \
    "$MEAN" \
    "$STD" || { echo "[FAIL] head calib generation"; exit 1; }
)
echo "[3/6] Quantize head"
"$PYTHON" "$SCRIPT_DIR/quantize_head.py" \
  "$HEAD" \
  "$HEAD_Q" \
  "$HEAD_CALIB_DIR" || { echo "[FAIL] quantization"; exit 1; }
echo "[4/6] Merge encoder + standardized quantized head"
MERGE_ARGS=("$ENC" "$HEAD_Q" "$MEAN" "$STD" "$MERGED" "--prefix" "$PREFIX")
if [[ "$TANH" != "0" && "$TANH" != "false" && "$TANH" != "False" ]]; then
  MERGE_ARGS+=("--tanh")
fi
"$PYTHON" "$SCRIPT_DIR/merging.py" "${MERGE_ARGS[@]}" || { echo "[FAIL] merge"; exit 1; }
echo "[5/6] Normalize opset imports"
"$PYTHON" "$SCRIPT_DIR/fix_opset.py" "$MERGED" "$MERGED_FIX" || { echo "[FAIL] opset fix"; exit 1; }
echo "[6/6] Validate merged model"
"$PYTHON" "$SCRIPT_DIR/validation.py" \
  --audio "$AUDIO" \
  --enc "$ENC" \
  --head "$HEAD_Q" \
  --merged "$MERGED_FIX" \
  --debug_out "$DEBUG" \
  --mean "$MEAN" \
  --std "$STD" || { echo "[FAIL] validation"; exit 1; }
echo "Done"
echo "Mean:   $MEAN"
echo "Std:    $STD"
echo "Head:   $HEAD_Q"
echo "Merged: $MERGED_FIX"
echo "Debug:  $DEBUG"