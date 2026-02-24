#!/usr/bin/env bash
set -euo pipefail

# basic paths (adjust if needed)
CALIB_DIR="calib_wavs"
ENCODER="msd_musicnn.onnx"
HEAD="deam_head.onnx"
MEAN="emb_mean.npy"
STD="emb_std.npy"
TEST_WAV="test.wav"
MERGED_IN="merged_std_final.onnx"
MERGED_CORRECT="merged_std_correct.onnx"
MERGED_FIXED="merged_fixed.onnx"

echo
echo "1) compute calibration stats (if missing)..."
if [ ! -f "$MEAN" ] || [ ! -f "$STD" ]; then
  python compute_calib_stats.py --calib "$CALIB_DIR" --out-mean "$MEAN" --out-std "$STD" --verbose
else
  echo "  -> emb_mean.npy and emb_std.npy present, skipping compute."
fi

echo
echo "2) encoder -> standardize -> head sanity check (FP32) ..."
python run_head_with_std.py --encoder "$ENCODER" --head "$HEAD" --mean "$MEAN" --std "$STD" --test "$TEST_WAV"

echo
echo "3) choose merged model (prefer corrected if present)..."
if [ -f "$MERGED_CORRECT" ]; then
  echo "  -> Found $MERGED_CORRECT; using it."
  MERGED="$MERGED_CORRECT"
else
  MERGED="$MERGED_IN"
  echo "  -> $MERGED_CORRECT not found, will try to fix $MERGED_IN -> $MERGED_FIXED"
  python fix_emb_initializers.py --model "$MERGED_IN" --out "$MERGED_FIXED"
  MERGED="$MERGED_FIXED"
fi

echo
echo "4) run merged FP32 model (quick smoke test) ..."
python run_fp32_correct.py "$MERGED" "$TEST_WAV"

echo
echo "5) run detailed parity verifier (encoder+head vs merged) ..."
python merged_verifier.py \
  --encoder "$ENCODER" \
  --head "$HEAD" \
  --merged "$MERGED" \
  --mean "$MEAN" \
  --std "$STD" \
  --test "$TEST_WAV"

echo
echo "Done. If parity is good (L2 small / per-element close) — proceed to quantization."