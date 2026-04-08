# @path: compute_calib_stats.py
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np

from audio_utils import (
    DEFAULT_FRAMES,
    DEFAULT_HOP,
    DEFAULT_N_FFT,
    DEFAULT_N_MELS,
    DEFAULT_POWER,
    DEFAULT_SR,
    STD_FLOOR,
    build_model_input_from_path,
    get_cpu_session,
)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute embedding mean/std from WAV calibration set")
    p.add_argument('--model', '-m', default='msd_musicnn.onnx', help='Path to encoder ONNX model')
    p.add_argument('--calib', '-c', default='calib_wavs', help='Directory containing WAV calibration files')
    p.add_argument('--out-mean', default='emb_mean.npy', help='Output .npy file for mean')
    p.add_argument('--out-std', default='emb_std.npy', help='Output .npy file for std')
    p.add_argument('--sr', type=int, default=DEFAULT_SR, help='Audio sampling rate (Hz)')
    p.add_argument('--n-fft', type=int, default=DEFAULT_N_FFT, help='STFT n_fft')
    p.add_argument('--hop', type=int, default=DEFAULT_HOP, help='STFT hop_length (samples)')
    p.add_argument('--n-mels', type=int, default=DEFAULT_N_MELS, help='Number of mel bins')
    p.add_argument('--frames', type=int, default=DEFAULT_FRAMES, help='Number of frames (time axis) expected by the model')
    p.add_argument('--power', type=float, default=DEFAULT_POWER, help='Power for mel spectrogram (2.0 for power)')
    p.add_argument('--min-samples', type=int, default=5, help='Minimum successfully embedded clips required')
    p.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    return p.parse_args()

def list_wavs(calib_dir: str) -> List[str]:
    return sorted(
        os.path.join(calib_dir, f)
        for f in os.listdir(calib_dir)
        if f.lower().endswith('.wav')
    )

def open_session(model_path: str):
    try:
        return get_cpu_session(model_path)
    except Exception as e:
        print("Error: Failed to open ONNX model:", e, file=sys.stderr)
        raise

def compute_stats_from_embeddings(emb_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, int]:
    embs = np.stack(emb_list, axis=0)
    mean = embs.mean(axis=0)
    std = embs.std(axis=0)
    std = np.maximum(std, STD_FLOOR)
    return mean.astype(np.float32), std.astype(np.float32), int(embs.shape[0])

def _output_dim_hint(output_shape) -> int | None:
    dims = []
    for dim in output_shape or []:
        try:
            dims.append(int(dim))
        except Exception:
            dims.append(None)

    for dim in reversed(dims):
        if dim is not None and dim > 0:
            return dim
    return None

def main():
    args = parse_args()

    if not os.path.isfile(args.model):
        print(f"ERROR: model not found: {args.model}", file=sys.stderr)
        sys.exit(2)
    if not os.path.isdir(args.calib):
        print(f"ERROR: calib dir not found: {args.calib}", file=sys.stderr)
        sys.exit(2)

    wavs = list_wavs(args.calib)
    if not wavs:
        print("ERROR: No WAV files found in calib directory. Please add representative audio.", file=sys.stderr)
        sys.exit(3)

    print(f"Loading ONNX model: {args.model}")
    sess = open_session(args.model)
    input_meta = sess.get_inputs()[0]
    in_name = input_meta.name
    in_shape = input_meta.shape
    out_meta = sess.get_outputs()[0]
    expected_dim = _output_dim_hint(out_meta.shape)

    if args.verbose:
        print("Model input name:", in_name, "shape:", in_shape)
        print("Model output name:", out_meta.name, "shape:", out_meta.shape)
        if expected_dim is not None:
            print("Expected embedding dim from model output metadata:", expected_dim)

    emb_list = []
    failed = 0
    total_patches = 0
    emb_dim = expected_dim

    for i, path in enumerate(wavs, 1):
        fname = os.path.basename(path)
        try:
            inp = build_model_input_from_path(
                path,
                in_shape,
                sr=args.sr,
                n_fft=args.n_fft,
                hop=args.hop,
                n_mels=args.n_mels,
                frames=args.frames,
                power=args.power,
            ).astype(np.float32)
            out = sess.run(None, {in_name: inp})
        except Exception as e:
            failed += 1
            print(f"Warning: failed on {fname}: {e}", file=sys.stderr)
            continue

        if not out:
            failed += 1
            print(f"Warning: unexpected encoder output for {fname}", file=sys.stderr)
            continue

        emb = np.asarray(out[0], dtype=np.float32).reshape(-1)
        if emb_dim is None:
            emb_dim = int(emb.size)
            if args.verbose:
                print("Inferred embedding dim from first successful sample:", emb_dim)
        elif emb.size != emb_dim:
            raise RuntimeError(
                f"Embedding dim mismatch for {fname}: got {emb.size}, expected {emb_dim}"
            )

        emb_list.append(emb)
        total_patches += 1

        if args.verbose:
            print(f"[{i}/{len(wavs)}] {fname}: patches=1 total_patches={total_patches}")

    if not emb_list:
        print("ERROR: No embeddings were produced. Check model input shape and calibration audio.", file=sys.stderr)
        sys.exit(4)

    if len(emb_list) < max(1, args.min_samples):
        print(
            f"ERROR: Only {len(emb_list)} calibration samples succeeded; "
            f"minimum required is {args.min_samples}.",
            file=sys.stderr,
        )
        sys.exit(5)

    mean, std, n_samples = compute_stats_from_embeddings(emb_list)

    np.save(args.out_mean, mean)
    np.save(args.out_std, std)
    print(f"Saved {args.out_mean} and {args.out_std}")
    print(f"Samples (patches): {n_samples}, embedding dim: {mean.shape[0]}")
    print(f"Failed files: {failed}")
    print(f"Embedding mean(mean) = {float(mean.mean()):.6f}, embedding mean(std) = {float(std.mean()):.6f}")

if __name__ == '__main__':
    main()
