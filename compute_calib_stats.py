# @path: compute_calib_stats.py

from __future__ import annotations
import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
import onnxruntime as ort

from audio_utils import (
    DEFAULT_FRAMES,
    DEFAULT_HOP,
    DEFAULT_N_FFT,
    DEFAULT_N_MELS,
    DEFAULT_POWER,
    DEFAULT_SR,
    STD_FLOOR,
    load_audio,
    make_mel_patch,
    prepare_input_for_model,
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
    p.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    return p.parse_args()

def list_wavs(calib_dir: str) -> List[str]:
    files = sorted([f for f in os.listdir(calib_dir) if f.lower().endswith('.wav')])
    return files

def prepare_input_for_model(patch: np.ndarray, model_input_shape, frames: int, n_mels: int) -> np.ndarray:
    return prepare_input_for_model(patch, model_input_shape, frames=frames, n_mels=n_mels)

def open_session(model_path: str) -> ort.InferenceSession:
    try:
        return ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    except Exception as e:
        print("Error: Failed to open ONNX model:", e, file=sys.stderr)
        raise

def compute_stats_from_embeddings(emb_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, int]:
    embs = np.stack(emb_list, axis=0)
    mean = embs.mean(axis=0)
    std = embs.std(axis=0)
    std = np.maximum(std, STD_FLOOR)
    return mean.astype(np.float32), std.astype(np.float32), int(embs.shape[0])

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
    if args.verbose:
        print("Model input name:", in_name, "shape:", in_shape)

    emb_list = []
    total_patches = 0
    for i, fname in enumerate(wavs, 1):
        path = os.path.join(args.calib, fname)
        try:
            y = load_audio(path, sr=args.sr)
        except Exception as e:
            print(f"Warning: failed to load {fname}: {e}", file=sys.stderr)
            continue

        patch = make_mel_patch(
            y,
            sr=args.sr,
            n_fft=args.n_fft,
            hop=args.hop,
            n_mels=args.n_mels,
            frames=args.frames,
            power=args.power,
        )
        inp = prepare_input_for_model(patch, in_shape, frames=args.frames, n_mels=args.n_mels).astype(np.float32)
        try:
            out = sess.run(None, {in_name: inp})
        except Exception as e:
            print(f"Warning: model inference failed on {fname}: {e}", file=sys.stderr)
            continue
        if not out or len(out[0].shape) == 0:
            print(f"Warning: unexpected encoder output shape for {fname}", file=sys.stderr)
            continue
        emb = np.asarray(out[0]).reshape(-1)

        if emb.shape[0] != 200:
            raise RuntimeError(f"Unexpected embedding dim {emb.shape[0]}")

        emb_list.append(emb.astype(np.float32))
        total_patches += 1

        if args.verbose:
            print(f"[{i}/{len(wavs)}] {fname}: patches=1 total_patches={total_patches}")

    if not emb_list:
        print("ERROR: No embeddings were produced. Check model input shape and calibration audio.", file=sys.stderr)
        sys.exit(4)

    mean, std, n_samples = compute_stats_from_embeddings(emb_list)

    np.save(args.out_mean, mean)
    np.save(args.out_std, std)
    print(f"Saved {args.out_mean} and {args.out_std}")
    print(f"Samples (patches): {n_samples}, embedding dim: {mean.shape[0]}")
    print(f"Embedding mean(mean) = {float(mean.mean()):.6f}, embedding mean(std) = {float(std.mean()):.6f}")

if __name__ == '__main__':
    main()
