#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
import argparse
import numpy as np
import librosa
import onnxruntime as ort
from typing import Tuple, List

DEFAULT_SR = 16000
DEFAULT_N_FFT = 512
DEFAULT_HOP = 256
DEFAULT_N_MELS = 96
DEFAULT_FRAMES = 187
DEFAULT_WIN_SEC = 3.0
DEFAULT_HOP_SEC = 1.0
EPS_STD_FLOOR = 1e-6
REPLACE_SMALL_STD_WITH = 1.0

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
    p.add_argument('--win-sec', type=float, default=DEFAULT_WIN_SEC, help='Window length in seconds (patch size)')
    p.add_argument('--hop-sec', type=float, default=DEFAULT_HOP_SEC, help='Hop/stride between patches in seconds')
    p.add_argument('--power', type=float, default=2.0, help='Power for mel spectrogram (2.0 for power)')
    p.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    return p.parse_args()

def list_wavs(calib_dir: str) -> List[str]:
    files = sorted([f for f in os.listdir(calib_dir) if f.lower().endswith('.wav')])
    return files

def load_audio(path: str, sr: int) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y

def make_mel_patch(y: np.ndarray, sr: int, n_fft: int, hop: int, n_mels: int, frames: int, power: float) -> np.ndarray:
    y = y.astype(np.float32)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop,
        n_mels=n_mels, power=power
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = np.clip(mel_db, -80.0, 0.0)

    frames_raw = mel_db.shape[1]
    if frames_raw % 2 != 0:
        mel_db = mel_db[:, :frames_raw - 1]

    mel_ds = 0.5 * (mel_db[:, 0::2] + mel_db[:, 1::2])
    mel_db = mel_ds.T

    if mel_db.shape[0] >= frames:
        start = (mel_db.shape[0] - frames) // 2
        patch = mel_db[start:start+frames].astype(np.float32)
    else:
        pad_val = float(mel_db.min()) if mel_db.size else -80.0
        pad = np.full((frames - mel_db.shape[0], n_mels), pad_val, dtype=np.float32)
        patch = np.vstack([mel_db.astype(np.float32), pad])
    return patch

def prepare_input_for_model(patch: np.ndarray, model_input_shape, frames: int, n_mels: int) -> np.ndarray:
    arr = patch[np.newaxis, :, :].astype(np.float32)

    if not model_input_shape:
        return arr

    shape = list(model_input_shape)
    if len(shape) == 3:
        s1 = shape[1]
        s2 = shape[2]
        if (s1 is None or int(s1) == frames) and (s2 is None or int(s2) == n_mels):
            return arr
        if (s1 is None or int(s1) == n_mels) and (s2 is None or int(s2) == frames):
            return arr.transpose(0,2,1)

    if len(shape) == 4:
        if (shape[1] is None or int(shape[1]) == frames) and (shape[2] is None or int(shape[2]) == n_mels):
            return arr[:, :, :, np.newaxis]
        if (shape[2] is None or int(shape[2]) == frames) and (shape[3] is None or int(shape[3]) == n_mels):
            return arr[:, np.newaxis, :, :]
        if (shape[1] is None or int(shape[1]) == n_mels) and (shape[2] is None or int(shape[2]) == frames):
            return arr.transpose(0,2,1)[:, :, :, np.newaxis]

    return arr

def open_session(model_path: str) -> ort.InferenceSession:
    try:
        sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        return sess
    except Exception as e:
        print("Error: Failed to open ONNX model:", e, file=sys.stderr)
        raise

def compute_stats_from_embeddings(emb_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    embs = np.stack(emb_list, axis=0)
    mean = embs.mean(axis=0)
    std = embs.std(axis=0)
    small_mask = std < EPS_STD_FLOOR
    if small_mask.any():
        std[small_mask] = REPLACE_SMALL_STD_WITH
    return mean.astype(np.float32), std.astype(np.float32), embs.shape[0]

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
    win_samples = int(args.win_sec * args.sr)
    hop_samples = int(args.hop_sec * args.sr)
    for i, fname in enumerate(wavs, 1):
        path = os.path.join(args.calib, fname)
        try:
            y = load_audio(path, sr=args.sr)
        except Exception as e:
            print(f"Warning: failed to load {fname}: {e}", file=sys.stderr)
            continue

        if len(y) < win_samples:
            starts = [0]
        else:
            starts = list(range(0, max(1, len(y) - win_samples + 1), hop_samples))

        for s in starts:
            chunk = y[s:s+win_samples]
            patch = make_mel_patch(chunk, sr=args.sr, n_fft=args.n_fft, hop=args.hop,
                                   n_mels=args.n_mels, frames=args.frames, power=args.power)
            inp = prepare_input_for_model(patch, in_shape, frames=args.frames, n_mels=args.n_mels)
            inp = inp.astype(np.float32)
            try:
                out = sess.run(None, {in_name: inp})
            except Exception as e:
                print(f"Warning: model inference failed on {fname} (start={s}): {e}", file=sys.stderr)
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
            print(f"[{i}/{len(wavs)}] {fname}: patches={len(starts)} total_patches={total_patches}")

    if not emb_list:
        print("ERROR: No embeddings were produced. Check model input shape and calibration audio.", file=sys.stderr)
        sys.exit(4)

    np.random.shuffle(emb_list)

    mean, std, n_samples = compute_stats_from_embeddings(emb_list)

    np.save(args.out_mean, mean)
    np.save(args.out_std, std)
    print(f"Saved {args.out_mean} and {args.out_std}")
    print(f"Samples (patches): {n_samples}, embedding dim: {mean.shape[0]}")
    print(f"Embedding mean(mean) = {float(mean.mean()):.6f}, embedding mean(std) = {float(std.mean()):.6f}")

if __name__ == '__main__':
    main()