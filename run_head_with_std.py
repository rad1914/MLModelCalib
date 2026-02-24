#!/usr/bin/env python3
"""
run_head_with_std.py

Load an encoder ONNX and a head ONNX, compute mel from a test WAV, run encoder -> standardize (mean/std) -> head,
and print the head outputs.

Designed to be robust to common ONNX input layout differences (time-major vs mel-major).
"""
import argparse
import os
import sys
import numpy as np
import soundfile as sf
import librosa
import onnxruntime as ort
import json

# ---- utils ----
def load_wav(path, sr=16000):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    y, file_sr = sf.read(path, dtype='float32')
    # soundfile can return multi-channel; convert to mono
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    return y, sr

def compute_mel(y, sr=16000, n_fft=512, hop_length=256, n_mels=96, power=2.0):
    # compute power mel spectrogram, then convert to dB (log) like many pipelines
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
                                       n_mels=n_mels, power=power)
    # convert to log-scale (dB). This mirrors typical power_to_db used in training.
    S_db = librosa.power_to_db(S, ref=1.0)
    # S_db shape: (n_mels, frames)
    return S_db.astype(np.float32)

def pad_or_trim_time_axis(mel, target_frames=187):
    # mel shape currently (n_mels, frames)
    n_mels, frames = mel.shape
    if frames == target_frames:
        return mel
    if frames < target_frames:
        pad = np.zeros((n_mels, target_frames - frames), dtype=mel.dtype)
        return np.concatenate([mel, pad], axis=1)
    # frames > target -> center-crop
    start = max(0, (frames - target_frames)//2)
    return mel[:, start:start+target_frames]

def load_npy_or_bin(path):
    if path.endswith('.npy'):
        return np.load(path).astype(np.float32)
    else:
        # assume raw float32 binary
        data = np.fromfile(path, dtype=np.float32)
        return data.astype(np.float32)

def prepare_input_for_session(session, mel):
    """
    session: an onnxruntime.InferenceSession
    mel: numpy array with shape (n_mels, frames)
    We'll inspect session input shape and transpose/reshape as required.
    Returns: (feed_dict, input_name)
    """
    input_meta = session.get_inputs()[0]
    name = input_meta.name
    shape = input_meta.shape  # may contain None
    # common possibilities:
    # [1, 187, 96] => batch, time, n_mels  -> we must provide shape (1,187,96)
    # [1, 96, 187] => batch, n_mels, time  -> (1,96,187)
    # [1, 96, 187, 1] etc. handle basic cases
    n_mels, frames = mel.shape

    # detect expected order by inspecting dims (ignoring None)
    dims = [d if isinstance(d, int) else -1 for d in shape]
    # check for a dim equal to n_mels and/or target frames
    # create candidates
    cand = None
    if len(dims) >= 3:
        # try batch, time, n_mels
        if dims[1] in (-1, frames) and dims[2] in (-1, n_mels):
            # shape is (1, time, n_mels)
            mel_ready = mel.T[np.newaxis, :, :]  # (1, time, n_mels)
            cand = mel_ready
        elif dims[1] in (-1, n_mels) and dims[2] in (-1, frames):
            # shape is (1, n_mels, time)
            mel_ready = mel[np.newaxis, :, :]  # (1, n_mels, time)
            cand = mel_ready
    if cand is None:
        # fallback: create (1, frames, n_mels)
        cand = mel.T[np.newaxis, :, :]
    return {name: cand.astype(np.float32)}

# ---- main ----
def main():
    p = argparse.ArgumentParser(description="Run encoder ONNX -> standardize -> head ONNX on a test wav")
    p.add_argument("--encoder", required=True, help="Path to encoder ONNX (e.g. msd_musicnn.onnx)")
    p.add_argument("--head", required=True, help="Path to head ONNX (e.g. deam_head.onnx)")
    p.add_argument("--mean", required=True, help="Path to emb_mean.npy (or .bin)")
    p.add_argument("--std", required=True, help="Path to emb_std.npy (or .bin)")
    p.add_argument("--test", required=True, help="Test WAV file")
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--frames", type=int, default=187, help="Target number of frames (time dimension)")
    args = p.parse_args()

    for pth in (args.encoder, args.head, args.mean, args.std, args.test):
        if not os.path.exists(pth):
            print(f"ERROR: missing file: {pth}", file=sys.stderr)
            sys.exit(2)

    # 1) load wav + mel
    y, sr = load_wav(args.test, sr=args.sr)
    mel = compute_mel(y, sr=sr)
    mel = pad_or_trim_time_axis(mel, target_frames=args.frames)  # (n_mels, frames)
    # keep a copy for debugging
    # 2) run encoder ONNX
    print("Starting encoder ONNX session...", file=sys.stderr)
    enc_sess = ort.InferenceSession(args.encoder, providers=["CPUExecutionProvider"])
    enc_feed = prepare_input_for_session(enc_sess, mel)
    enc_out = enc_sess.run(None, enc_feed)
    # pick first output as embedding
    emb = np.array(enc_out[0], dtype=np.float32)
    # squeeze to 1d or (1,dim)
    if emb.ndim > 1 and emb.shape[0] == 1:
        emb = emb.squeeze(0)
    emb = emb.flatten()
    print(f"Encoder produced embedding shape: {emb.shape}", file=sys.stderr)

    # 3) load mean/std and standardize
    mean = load_npy_or_bin(args.mean)
    std = load_npy_or_bin(args.std)
    # try to broadcast: if mean is 1D, match emb shape
    if mean.size != emb.size:
        # allow shape (dim,) vs (1,dim)
        if mean.size == 1 and emb.size > 1:
            mean = np.full((emb.size,), float(mean), dtype=np.float32)
        else:
            raise ValueError(f"mean vector length {mean.size} does not match embedding length {emb.size}")
    if std.size != emb.size:
        if std.size == 1 and emb.size > 1:
            std = np.full((emb.size,), float(std), dtype=np.float32)
        else:
            raise ValueError(f"std vector length {std.size} does not match embedding length {emb.size}")
    emb_std = (emb - mean) / (std + 1e-12)
    emb_std = emb_std.astype(np.float32)
    print("Standardized embedding (first 8 values):", emb_std[:8], file=sys.stderr)

    # 4) run head ONNX
    print("Starting head ONNX session...", file=sys.stderr)
    head_sess = ort.InferenceSession(args.head, providers=["CPUExecutionProvider"])
    head_input_meta = head_sess.get_inputs()[0]
    head_name = head_input_meta.name
    # prepare head input to match expected shape
    # many heads expect (1, dim)
    head_shape = head_input_meta.shape
    if len(head_shape) >= 2:
        # create (1, dim)
        head_in = emb_std.reshape((1, -1)).astype(np.float32)
    else:
        head_in = emb_std.astype(np.float32)
    feed = {head_name: head_in}
    head_out = head_sess.run(None, feed)
    # print outputs
    out_dict = {}
    for i, out in enumerate(head_out):
        arr = np.array(out)
        out_dict[f"output_{i}"] = arr.tolist()
        print(f"Head output {i} shape: {arr.shape}", file=sys.stderr)

    # pretty print JSON to stdout
    print(json.dumps(out_dict, indent=2))

if __name__ == "__main__":
    main()