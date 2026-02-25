#!/usr/bin/env python3
import argparse
import os
import sys
import numpy as np
import soundfile as sf
import librosa
import onnxruntime as ort
import json

def load_wav(path, sr=16000):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    y, file_sr = sf.read(path, dtype='float32')
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    return y, sr

def compute_mel(y, sr=16000, n_fft=512, hop_length=256, n_mels=96, power=2.0):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
                                       n_mels=n_mels, power=power)

    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db.astype(np.float32)

def pad_or_trim_time_axis(mel, target_frames=187):
    n_mels, frames = mel.shape
    if frames == target_frames:
        return mel
    if frames < target_frames:
        pad = np.zeros((n_mels, target_frames - frames), dtype=mel.dtype)
        return np.concatenate([mel, pad], axis=1)
    start = max(0, (frames - target_frames)//2)
    return mel[:, start:start+target_frames]

def load_npy_or_bin(path):
    if path.endswith('.npy'):
        return np.load(path).astype(np.float32)
    else:
        data = np.fromfile(path, dtype=np.float32)
        return data.astype(np.float32)

def prepare_input_for_session(session, mel):
    input_meta = session.get_inputs()[0]
    name = input_meta.name
    shape = input_meta.shape
    n_mels, frames = mel.shape

    dims = [d if isinstance(d, int) else -1 for d in shape]
    cand = None
    if len(dims) >= 3:
        if dims[1] in (-1, frames) and dims[2] in (-1, n_mels):
            mel_ready = mel.T[np.newaxis, :, :]
            cand = mel_ready
        elif dims[1] in (-1, n_mels) and dims[2] in (-1, frames):
            mel_ready = mel[np.newaxis, :, :]
            cand = mel_ready
    if cand is None:
        cand = mel.T[np.newaxis, :, :]
    return {name: cand.astype(np.float32)}

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

    y, sr = load_wav(args.test, sr=args.sr)
    mel = compute_mel(y, sr=sr)
    mel = pad_or_trim_time_axis(mel, target_frames=args.frames)

    print("Starting encoder ONNX session...", file=sys.stderr)
    enc_sess = ort.InferenceSession(args.encoder, providers=["CPUExecutionProvider"])
    enc_feed = prepare_input_for_session(enc_sess, mel)
    enc_out = enc_sess.run(None, enc_feed)

    emb = np.array(enc_out[0], dtype=np.float32)
    if emb.ndim > 1 and emb.shape[0] == 1:
        emb = emb.squeeze(0)
    emb = emb.flatten()
    print(f"Encoder produced embedding shape: {emb.shape}", file=sys.stderr)

    mean = load_npy_or_bin(args.mean)
    std = load_npy_or_bin(args.std)

    if mean.size != emb.size:
        if mean.size == 1 and emb.size > 1:
            mean = np.full((emb.size,), float(mean), dtype=np.float32)
        else:
            raise ValueError(f"mean vector length {mean.size} does not match embedding length {emb.size}")
            
    if std.size != emb.size:
        if std.size == 1 and emb.size > 1:
            std = np.full((emb.size,), float(std), dtype=np.float32)
        else:
            raise ValueError(f"std vector length {std.size} does not match embedding length {emb.size}")

    std = np.maximum(std, 1e-6)
    emb_std = (emb - mean) / std
    emb_std = emb_std.astype(np.float32)
    
    print("Standardized embedding (first 8 values):", emb_std[:8], file=sys.stderr)

    print("Starting head ONNX session...", file=sys.stderr)
    head_sess = ort.InferenceSession(args.head, providers=["CPUExecutionProvider"])
    head_input_meta = head_sess.get_inputs()[0]
    head_name = head_input_meta.name

    head_shape = head_input_meta.shape
    if len(head_shape) >= 2:
        head_in = emb_std.reshape((1, -1)).astype(np.float32)
    else:
        head_in = emb_std.astype(np.float32)
        
    feed = {head_name: head_in}
    head_out = head_sess.run(None, feed)

    out_dict = {}
    for i, out in enumerate(head_out):
        arr = np.array(out)
        out_dict[f"output_{i}"] = arr.tolist()
        print(f"Head output {i} shape: {arr.shape}", file=sys.stderr)

    print(json.dumps(out_dict, indent=2))

if __name__ == "__main__":
    main()