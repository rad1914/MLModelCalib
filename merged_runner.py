# @path: merged_runner.py
import argparse
from typing import Optional

import numpy as np
import sys

from audio_utils import (
    DEFAULT_FRAMES,
    DEFAULT_HOP,
    DEFAULT_N_FFT,
    DEFAULT_N_MELS,
    DEFAULT_POWER,
    DEFAULT_SR,
    get_cpu_session,
    load_mel_patch,
    prepare_vector_for_model,
    standardize_embedding,
)

def make_mel(path):
    return load_mel_patch(
        path,
        sr=DEFAULT_SR,
        n_fft=DEFAULT_N_FFT,
        hop=DEFAULT_HOP,
        n_mels=DEFAULT_N_MELS,
        frames=DEFAULT_FRAMES,
        power=DEFAULT_POWER,
    )

def run_onnx(model_path: str, input_arr: np.ndarray) -> np.ndarray:
    sess = get_cpu_session(model_path)
    inp_name = sess.get_inputs()[0].name
    out = sess.run(None, {inp_name: input_arr.astype(np.float32)})
    return np.asarray(out[0], dtype=np.float32).squeeze()

def run_encoder_get_emb(encoder_path: str, mel: np.ndarray) -> np.ndarray:
    return run_onnx(encoder_path, mel[np.newaxis]).squeeze()

def run_head_on_std_emb(head_path: str, std_emb: np.ndarray) -> np.ndarray:
    return run_onnx(head_path, std_emb[np.newaxis]).squeeze()

def find_best_slice(big: np.ndarray, small: np.ndarray):
    big1 = np.asarray(big, dtype=np.float32).ravel()
    small1 = np.asarray(small, dtype=np.float32).ravel()
    n, m = big1.size, small1.size
    if m > n:
        return None
    windows = np.lib.stride_tricks.sliding_window_view(big1, m)
    dists = np.linalg.norm(windows - small1, axis=1)
    best_i = int(np.argmin(dists))
    return best_i, float(dists[best_i])

def safe_load_np(path: Optional[str]) -> Optional[np.ndarray]:
    if path is None:
        return None
    arr = np.load(path, allow_pickle=False)
    arr = np.asarray(arr, dtype=np.float32)
    if not np.isfinite(arr).all():
        raise ValueError(f"Non-finite values found in {path}")
    return arr

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
    p = argparse.ArgumentParser()
    p.add_argument("--merged", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--mean", required=False)
    p.add_argument("--std", required=False)
    p.add_argument("--head", required=False, help="Path to head ONNX (deam_head.onnx)")
    p.add_argument("--encoder", required=False, help="Optional encoder ONNX (msd_musicnn.onnx) for parity/search")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    mel = make_mel(args.test)
    if args.verbose:
        print("mel.shape:", mel.shape)

    merged_sess = get_cpu_session(args.merged)
    merged_input = merged_sess.get_inputs()[0]
    merged_hint = _output_dim_hint(merged_sess.get_outputs()[0].shape)

    print("Running merged model:", args.merged)
    merged_out = np.asarray(
        merged_sess.run(None, {merged_input.name: mel[np.newaxis].astype(np.float32)})[0],
        dtype=np.float32,
    ).squeeze()
    print("merged_out shape:", merged_out.shape, "size:", merged_out.size)

    if args.verbose:
        print("merged_out dtype:", merged_out.dtype)
        print("merged_out first_10:", merged_out.ravel()[:10].tolist())

    mean = safe_load_np(args.mean) if args.mean else None
    std = safe_load_np(args.std) if args.std else None

    if mean is not None:
        mean = np.asarray(mean, dtype=np.float32).reshape(-1)
    if std is not None:
        std = np.asarray(std, dtype=np.float32).reshape(-1)

    if mean is not None and merged_out.size == mean.size:
        if args.verbose:
            print("Merged output length equals mean length -> interpreting as embedding")
            print("mean.shape:", mean.shape, "std.shape:", None if std is None else std.shape)
        if std is None:
            print("ERROR: std missing for embedding interpretation", file=sys.stderr)
            sys.exit(2)
        if std.size != mean.size:
            print("ERROR: std size mismatch", file=sys.stderr)
            sys.exit(3)

        emb = merged_out.astype(np.float32).reshape(mean.shape)
        std_emb = standardize_embedding(emb, mean, std)
        if args.head is None:
            print("HEAD model is required to process standardized embedding. Provide --head path.")
            sys.exit(2)
        head_out = run_head_on_std_emb(args.head, std_emb)
        print("Valence/Arousal from head (merged->std_emb->head):", head_out.tolist())
        return

    if merged_hint == 2 and merged_out.size == 2:
        va = merged_out.ravel()
        print("Merged model produced valence/arousal:", va.tolist())
        return

    if args.head and args.encoder:
        print("Merged output is not a 2-value VA tensor and does not match the embedding stats size. Running encoder->head parity search.")
        enc_emb = np.asarray(run_encoder_get_emb(args.encoder, mel), dtype=np.float32).squeeze()
        if mean is not None:
            if std is None:
                print("ERROR: std missing for embedding standardization", file=sys.stderr)
                sys.exit(2)
            if mean.size != enc_emb.size or std.size != enc_emb.size:
                print("ERROR: calibration vectors do not match encoder embedding size", file=sys.stderr)
                sys.exit(3)
            std_emb = standardize_embedding(enc_emb, mean, std)
        else:
            std_emb = enc_emb
        head_out = run_head_on_std_emb(args.head, std_emb)
        print("Computed head_out (encoder->std->head):", head_out.tolist())

        res = find_best_slice(merged_out, head_out)
        if res is None:
            print("Head output longer than merged output or no match possible.")
        else:
            offset, l2 = res
            print(f"Best-match slice at offset {offset} with L2={l2:.6e}")
            candidate = merged_out.ravel()[offset:offset + head_out.size]
            print("Candidate slice:", candidate.tolist())
            print("Head_out:", head_out.tolist())
        return

    print("Merged model produced an output of size", merged_out.size)
    print("No mean/std or head/encoder provided to interpret it further.")
    if args.verbose:
        print("Merged raw output (first 200 elems):", merged_out.ravel()[:200].tolist())

if __name__ == "__main__":
    main()
