# @path: merged_runner.py
import argparse
import numpy as np
import onnxruntime as ort
import librosa
import sys
from typing import Optional

SR = 16000
N_FFT = 512
HOP = 256
N_MELS = 96
FRAMES = 187

def make_mel(path: str) -> np.ndarray:
    y, _ = librosa.load(path, sr=SR, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, power=2.0)
    mel_db = librosa.power_to_db(mel, ref=np.max).T.astype(np.float32)
    if mel_db.shape[0] >= FRAMES:
        return mel_db[:FRAMES]
    pad = FRAMES - mel_db.shape[0]
    pad_val = mel_db.min() if mel_db.shape[0] > 0 else -80.0
    return np.vstack([mel_db, np.full((pad, N_MELS), pad_val, dtype=np.float32)])

def run_onnx(model_path: str, input_arr: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    out = sess.run(None, {inp_name: input_arr.astype(np.float32)})

    out0 = np.array(out[0])
    return out0.squeeze()

def run_encoder_get_emb(encoder_path: str, mel: np.ndarray) -> np.ndarray:
    emb = run_onnx(encoder_path, mel[np.newaxis])
    return emb.squeeze()

def run_head_on_std_emb(head_path: str, std_emb: np.ndarray) -> np.ndarray:
    out = run_onnx(head_path, std_emb[np.newaxis])
    return out.squeeze()

def find_best_slice(big: np.ndarray, small: np.ndarray):
    big1 = big.ravel()
    small1 = small.ravel()
    n, m = big1.size, small1.size
    if m > n:
        return None
    best_i = None
    best_l2 = float("inf")
    for i in range(0, n - m + 1):
        s = big1[i:i+m]
        d = np.linalg.norm(s - small1)
        if d < best_l2:
            best_l2 = d
            best_i = i
    return best_i, best_l2

def safe_load_np(path: Optional[str]) -> Optional[np.ndarray]:
    if path is None:
        return None
    return np.load(path)

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

    print("Running merged model:", args.merged)
    merged_out = run_onnx(args.merged, mel[np.newaxis])
    merged_out = np.array(merged_out).squeeze()
    print("merged_out shape:", merged_out.shape, "size:", merged_out.size)

    if args.verbose:
        if merged_out.size not in (2, 200):
            print("WARNING: unexpected output size:", merged_out.size)

        print("merged_out dtype:", merged_out.dtype)
        print("merged_out first_10:", merged_out.ravel()[:10].tolist())

    if merged_out.size == 2:
        va = merged_out.ravel()
        print("Merged model produced valence/arousal:", va.tolist())
        return

    mean = safe_load_np(args.mean) if args.mean else None
    std = safe_load_np(args.std) if args.std else None

    if mean is not None and merged_out.size == mean.size:
        if args.verbose:
            print("Merged output length equals mean length -> interpreting as embedding")
            print("mean.shape:", mean.shape, "std.shape:", None if std is None else std.shape)
        if std is None:
            print("ERROR: std missing for embedding interpretation", file=sys.stderr)
            sys.exit(2)

        if mean.shape[0] != 200:
            print("ERROR: unexpected embedding size", mean.shape, file=sys.stderr)
            sys.exit(3)

        emb = merged_out.astype(np.float32).reshape(mean.shape)
        std_safe = np.where(std == 0.0, 1.0, std) if std is not None else np.ones_like(mean)
        std_emb = (emb - mean) / std_safe
        if args.head is None:
            print("HEAD model is required to process standardized embedding. Provide --head path.")
            sys.exit(2)
        head_out = run_head_on_std_emb(args.head, std_emb)
        print("Valence/Arousal from head (merged->std_emb->head):", head_out.tolist())
        return

    if args.head and args.encoder:
        print("Merged output != 2 and != emb_dim. Will compute encoder->head and search for a matching slice inside merged output.")
        enc_emb = run_encoder_get_emb(args.encoder, mel)
        enc_emb = np.array(enc_emb).squeeze()
        if mean is not None:
            mean = mean.reshape(enc_emb.shape)
            std = std.reshape(enc_emb.shape)
            std_safe = np.where(std == 0.0, 1.0, std)
            std_emb = (enc_emb - mean) / std_safe
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
            candidate = merged_out.ravel()[offset:offset+head_out.size]
            print("Candidate slice:", candidate.tolist())
            print("Head_out:", head_out.tolist())
        return

    print("Merged model produced an output of size", merged_out.size)
    print("No mean/std or head/encoder provided to interpret it further.")
    if args.verbose:
        print("Merged raw output (first 200 elems):", merged_out.ravel()[:200].tolist())

if __name__ == "__main__":
    main()
