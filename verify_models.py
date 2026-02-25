from __future__ import annotations
import argparse
import os
import json
import time
import sys
from typing import Tuple, List, Dict, Any

import numpy as np
import soundfile as sf
import librosa
import onnxruntime as ort

SR = 16000
N_FFT = 512
HOP = 256
N_MELS = 96
TARGET_FRAMES = 187
EPS = 1e-12

def load_wav(path: str, sr: int = SR) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    y, file_sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    return y.astype(np.float32)

def compute_mel(y: np.ndarray) -> np.ndarray:
    S = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, power=2.0
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db.astype(np.float32)

def pad_or_trim_time_axis(mel: np.ndarray, target_frames: int = TARGET_FRAMES) -> np.ndarray:
    n_mels, frames = mel.shape
    if frames == target_frames:
        return mel
    if frames < target_frames:
        pad = np.zeros((n_mels, target_frames - frames), dtype=mel.dtype)
        return np.concatenate([mel, pad], axis=1)
    start = max(0, (frames - target_frames) // 2)
    return mel[:, start : start + target_frames]

def prepare_input_for_session(session: ort.InferenceSession, mel: np.ndarray) -> Dict[str, np.ndarray]:
    input_meta = session.get_inputs()[0]
    name = input_meta.name
    shape = input_meta.shape
    dims = [d if isinstance(d, int) else -1 for d in shape]

    n_mels, frames = mel.shape

    cand = None
    if len(dims) >= 3:
        if (dims[1] in (-1, frames)) and (dims[2] in (-1, n_mels)):
            cand = mel.T[np.newaxis, :, :].astype(np.float32)
        elif (dims[1] in (-1, n_mels)) and (dims[2] in (-1, frames)):
            cand = mel[np.newaxis, :, :].astype(np.float32)

    if cand is None:
        cand = mel.T[np.newaxis, :, :].astype(np.float32)

    return {name: cand}

def run_session(session: ort.InferenceSession, mel: np.ndarray) -> List[np.ndarray]:
    feed = prepare_input_for_session(session, mel)
    outs = session.run(None, feed)
    np_outs = [np.array(o).astype(np.float32) for o in outs]
    return np_outs

def pearsonr_safe(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size == 0 or b.size == 0:
        return float("nan")
    if a.shape != b.shape:
        minlen = min(a.size, b.size)
        a = a.ravel()[:minlen]
        b = b.ravel()[:minlen]
    a_mean = a.mean()
    b_mean = b.mean()
    a_dev = a - a_mean
    b_dev = b - b_mean
    denom = np.sqrt((a_dev ** 2).sum() * (b_dev ** 2).sum())
    if denom < 1e-15:
        if np.allclose(a, b, atol=1e-9):
            return 1.0
        return 0.0
    return float((a_dev * b_dev).sum() / denom)

def mse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        minlen = min(a.size, b.size)
        a = a.ravel()[:minlen]
        b = b.ravel()[:minlen]
    return float(np.mean((a - b) ** 2))

def prepare_mel_input(path: str) -> np.ndarray:
    y = load_wav(path)
    mel = compute_mel(y)
    mel = pad_or_trim_time_axis(mel, TARGET_FRAMES)
    return mel

def collect_wav_files(test_dir: str, max_files: int | None = None) -> List[str]:
    wavs = sorted(
        [
            os.path.join(test_dir, f)
            for f in os.listdir(test_dir)
            if f.lower().endswith(".wav")
        ]
    )
    if max_files:
        wavs = wavs[:max_files]
    return wavs

def main():
    p = argparse.ArgumentParser(description="Verify FP32 vs QDQ ONNX parity on test WAVs")
    p.add_argument("--fp32", required=True, help="Path to FP32 merged ONNX (reference)")
    p.add_argument("--int8", required=True, help="Path to quantized QDQ ONNX (candidate)")
    p.add_argument("--test_dir", required=True, help="Directory of WAV files to test")
    p.add_argument("--out", default="verify_report.json", help="Output JSON report")
    p.add_argument("--max_files", type=int, default=200, help="Max number of files to test")
    p.add_argument("--warmup", type=int, default=1, help="Warmup runs per session")
    args = p.parse_args()

    if not os.path.exists(args.fp32):
        print("ERROR: fp32 model not found:", args.fp32, file=sys.stderr)
        sys.exit(2)
    if not os.path.exists(args.int8):
        print("ERROR: int8 model not found:", args.int8, file=sys.stderr)
        sys.exit(2)
    if not os.path.isdir(args.test_dir):
        print("ERROR: test_dir missing or not a directory:", args.test_dir, file=sys.stderr)
        sys.exit(2)

    print("Creating ONNX Runtime sessions...", file=sys.stderr)
    sess_fp32 = ort.InferenceSession(args.fp32, providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(args.int8, providers=["CPUExecutionProvider"])

    def warmup(sess):
        meta = sess.get_inputs()[0]
        zero = np.zeros((1, TARGET_FRAMES, N_MELS), dtype=np.float32)
        try:
            sess.run(None, {meta.name: zero})
        except Exception:
            pass

    for _ in range(args.warmup):
        warmup(sess_fp32)
        warmup(sess_int8)

    wav_files = collect_wav_files(args.test_dir, max_files=args.max_files)
    if not wav_files:
        print("No wav files found in test_dir", file=sys.stderr)
        sys.exit(2)

    results: Dict[str, Any] = {
        "meta": {
            "fp32_model": args.fp32,
            "int8_model": args.int8,
            "test_files_count": len(wav_files),
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "per_file": {},
        "summary": {},
    }

    fp32_outputs_list = []
    int8_outputs_list = []
    absdiffs_max_per_file = {}
    file_times: Dict[str, float] = {}

    print(f"Running inference on {len(wav_files)} files...", file=sys.stderr)
    for wav in wav_files:
        try:
            t0 = time.time()
            mel = prepare_mel_input(wav)
            out_fp32_list = run_session(sess_fp32, mel)
            out_int8_list = run_session(sess_int8, mel)
            t1 = time.time()
        except Exception as e:
            results["per_file"][wav] = {"error": repr(e)}
            print(f"ERROR running {wav}: {e}", file=sys.stderr)
            continue

        def norm_out(o_list):
            normed = []
            for o in o_list:
                a = np.array(o).astype(np.float32)
                if a.ndim >= 2 and a.shape[0] == 1:
                    a = a.squeeze(0)
                normed.append(a)
            return normed

        fp32_norm = norm_out(out_fp32_list)
        int8_norm = norm_out(out_int8_list)

        primary_fp32 = fp32_norm[0].ravel()
        primary_int8 = int8_norm[0].ravel()

        minlen_f = min(primary_fp32.size, primary_int8.size)
        primary_fp32_a = primary_fp32[:minlen_f]
        primary_int8_a = primary_int8[:minlen_f]

        absdiff = np.abs(primary_fp32_a - primary_int8_a)
        max_abs = float(absdiff.max() if absdiff.size else 0.0)
        mean_abs = float(absdiff.mean() if absdiff.size else 0.0)
        mse_v = mse(primary_fp32_a, primary_int8_a)
        pear = pearsonr_safe(primary_fp32_a, primary_int8_a)

        fp32_outputs_list.append(primary_fp32_a)
        int8_outputs_list.append(primary_int8_a)

        file_times[wav] = float(t1 - t0)
        absdiffs_max_per_file[wav] = {"max_abs": max_abs, "mean_abs": mean_abs, "mse": mse_v, "pearson": pear}

        results["per_file"][wav] = {
            "max_abs_diff": max_abs,
            "mean_abs_diff": mean_abs,
            "mse": mse_v,
            "pearson": pear,
            "runtime_s": file_times[wav],
        }

        print(f"{os.path.basename(wav)}: pear={pear:.6f} max_abs={max_abs:.6e} mse={mse_v:.6e} t={file_times[wav]:.3f}s", file=sys.stderr)

    if fp32_outputs_list and int8_outputs_list:
        # Ensure consistent lengths across all outputs
        minlen = min(
            min(a.size for a in fp32_outputs_list),
            min(b.size for b in int8_outputs_list),
        )

        A = np.stack([a.ravel()[:minlen] for a in fp32_outputs_list], axis=0)
        B = np.stack([a.ravel()[:minlen] for a in int8_outputs_list], axis=0)

        per_dim_pearsons = []
        per_dim_mse = []
        for dim in range(A.shape[1]):
            a_col = A[:, dim]
            b_col = B[:, dim]
            per_dim_pearsons.append(pearsonr_safe(a_col, b_col))
            per_dim_mse.append(float(np.mean((a_col - b_col) ** 2)))

        flat_fp32 = A.ravel()
        flat_int8 = B.ravel()
        overall_pearson = pearsonr_safe(flat_fp32, flat_int8)
        overall_mse = mse(flat_fp32, flat_int8)
        overall_max_abs = float(np.max(np.abs(flat_fp32 - flat_int8)))
        overall_mean_abs = float(np.mean(np.abs(flat_fp32 - flat_int8)))

        # Diagnostic: detect systematic bias from quantization
        mean_fp32 = float(flat_fp32.mean())
        mean_int8 = float(flat_int8.mean())
        mean_drift = float(mean_fp32 - mean_int8)

        results["summary"] = {
            "per_dim_pearson": per_dim_pearsons,
            "per_dim_mse": per_dim_mse,
            "overall_pearson": overall_pearson,
            "overall_mse": overall_mse,
            "overall_max_abs": overall_max_abs,
            "overall_mean_abs": overall_mean_abs,
            "mean_fp32": mean_fp32,
            "mean_int8": mean_int8,
            "mean_drift": mean_drift,
            "per_file_max_abs": absdiffs_max_per_file,
            "runtime_seconds": file_times,
        }
    else:
        results["summary"] = {"error": "No successful outputs to compare."}

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Verification complete. Report written to:", args.out, file=sys.stderr)
    print(json.dumps(results.get("summary", {}), indent=2))

if __name__ == "__main__":
    main()