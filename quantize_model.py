# @path: quantize_model.py
import sys
import os
import numpy as np
import onnx
import onnxruntime as ort
import librosa
from onnxruntime.quantization import quantize_dynamic, QuantType
SR = 16000
N_FFT = 512
HOP = 256
N_MELS = 96
FRAMES = 187
def quantize_model(fp32_path, qdq_path):
    quantize_dynamic(
        fp32_path,
        qdq_path,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
        reduce_range=True,
        extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT},
    )
    print("Saved:", qdq_path)
def main():
    if len(sys.argv) < 3:
        print("Usage: python script.py model_fp32.onnx model_qdq.onnx")
        sys.exit(1)
    fp32_path = sys.argv[1]
    qdq_path  = sys.argv[2]
    global mean, std
    if os.path.exists("emb_mean.npy") and os.path.exists("emb_std.npy"):
        mean = np.load("emb_mean.npy").astype(np.float32)
        std  = np.load("emb_std.npy").astype(np.float32)
    
    quantize_model(fp32_path, qdq_path)
    wav_dir = "test_wavs"
    if not os.path.isdir(wav_dir):
        print(f"[WARN] Missing '{wav_dir}/' → skipping validation")
        return
    files = list_wavs(wav_dir)
    if not files:
        print(f"[WARN] No WAV files in '{wav_dir}/' → skipping validation")
        return
    fp32_sess = load_session(fp32_path)
    qdq_sess  = load_session(qdq_path)
    results_fp32 = []
    results_qdq  = []
    for f in files:
        x = preprocess_audio(f)
        y_fp32 = run(fp32_sess, x)
        y_qdq  = run(qdq_sess, x)
        results_fp32.append(y_fp32)
        results_qdq.append(y_qdq)
    metrics = compute_metrics(results_fp32, results_qdq)
    print_report(metrics)
def list_wavs(dir_path):
    return [
        os.path.join(dir_path, f)
        for f in sorted(os.listdir(dir_path))
        if f.lower().endswith(".wav")
    ]
def load_session(path):
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])
def preprocess_audio(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP,
        n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_t = mel_db.T.astype(np.float32)
    if mel_t.shape[0] >= FRAMES:
        mel_fixed = mel_t[:FRAMES]
    else:
        pad = np.full(
            (FRAMES - mel_t.shape[0], N_MELS),
            mel_t.min() if mel_t.size else -80.0,
            dtype=np.float32
        )
        mel_fixed = np.vstack((mel_t, pad))
    return mel_fixed[None, :, :]
def run(session, x):
    inp = session.get_inputs()[0]
    name = inp.name
    if len(inp.shape) >= 3 and inp.shape[2] == x.shape[1]:
        x = x.transpose(0, 2, 1)
    out = session.run(None, {name: x.astype(np.float32)})[0]
    if "mean" in globals() and "std" in globals():
        if out.shape[-1] == mean.shape[0]:
            out = (out - mean) / np.maximum(std, 1e-8)
    return out
def compute_metrics(A, B):
    A = np.vstack(A)
    B = np.vstack(B)
    diff = np.abs(A - B)
    max_diff  = diff.max()
    mean_diff = diff.mean()
    corr_0 = pearson(A[:, 0], B[:, 0])
    corr_1 = pearson(A[:, 1], B[:, 1])
    return {
        "max_diff": float(max_diff),
        "mean_diff": float(mean_diff),
        "pearson_valence": float(corr_0),
        "pearson_arousal": float(corr_1),
        "n_samples": len(A),
    }
def pearson(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) < 2:
        return float("nan")
    xm = x - x.mean()
    ym = y - y.mean()
    denom = np.sqrt((xm**2).sum() * (ym**2).sum())
    if denom < 1e-12:
        return 0.0
    return float((xm * ym).sum() / denom)
def print_report(m):
    print("Max diff:", m["max_diff"])
    print("Mean diff:", m["mean_diff"])
    corr_v = m["pearson_valence"]
    corr_a = m["pearson_arousal"]
    n      = m.get("n_samples", "?")
    print("Corr valence:", corr_v)
    print("Corr arousal:", corr_a)
    if np.isnan(corr_v) or np.isnan(corr_a):
        print(f"[WARN] Pearson undefined (n_samples={n}); using diff-based check.")
        if m["max_diff"] <= 0.01 and m["mean_diff"] <= 0.001:
            print("PASS: quantization acceptable (diff-based)")
        else:
            print("FAIL: quantization degraded model (diff-based)")
    elif corr_v < 0.92 or corr_a < 0.92:
        print("FAIL: quantization degraded model")
    else:
        print("PASS: quantization acceptable")
if __name__ == "__main__":
    main()
