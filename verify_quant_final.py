# @path: verify_quant_final.py
import sys, os, numpy as np, librosa, onnxruntime as ort, math
from scipy.stats import pearsonr
SR,N_FFT,HOP,N_MELS,FRAMES = 16000,512,256,96,187
def mel(f):
    y,_ = librosa.load(f, sr=SR, mono=True)
    m = librosa.power_to_db(librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS
    ), ref=np.max).T.astype(np.float32)
    return m[:FRAMES] if len(m)>=FRAMES else np.pad(
        m, ((0,FRAMES-len(m)),(0,0)), constant_values=m.min()
    )
def run(sess, x):
    return sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
def main():
    if len(sys.argv) != 4:
        print("Usage: verify_quant_final.py fp32.onnx qdq.onnx wav_dir")
        sys.exit(1)
    fp32, qdq, wav_dir = sys.argv[1:]
    s_fp32 = ort.InferenceSession(fp32)
    s_qdq  = ort.InferenceSession(qdq)
    files = [os.path.join(wav_dir,f) for f in os.listdir(wav_dir) if f.endswith(".wav")]
    if len(files) < 10:
        raise RuntimeError("Need at least 10 samples (not 1).")
    fp32_out, qdq_out = [], []
    for f in files:
        x = mel(f)[None]
        fp32_out.append(run(s_fp32, x))
        qdq_out.append(run(s_qdq, x))
    fp32_out = np.array(fp32_out)
    qdq_out  = np.array(qdq_out)
    failed = False
    for i, name in enumerate(["valence","arousal"]):
        corr,_ = pearsonr(fp32_out[:,i], qdq_out[:,i])
        diff = np.abs(fp32_out[:,i] - qdq_out[:,i])
        print(name)
        print("  Pearson:", corr)
        print("  Max diff:", diff.max())
        print("  Mean diff:", diff.mean())
        if math.isnan(corr) or corr < 0.92:
            print(f"  [FAIL] {name} Pearson {corr:.4f} < 0.92 threshold")
            failed = True
    if failed:
        sys.exit(1)
    print("PASS: quantization acceptable (Pearson >= 0.92 for all outputs)")
if __name__ == "__main__":
    main()
