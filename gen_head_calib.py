#!/usr/bin/env python3
# gen_head_calib.py
import os, glob, numpy as np, onnxruntime as ort, librosa, sys
SR=16000; N_FFT=512; HOP=256; N_MELS=96; FRAMES=187

def make_mel(path):
    y,_ = librosa.load(path, sr=SR, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, power=2.0)
    mel_db = librosa.power_to_db(mel, ref=np.max).T.astype('float32')
    if mel_db.shape[0] >= FRAMES: return mel_db[:FRAMES]
    pad = FRAMES - mel_db.shape[0]; padv = mel_db.min() if mel_db.shape[0]>0 else -80.0
    return np.vstack([mel_db, np.full((pad,N_MELS), padv, dtype=np.float32)])

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: gen_head_calib.py ENCODER_ONNX CALIB_WAV_DIR MEAN_NPY STD_NPY")
        sys.exit(1)
    encoder, wavdir, meanp, stdp = sys.argv[1:]
    outdir = "head_calib"
    os.makedirs(outdir, exist_ok=True)
    mean = np.load(meanp).astype('float32')
    std = np.load(stdp).astype('float32')
    sess = ort.InferenceSession(encoder, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    files = glob.glob(os.path.join(wavdir, "*.wav"))
    i=0
    for f in files:
        mel = make_mel(f)
        emb = sess.run(None, {inp: mel[np.newaxis].astype('float32')})[0]
        emb = np.array(emb).squeeze().astype('float32')
        std_safe = np.where(std==0.0, 1.0, std)
        std_emb = ((emb - mean) / std_safe).astype('float32')
        np.save(os.path.join(outdir, f"calib_{i:04d}.npy"), std_emb)
        i+=1
    print("Saved", i, "standardized embeddings to", outdir)