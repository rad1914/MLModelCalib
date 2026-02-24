#!/usr/bin/env python3
# run_debug_inference.py
import sys, os
import numpy as np
import onnxruntime as ort
import librosa

if len(sys.argv) < 3:
    print("Usage: python run_debug_inference.py <debug_model.onnx> <test.wav>")
    sys.exit(1)

model_path = sys.argv[1]
wav_path = sys.argv[2]

# audio / mel params (match your pipeline)
SR = 16000
N_FFT = 512
HOP = 256
N_MELS = 96
FRAMES = 187
WIN_SEC = 3.0

def make_mel_from_wav(wav):
    y, _ = librosa.load(wav, sr=SR, mono=True)
    win_samples = int(WIN_SEC * SR)
    chunk = y[:win_samples] if len(y)>=win_samples else y
    mel = librosa.feature.melspectrogram(y=chunk, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, power=2.0)
    mel_db = librosa.power_to_db(mel, ref=np.max).T
    if mel_db.shape[0] >= FRAMES:
        patch = mel_db[:FRAMES,:].astype(np.float32)
    else:
        pad_val = float(mel_db.min()) if mel_db.size else -80.0
        pad = np.full((FRAMES - mel_db.shape[0], N_MELS), pad_val, dtype=np.float32)
        patch = np.vstack([mel_db.astype(np.float32), pad])
    # default expected input shape: (1, FRAMES, N_MELS)
    return patch[np.newaxis, :, :].astype(np.float32)

print("Loading model:", model_path)
sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
in_name = sess.get_inputs()[0].name
outs = [o.name for o in sess.get_outputs()]
print("Model I/O -> in:", in_name, "outs:", outs)

mel = make_mel_from_wav(wav_path)
print("Mel shape:", mel.shape)

# try to adapt input shape if needed
inp_shape = sess.get_inputs()[0].shape
# if model expects 4 dims and we have 3 dims, add trailing singleton
if len(inp_shape) == 4 and mel.ndim == 3:
    mel = mel[:, :, :, np.newaxis]

print("Running inference...")
res = sess.run(None, {in_name: mel})
print("Got", len(res), "outputs.\n")
for name, arr in zip(outs, res):
    a = np.asarray(arr)
    print("Output:", name, "shape:", a.shape)
    print("  mean: %.6f  std: %.6f  min: %.6f  max: %.6f" % (float(a.mean()), float(a.std()), float(a.min()), float(a.max())))
    # show small sample
    flat = a.flatten()
    print("  sample[0..7]:", flat[:8].tolist())
    print()