# diagnose.py
import numpy as np, onnx, onnxruntime as ort
import librosa

# files
ENC = "msd_musicnn.onnx"
HEAD = "deam_head.onnx"
WAV = "test.wav"

# load audio & embedding
y, sr = librosa.load(WAV, sr=16000, mono=True)
mel = librosa.feature.melspectrogram(y=y, sr=16000, n_fft=512, hop_length=256, n_mels=96, power=2.0)
mel_db = librosa.power_to_db(mel, ref=np.max).T.astype(np.float32)
mel_patch = mel_db[:187,:] if mel_db.shape[0] >= 187 else np.vstack([mel_db, np.full((187-mel_db.shape[0],96), mel_db.min(), dtype=np.float32)])
inp = mel_patch[np.newaxis,:,:].astype(np.float32)

sess_enc = ort.InferenceSession(ENC, providers=['CPUExecutionProvider'])
emb = sess_enc.run(None, {sess_enc.get_inputs()[0].name: inp})[0]  # shape (1,200)

print("=== EMBEDDING STATS ===")
print("shape:", emb.shape)
print("min,median,mean,max,std:", float(emb.min()), float(np.median(emb)), float(emb.mean()), float(emb.max()), float(emb.std()))

# run head raw
sess_head = ort.InferenceSession(HEAD, providers=['CPUExecutionProvider'])
head_in = sess_head.get_inputs()[0].name
out = sess_head.run(None, {head_in: emb.astype(np.float32)})[0]
print("\n=== HEAD RAW OUTPUT ===")
print("output shape(s):", [o.shape for o in out if hasattr(o,'shape')])
print("values:", out[0])

# inspect ONNX head initializers for potential mean/std (shape 200 or 200x1)
print("\n=== HEAD GRAPH INSPECT ===")
h = onnx.load(HEAD)
names = [init.name for init in h.graph.initializer]
print("initializers count:", len(names))
# show initializers with shape lengths 200 (likely mean/std or bias)
for t in h.graph.initializer:
    dims = [d for d in t.dims]
    if 190 <= (dims[0] if dims else 0) <= 210 or (len(dims)>1 and 190 <= dims[1] <= 210):
        print("possible vector initializer:", t.name, "shape=", dims)