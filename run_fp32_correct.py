import numpy as np, librosa, onnxruntime as ort
import sys

wav = sys.argv[1]
model = sys.argv[2]

y, sr = librosa.load(wav, sr=16000, mono=True)

# 6s pad/crop
if len(y) < 16000*6:
    y = np.pad(y, (0, 16000*6 - len(y)))
else:
    y = y[:16000*6]

# MEL → (96, 376)
S = librosa.feature.melspectrogram(
    y=y, sr=16000,
    n_fft=512, hop_length=256,
    n_mels=96, power=2.0
)
S_db = librosa.power_to_db(S, ref=np.max)

# Trim 376 → 374
S_db = S_db[:, :374]

# Downsample: 374 → 187
mel = 0.5 * (S_db[:, 0::2] + S_db[:, 1::2])   # (96,187)

# Transpose → (1,187,96)
inp = mel.T.astype(np.float32)[None, :, :]

sess = ort.InferenceSession(model, providers=['CPUExecutionProvider'])
inp_name = sess.get_inputs()[0].name
out = sess.run(None, {inp_name: inp})

print("Output:", out)