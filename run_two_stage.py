# @path: run_two_stage.py

import onnxruntime as ort
import numpy as np
import librosa
import sys

SR = 16000
N_FFT = 512
HOP = 256
N_MELS = 96
FRAMES = 187

def make_mel(path):
    y, _ = librosa.load(path, sr=SR, mono=True)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP,
        n_mels=N_MELS,
        power=2.0
    )

    mel_db = librosa.power_to_db(mel, ref=np.max).T.astype("float32")

    if mel_db.shape[0] >= FRAMES:
        return mel_db[:FRAMES]

    pad = FRAMES - mel_db.shape[0]
    padv = mel_db.min() if mel_db.shape[0] > 0 else -80.0

    return np.vstack([mel_db, np.full((pad, N_MELS), padv, dtype=np.float32)])

if len(sys.argv) != 6:
    print("Usage: run_two_stage.py ENCODER_QDQ HEAD_QDQ MEAN_NPY STD_NPY TEST_WAV")
    sys.exit(1)

ENCODER = sys.argv[1]
HEAD = sys.argv[2]
MEAN = sys.argv[3]
STD = sys.argv[4]
WAV = sys.argv[5]

mean = np.load(MEAN).astype("float32")
std = np.load(STD).astype("float32")

mel = make_mel(WAV)

enc_sess = ort.InferenceSession(ENCODER, providers=["CPUExecutionProvider"])
head_sess = ort.InferenceSession(HEAD, providers=["CPUExecutionProvider"])

enc_input = enc_sess.get_inputs()[0].name
head_input = head_sess.get_inputs()[0].name

emb = enc_sess.run(None, {enc_input: mel[np.newaxis].astype("float32")})[0]
emb = np.array(emb).squeeze().astype("float32")

std_safe = np.where(std == 0.0, 1.0, std)
std_emb = ((emb - mean) / std_safe).astype("float32")

head_out = head_sess.run(None, {head_input: std_emb[np.newaxis]})[0]
head_out = np.array(head_out).squeeze()

print("Head pre-tanh:", head_out.tolist())
print("Head tanh:", np.tanh(head_out).tolist())
