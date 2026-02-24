import sys
import numpy as np
import onnxruntime as ort
import librosa

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
        power=2.0,
    )

    mel_db = librosa.power_to_db(mel, ref=np.max).T.astype(np.float32)

    if mel_db.shape[0] >= FRAMES:
        mel_db = mel_db[:FRAMES]
    else:
        pad = FRAMES - mel_db.shape[0]
        mel_db = np.vstack(
            [mel_db, np.full((pad, N_MELS), mel_db.min(), dtype=np.float32)]
        )

    return mel_db


def run(model, wav):
    mel = make_mel(wav)

    sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name

    out = sess.run(None, {inp: mel[np.newaxis]})

    print("Output:", out[0][0])


if __name__ == "__main__":
    model = sys.argv[1]
    wav = sys.argv[2]
    run(model, wav)