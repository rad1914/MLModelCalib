# @path: validation.py
import sys
import numpy as np
import librosa
import onnxruntime as ort

SR = 16000
N_FFT = 512
HOP = 256
N_MELS = 96
FRAMES = 187

AUDIO_PATH = "test.wav"

ENCODER = "msd_musicnn.onnx"
HEAD = "deam_head.onnx"
MERGED = "merged_std_correct_qdq.onnx"

EMB_MEAN = "emb_mean.npy"
EMB_STD = "emb_std.npy"

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

    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = mel_db.T.astype(np.float32)

    T = mel_db.shape[0]

    if T >= FRAMES:
        mel_db = mel_db[:FRAMES, :]
    else:
        pad_rows = FRAMES - T
        pad_value = mel_db.min() if T > 0 else -80.0
        mel_db = np.vstack([
            mel_db,
            np.full((pad_rows, N_MELS), pad_value, dtype=np.float32)
        ])

    return mel_db

def python_pipeline(mel):
    enc_sess = ort.InferenceSession(ENCODER, providers=["CPUExecutionProvider"])
    head_sess = ort.InferenceSession(HEAD, providers=["CPUExecutionProvider"])

    enc_input = enc_sess.get_inputs()[0].name
    head_input = head_sess.get_inputs()[0].name

    emb = enc_sess.run(
        None,
        {enc_input: mel[np.newaxis, :, :]}
    )[0]

    emb_mean = np.load(EMB_MEAN)
    emb_std = np.load(EMB_STD)
    emb_std = np.where(emb_std < 1e-8, 1.0, emb_std)

    emb_stdized = (emb - emb_mean) / emb_std

    raw = head_sess.run(
        None,
        {head_input: emb_stdized.astype(np.float32)}
    )[0][0]

    final = np.tanh(raw)

    return final

def merged_pipeline(mel):
    sess = ort.InferenceSession(MERGED, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    out = sess.run(
        None,
        {input_name: mel[np.newaxis, :, :]}
    )[0][0]

    return out

if __name__ == "__main__":
    mel = make_mel(AUDIO_PATH)

    py_out = python_pipeline(mel)
    merged_out = merged_pipeline(mel)

    print("Python reference:", py_out)
    print("Merged model   :", merged_out)

    diff = np.abs(py_out - merged_out)
    print("Absolute diff  :", diff)
    print("Max diff       :", diff.max())
    print("L2 diff        :", float(np.linalg.norm(py_out - merged_out)))

    print("Python out:", py_out.tolist())
    print("Merged out:", merged_out.tolist())

    if merged_out.shape != (2,):
        print(f"ERROR: expected merged output shape (2,), got {merged_out.shape}", file=sys.stderr)
        sys.exit(2)

    if diff.max() >= 1e-3:
        print(f"ERROR: parity check failed, max diff={diff.max():.6e}", file=sys.stderr)
        sys.exit(3)
