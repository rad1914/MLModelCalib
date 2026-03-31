# @path: validation.py

import sys
import numpy as np
import onnxruntime as ort

from audio_utils import (
    DEFAULT_FRAMES,
    DEFAULT_HOP,
    DEFAULT_N_FFT,
    DEFAULT_N_MELS,
    DEFAULT_POWER,
    DEFAULT_SR,
    load_audio,
    make_mel_patch,
    prepare_input_for_model,
    prepare_vector_for_model,
    standardize_embedding,
)

AUDIO_PATH = "test.wav"

ENCODER = "msd_musicnn.onnx"
HEAD = "deam_head.onnx"
MERGED = "merged_std_correct_qdq.onnx"

EMB_MEAN = "emb_mean.npy"
EMB_STD = "emb_std.npy"

def make_mel(path):
    y = load_audio(path, sr=DEFAULT_SR)
    return make_mel_patch(
        y,
        sr=DEFAULT_SR,
        n_fft=DEFAULT_N_FFT,
        hop=DEFAULT_HOP,
        n_mels=DEFAULT_N_MELS,
        frames=DEFAULT_FRAMES,
        power=DEFAULT_POWER,
    )

def python_pipeline(mel):
    enc_sess = ort.InferenceSession(ENCODER, providers=["CPUExecutionProvider"])
    head_sess = ort.InferenceSession(HEAD, providers=["CPUExecutionProvider"])

    enc_input = enc_sess.get_inputs()[0].name
    head_input = head_sess.get_inputs()[0].name
    enc_shape = enc_sess.get_inputs()[0].shape
    head_shape = head_sess.get_inputs()[0].shape

    enc_inp = prepare_input_for_model(
        mel,
        enc_shape,
        frames=DEFAULT_FRAMES,
        n_mels=DEFAULT_N_MELS,
    )

    emb = enc_sess.run(
        None,
        {enc_input: enc_inp.astype(np.float32)}
    )[0]

    emb_mean = np.load(EMB_MEAN)
    emb_std = np.load(EMB_STD)
    emb_std = np.asarray(emb_std, dtype=np.float32)

    emb_stdized = standardize_embedding(emb, emb_mean, emb_std)
    head_inp = prepare_vector_for_model(emb_stdized, head_shape)

    raw = head_sess.run(
        None,
        {head_input: head_inp.astype(np.float32)}
    )[0][0]

    return np.asarray(raw, dtype=np.float32).reshape(-1)

def merged_pipeline(mel):
    sess = ort.InferenceSession(MERGED, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape

    inp = prepare_input_for_model(
        mel,
        input_shape,
        frames=DEFAULT_FRAMES,
        n_mels=DEFAULT_N_MELS,
    )

    out = sess.run(
        None,
        {input_name: inp.astype(np.float32)}
    )[0][0]

    return np.asarray(out, dtype=np.float32).reshape(-1)

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

    if diff.max() >= 1e-4:
        print(f"ERROR: parity check failed, max diff={diff.max():.6e}", file=sys.stderr)
        sys.exit(3)
