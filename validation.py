# @path: validation.py
import sys
import numpy as np

from audio_utils import (
    DEFAULT_FRAMES,
    DEFAULT_HOP,
    DEFAULT_N_FFT,
    DEFAULT_N_MELS,
    DEFAULT_POWER,
    DEFAULT_SR,
    get_cpu_session,
    load_mel_patch,
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
    return load_mel_patch(
        path,
        sr=DEFAULT_SR,
        n_fft=DEFAULT_N_FFT,
        hop=DEFAULT_HOP,
        n_mels=DEFAULT_N_MELS,
        frames=DEFAULT_FRAMES,
        power=DEFAULT_POWER,
    )

def _load_stats(path):
    arr = np.load(path, allow_pickle=False)
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if not np.isfinite(arr).all():
        raise ValueError(f"Non-finite calibration stats found in {path}")
    return arr

def python_pipeline(mel):
    enc_sess = get_cpu_session(ENCODER)
    head_sess = get_cpu_session(HEAD)

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

    emb_mean = _load_stats(EMB_MEAN)
    emb_std = _load_stats(EMB_STD)

    emb_standardized = standardize_embedding(emb, emb_mean, emb_std)
    head_inp = prepare_vector_for_model(emb_standardized, head_shape)

    raw = np.asarray(
        head_sess.run(
            None,
            {head_input: head_inp.astype(np.float32)}
        )[0],
        dtype=np.float32,
    ).squeeze()

    return raw.reshape(-1)

def merged_pipeline(mel):
    sess = get_cpu_session(MERGED)
    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape

    inp = prepare_input_for_model(
        mel,
        input_shape,
        frames=DEFAULT_FRAMES,
        n_mels=DEFAULT_N_MELS,
    )

    out = np.asarray(
        sess.run(
            None,
            {input_name: inp.astype(np.float32)}
        )[0],
        dtype=np.float32,
    ).squeeze()

    return out.reshape(-1)

if __name__ == "__main__":
    mel = make_mel(AUDIO_PATH)

    py_out = python_pipeline(mel)
    merged_out = merged_pipeline(mel)

    print("Python reference:", py_out)
    print("Merged model    :", merged_out)

    diff = np.abs(py_out - merged_out)
    print("Absolute diff  :", diff)
    print("Max diff        :", diff.max())
    print("L2 diff         : ", float(np.linalg.norm(py_out - merged_out)))

    print("Python out:", py_out.tolist())
    print("Merged out:", merged_out.tolist())

    if merged_out.shape != (2,):
        print(f"ERROR: expected merged output shape (2,), got {merged_out.shape}", file=sys.stderr)
        sys.exit(2)

    if diff.max() >= 1e-4:
        print(f"ERROR: parity check failed, max diff={diff.max():.6e}", file=sys.stderr)
        sys.exit(3)
