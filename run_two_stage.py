# @path: run_two_stage.py
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

if len(sys.argv) != 6:
    print("Usage: run_two_stage.py ENCODER_QDQ HEAD_QDQ MEAN_NPY STD_NPY TEST_WAV")
    sys.exit(1)

ENCODER = sys.argv[1]
HEAD = sys.argv[2]
MEAN = sys.argv[3]
STD = sys.argv[4]
WAV = sys.argv[5]

mean = _load_stats(MEAN)
std = _load_stats(STD)

mel = make_mel(WAV)

enc_sess = get_cpu_session(ENCODER)
head_sess = get_cpu_session(HEAD)

enc_input = enc_sess.get_inputs()[0].name
head_input = head_sess.get_inputs()[0].name
enc_input_shape = enc_sess.get_inputs()[0].shape
head_input_shape = head_sess.get_inputs()[0].shape

enc_inp = prepare_input_for_model(
    mel,
    enc_input_shape,
    frames=DEFAULT_FRAMES,
    n_mels=DEFAULT_N_MELS,
).astype(np.float32)

emb = enc_sess.run(None, {enc_input: enc_inp})[0]
emb = np.array(emb).squeeze().astype("float32")

std_emb = standardize_embedding(emb, mean, std)
head_inp = prepare_vector_for_model(std_emb, head_input_shape)

head_out = head_sess.run(None, {head_input: head_inp.astype(np.float32)})[0]
head_out = np.array(head_out).squeeze()

print("Head pre-tanh:", head_out.tolist())
print("Head tanh:", np.tanh(head_out).tolist())
