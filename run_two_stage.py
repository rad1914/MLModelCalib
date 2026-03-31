# @path: run_two_stage.py

import onnxruntime as ort
import numpy as np
import sys

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
