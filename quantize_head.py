# @path: quantize_head.py

import argparse
import glob
import os

import numpy as np
import onnxruntime as ort

from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)

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

parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("out")
parser.add_argument("calib_dir")
parser.add_argument("--encoder", required=True, help="Encoder ONNX used to produce calibration embeddings")
parser.add_argument("--emb-mean", required=True, help="Calibration embedding mean .npy")
parser.add_argument("--emb-std", required=True, help="Calibration embedding std .npy")
args = parser.parse_args()

MODEL = args.model
OUT = args.out
CALIB_DIR = args.calib_dir
ENCODER_PATH = args.encoder
EMB_MEAN_PATH = args.emb_mean
EMB_STD_PATH = args.emb_std

enc_sess = ort.InferenceSession(ENCODER_PATH, providers=["CPUExecutionProvider"])
enc_input_name = enc_sess.get_inputs()[0].name
enc_input_shape = enc_sess.get_inputs()[0].shape
head_sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
HEAD_INPUT_NAME = head_sess.get_inputs()[0].name
HEAD_INPUT_SHAPE = head_sess.get_inputs()[0].shape
print("Detected head input:", HEAD_INPUT_NAME)

emb_mean = np.load(EMB_MEAN_PATH).astype(np.float32)
emb_std = np.load(EMB_STD_PATH).astype(np.float32)

class MelReader(CalibrationDataReader):

    def __init__(self, files):
        self.files = files
        self.index = 0

    def get_next(self):

        if self.index >= len(self.files):
            return None

        f = self.files[self.index]
        self.index += 1

        y = load_audio(f, sr=DEFAULT_SR)
        mel = make_mel_patch(
            y,
            sr=DEFAULT_SR,
            n_fft=DEFAULT_N_FFT,
            hop=DEFAULT_HOP,
            n_mels=DEFAULT_N_MELS,
            frames=DEFAULT_FRAMES,
            power=DEFAULT_POWER,
        )
        enc_inp = prepare_input_for_model(
            mel,
            enc_input_shape,
            frames=DEFAULT_FRAMES,
            n_mels=DEFAULT_N_MELS,
        ).astype(np.float32)
        emb = np.asarray(
            enc_sess.run(None, {enc_input_name: enc_inp})[0]
        ).squeeze()
        std_emb = standardize_embedding(emb, emb_mean, emb_std)
        head_inp = prepare_vector_for_model(std_emb, HEAD_INPUT_SHAPE)

        return {
            HEAD_INPUT_NAME: head_inp.astype(np.float32)
        }

    def rewind(self):
        self.index = 0

files = sorted(glob.glob(os.path.join(CALIB_DIR, "*.wav")))

if not files:
    raise RuntimeError("No calibration WAV files found in " + CALIB_DIR)

print("Calibration files:", len(files))

reader = MelReader(files)

quantize_static(
    MODEL,
    OUT,
    reader,
    quant_format=QuantFormat.QDQ,
    per_channel=False,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
)

print("Quantized model written to:", OUT)
