# @path: quantize_head.py
import argparse
import glob
import os

import numpy as np

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
    build_model_input_from_path,
    get_cpu_session,
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

enc_sess = get_cpu_session(ENCODER_PATH)
enc_input_name = enc_sess.get_inputs()[0].name
enc_input_shape = enc_sess.get_inputs()[0].shape
head_sess = get_cpu_session(MODEL)
HEAD_INPUT_NAME = head_sess.get_inputs()[0].name
HEAD_INPUT_SHAPE = head_sess.get_inputs()[0].shape
print("Detected head input:", HEAD_INPUT_NAME)

def _load_stats(path):
    arr = np.load(path, allow_pickle=False)
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise RuntimeError(f"Empty calibration stats in {path}")
    if not np.isfinite(arr).all():
        raise RuntimeError(f"Non-finite calibration stats in {path}")
    return arr

emb_mean = _load_stats(EMB_MEAN_PATH)
emb_std = _load_stats(EMB_STD_PATH)

class MelReader(CalibrationDataReader):

    def __init__(self, files):
        self.files = files
        self.index = 0

    def get_next(self):

        if self.index >= len(self.files):
            return None

        f = self.files[self.index]
        self.index += 1

        enc_inp = build_model_input_from_path(
            f,
            enc_input_shape,
            sr=DEFAULT_SR,
            n_fft=DEFAULT_N_FFT,
            hop=DEFAULT_HOP,
            n_mels=DEFAULT_N_MELS,
            frames=DEFAULT_FRAMES,
            power=DEFAULT_POWER,
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
