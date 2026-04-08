# @path: quantize_encoder.py
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)
import glob
import os
import sys

import numpy as np

from audio_utils import (
    DEFAULT_FRAMES,
    DEFAULT_HOP,
    DEFAULT_N_FFT,
    DEFAULT_N_MELS,
    DEFAULT_POWER,
    DEFAULT_SR,
    build_model_input_from_path,
    get_cpu_session,
)

MODEL = sys.argv[1]
OUT = sys.argv[2]
CALIB_DIR = sys.argv[3]

sess = get_cpu_session(MODEL)
INPUT_NAME = sess.get_inputs()[0].name
INPUT_SHAPE = sess.get_inputs()[0].shape
print("Detected model input:", INPUT_NAME)

class MelReader(CalibrationDataReader):
    def __init__(self, files):
        self.files = files
        self.iter = iter(self.files)

    def get_next(self):
        try:
            f = next(self.iter)
        except StopIteration:
            return None

        inp = build_model_input_from_path(
            f,
            INPUT_SHAPE,
            sr=DEFAULT_SR,
            n_fft=DEFAULT_N_FFT,
            hop=DEFAULT_HOP,
            n_mels=DEFAULT_N_MELS,
            frames=DEFAULT_FRAMES,
            power=DEFAULT_POWER,
        )

        return {
            INPUT_NAME: inp.astype(np.float32)
        }

    def rewind(self):
        self.iter = iter(self.files)

files = sorted(glob.glob(os.path.join(CALIB_DIR, "*.wav")))

if not files:
    raise RuntimeError("No calibration WAV files found in " + CALIB_DIR)

dr = MelReader(files)

quantize_static(
    MODEL,
    OUT,
    dr,
    quant_format=QuantFormat.QDQ,
    per_channel=False,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8
)

print("Wrote", OUT)
