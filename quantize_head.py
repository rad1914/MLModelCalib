
#!/usr/bin/env python3
# quantize_head.py
# Static QDQ quantization for merged MusiCNN + DEAM model (mel input)

import os
import sys
import glob
import numpy as np
import librosa
import onnxruntime as ort

from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantFormat,
    QuantType,
)

# ======================
# CLI
# ======================

MODEL = sys.argv[1]
OUT = sys.argv[2]
CALIB_DIR = sys.argv[3]

# ======================
# AUDIO PARAMS (MUST MATCH TRAINING)
# ======================

SR = 16000
N_FFT = 512
HOP = 256
N_MELS = 96
FRAMES = 187

# ======================
# DETECT MODEL INPUT NAME
# ======================

sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
INPUT_NAME = sess.get_inputs()[0].name
print("Detected model input:", INPUT_NAME)

# ======================
# MEL GENERATION
# ======================

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
        pad_val = mel_db.min() if mel_db.shape[0] > 0 else -80.0
        mel_db = np.vstack(
            [mel_db, np.full((pad, N_MELS), pad_val, dtype=np.float32)]
        )

    return mel_db

# ======================
# CALIBRATION READER
# ======================

class MelReader(CalibrationDataReader):

    def __init__(self, files):
        self.files = files
        self.index = 0

    def get_next(self):

        if self.index >= len(self.files):
            return None

        f = self.files[self.index]
        self.index += 1

        mel = make_mel(f)

        return {
            INPUT_NAME: mel[np.newaxis, :, :].astype(np.float32)
        }

    def rewind(self):
        self.index = 0

# ======================
# LOAD CALIB FILES
# ======================

files = sorted(glob.glob(os.path.join(CALIB_DIR, "*.wav")))

if not files:
    raise RuntimeError("No calibration WAV files found in " + CALIB_DIR)

print("Calibration files:", len(files))

reader = MelReader(files)

# ======================
# QUANTIZATION
# ======================

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