# @path: quantize_merged.py

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
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("out")
    parser.add_argument("calib_dir")
    args = parser.parse_args()

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape
    print("Detected merged input:", input_name)

    files = sorted(glob.glob(os.path.join(args.calib_dir, "*.wav")))
    if not files:
        raise RuntimeError("No calibration WAV files found in " + args.calib_dir)

    class MelReader(CalibrationDataReader):
        def __init__(self, paths):
            self.paths = paths
            self.index = 0

        def get_next(self):
            if self.index >= len(self.paths):
                return None
            f = self.paths[self.index]
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
            inp = prepare_input_for_model(
                mel,
                input_shape,
                frames=DEFAULT_FRAMES,
                n_mels=DEFAULT_N_MELS,
            )
            return {input_name: inp.astype(np.float32)}

        def rewind(self):
            self.index = 0

    reader = MelReader(files)

    quantize_static(
        args.model,
        args.out,
        reader,
        quant_format=QuantFormat.QDQ,
        per_channel=False,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
    )

    print("Quantized merged model written to:", args.out)

if __name__ == "__main__":
    main()
