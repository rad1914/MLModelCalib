# @path: verify_merged.py

import argparse
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

def run_onnx(model_path, mel):
    sess = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"]
    )

    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape
    inp = prepare_input_for_model(
        mel,
        input_shape,
        frames=DEFAULT_FRAMES,
        n_mels=DEFAULT_N_MELS,
    )

    outputs = sess.run(
        None,
        {input_name: inp.astype(np.float32)}
    )

    return outputs[0][0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--audio", required=True)

    args = parser.parse_args()

    mel = make_mel(args.audio)

    out = run_onnx(args.model, mel)

    print("Model:", args.model)
    print("Output:", out)

if __name__ == "__main__":
    main()
