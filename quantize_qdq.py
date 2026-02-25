#!/usr/bin/env python3
"""
quantize_qdq.py

Static QDQ quantization helper for merged_std_final.onnx using a mel-based
calibration set. Compatible with common ONNX Runtime versions (tries
keyword-style call first, falls back to positional if needed).
"""

import argparse
import os
import sys
import numpy as np
import librosa
import soundfile as sf
import onnx
import inspect
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantFormat,
    QuantType,
)

SR = 16000
N_FFT = 512
HOP = 256
N_MELS = 96
TARGET_FRAMES = 187


def load_wav(path: str) -> np.ndarray:
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return y


def compute_mel(y: np.ndarray) -> np.ndarray:
    S = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, power=2.0
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db.astype(np.float32)


def pad_or_trim(mel: np.ndarray, frames: int = TARGET_FRAMES) -> np.ndarray:
    n_mels, t = mel.shape
    if t == frames:
        return mel
    if t < frames:
        pad = np.zeros((n_mels, frames - t), dtype=np.float32)
        return np.concatenate([mel, pad], axis=1)
    start = (t - frames) // 2
    return mel[:, start : start + frames]


def wav_to_input(path: str) -> np.ndarray:
    y = load_wav(path)
    mel = compute_mel(y)
    mel = pad_or_trim(mel)
    mel = mel.T[np.newaxis, :, :]  # shape -> (1, 187, 96)
    return mel.astype(np.float32)


class MelCalibrationReader(CalibrationDataReader):
    """
    CalibrationDataReader implementation that yields mel spectrogram inputs for the model.
    """

    def __init__(self, model_path: str, wav_dir: str):
        # Load the model to extract the first input name
        self.model = onnx.load(model_path)
        if len(self.model.graph.input) == 0:
            raise RuntimeError("Model graph has no inputs")
        self.input_name = self.model.graph.input[0].name

        files = [
            os.path.join(wav_dir, f)
            for f in sorted(os.listdir(wav_dir))
            if f.lower().endswith(".wav")
        ]
        if not files:
            raise RuntimeError("No WAV files found in calibration directory")

        self.files = files
        self.enum_data = None
        self._prepare()

    def _prepare(self):
        data = []
        for f in self.files:
            mel = wav_to_input(f)
            data.append({self.input_name: mel})
        self.enum_data = iter(data)

    def get_next(self):
        # ONNX Runtime expects this method name
        return next(self.enum_data, None)


def call_quantize_static_safe(
    model_fp32: str,
    model_output: str,
    reader: CalibrationDataReader,
    quant_format=QuantFormat.QDQ,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    per_channel: bool = False,
):
    """
    Call quantize_static while handling API differences between ONNX Runtime releases.
    Tries keyword-style call first; on TypeError or unexpected signature falls back to positional.
    """
    sig = inspect.signature(quantize_static)
    params = list(sig.parameters.keys())

    # Preferred kwargs set (common in docs)
    kwargs = dict(
        model_input=model_fp32,
        model_output=model_output,
        calibration_data_reader=reader,
        quant_format=quant_format,
        activation_type=activation_type,
        weight_type=weight_type,
        per_channel=per_channel,
    )

    try:
        # Try calling with keywords (most readable)
        quantize_static(**kwargs)
        return
    except TypeError as e_kw:
        # Fallback: try positional call (older/newer variants)
        try:
            quantize_static(
                model_fp32,
                model_output,
                reader,
                quant_format=quant_format,
                activation_type=activation_type,
                weight_type=weight_type,
                per_channel=per_channel,
            )
            return
        except Exception as e_pos:
            # Last-ditch: print both errors for debugging and re-raise the last
            print("quantize_static keyword-call error:", file=sys.stderr)
            print(repr(e_kw), file=sys.stderr)
            print("quantize_static positional-call error:", file=sys.stderr)
            print(repr(e_pos), file=sys.stderr)
            raise


def main():
    parser = argparse.ArgumentParser(description="Static QDQ quantization helper")
    parser.add_argument("--model_fp32", required=True, help="Path to FP32 ONNX model")
    parser.add_argument("--calib_dir", required=True, help="Directory with WAV files for calibration")
    parser.add_argument("--output", required=True, help="Path to write quantized ONNX")
    parser.add_argument("--per_channel", action="store_true", help="Enable per-channel quantization for weights")
    args = parser.parse_args()

    if not os.path.exists(args.model_fp32):
        print("ERROR: model_fp32 not found:", args.model_fp32, file=sys.stderr)
        sys.exit(2)
    if not os.path.isdir(args.calib_dir):
        print("ERROR: calib_dir not found or not a directory:", args.calib_dir, file=sys.stderr)
        sys.exit(2)

    try:
        import onnxruntime as ort

        print("onnxruntime version:", ort.__version__, file=sys.stderr)
    except Exception:
        print("Warning: unable to import onnxruntime for version info", file=sys.stderr)

    print("Loading calibration data...", file=sys.stderr)
    reader = MelCalibrationReader(args.model_fp32, args.calib_dir)

    print("Running static QDQ quantization...", file=sys.stderr)
    call_quantize_static_safe(
        model_fp32=args.model_fp32,
        model_output=args.output,
        reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=args.per_channel,
    )

    if os.path.exists(args.output):
        print("Quantized model saved to:", args.output, file=sys.stderr)
    else:
        print("Quantize finished but output file not found. Check for errors.", file=sys.stderr)


if __name__ == "__main__":
    main()