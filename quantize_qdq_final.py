# @path: quantize_qdq_final.py
import sys, os, numpy as np
import soundfile as sf
from onnxruntime.quantization import (
     quantize_static, CalibrationDataReader,
     QuantFormat, QuantType,
     CalibrationMethod
)
class AudioCalibReader(CalibrationDataReader):
    MAX_CALIB_FILES = 60
    def __init__(self, calib_dir, model_path):
        import onnxruntime as ort
        self.idx    = 0
        _sess = ort.InferenceSession(model_path,
                                     providers=["CPUExecutionProvider"])
        self.input_name = _sess.get_inputs()[0].name
        all_files = sorted(
            os.path.join(calib_dir, f)
            for f in os.listdir(calib_dir) if f.endswith(".wav")
        )
        files = all_files[: self.MAX_CALIB_FILES]
        if not files:
            raise RuntimeError(f"No .wav files found in {calib_dir}")
        print(f"  ↳ streaming {len(files)} / {len(all_files)} mel patches ...")
        self.files = files
    def rewind(self):
        self.idx = 0
    @staticmethod
    def _mel(f, SR=16000, N_FFT=512, HOP=256, N_MELS=96, FRAMES=187):
        import librosa
        y, sr = sf.read(f, dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        m = librosa.power_to_db(librosa.feature.melspectrogram(
            y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS
        ), ref=np.max).T.astype(np.float32)
        if len(m) < FRAMES:
            m = np.pad(m, ((0, FRAMES - len(m)), (0, 0)), constant_values=m.min())
        return m[:FRAMES][None]
    def get_next(self):
        if self.idx >= len(self.files):
            return None
        f = self.files[self.idx]
        item = {self.input_name: self._mel(f)}
        self.idx += 1
        return item
def main():
    if len(sys.argv) != 4:
        print("Usage: quantize_qdq_final.py fp32.onnx calib_dir out.onnx")
        sys.exit(1)
    fp32, calib_dir, out = sys.argv[1:]
    reader = AudioCalibReader(calib_dir, fp32)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    quantize_static(
        model_input=fp32,
        model_output=out,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=False,
        reduce_range=False,
        calibrate_method=CalibrationMethod.Entropy,
        op_types_to_quantize=["MatMul", "Gemm"],
    )
    print("Saved:", out)
if __name__ == "__main__":
    main()
