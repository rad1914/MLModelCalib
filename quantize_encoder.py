# @path: quantize_encoder.py

from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType
import glob, os, librosa, numpy as np, sys, onnxruntime as ort

MODEL=sys.argv[1]
OUT=sys.argv[2]
CALIB_DIR=sys.argv[3]

SR=16000
N_FFT=512
HOP=256
N_MELS=96
FRAMES=187

def make_mel(path):
    y,_ = librosa.load(path, sr=SR, mono=True)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP,
        n_mels=N_MELS,
        power=2.0
    )
    mel_db = librosa.power_to_db(mel, ref=np.max).T.astype('float32')

    if mel_db.shape[0] >= FRAMES:
        return mel_db[:FRAMES]

    pad = FRAMES - mel_db.shape[0]
    padv = mel_db.min() if mel_db.shape[0] > 0 else -80.0
    return np.vstack([mel_db, np.full((pad, N_MELS), padv, dtype=np.float32)])

sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
INPUT_NAME = sess.get_inputs()[0].name
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

        mel = make_mel(f)

        return {
            INPUT_NAME: mel[np.newaxis].astype('float32')
        }

    def rewind(self):
        self.iter = iter(self.files)

files = glob.glob(os.path.join(CALIB_DIR, "*.wav"))

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
