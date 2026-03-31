# @path: CalibrationDataReader.py
import librosa

SR = 16000
N_FFT = 512
HOP = 256
N_MELS = 96
FRAMES = 187

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
            INPUT_NAME: mel[np.newaxis].astype(np.float32)
        }

    def rewind(self):
        self.iter = iter(self.files)
