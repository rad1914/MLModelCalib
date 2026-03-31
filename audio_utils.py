# @path: audio_utils.py
from __future__ import annotations

from typing import Sequence

import librosa
import numpy as np

DEFAULT_SR = 16000
DEFAULT_N_FFT = 512
DEFAULT_HOP = 256
DEFAULT_N_MELS = 96
DEFAULT_FRAMES = 187
DEFAULT_POWER = 2.0
DEFAULT_CLIP_MIN = -80.0
DEFAULT_CLIP_MAX = 0.0
STD_FLOOR = 1e-6

def _dim_to_int(value):
    try:
        return int(value)
    except Exception:
        return None

def load_audio(path: str, sr: int = DEFAULT_SR) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    return np.asarray(y, dtype=np.float32)

def make_mel_patch(
    y: np.ndarray,
    sr: int = DEFAULT_SR,
    n_fft: int = DEFAULT_N_FFT,
    hop: int = DEFAULT_HOP,
    n_mels: int = DEFAULT_N_MELS,
    frames: int = DEFAULT_FRAMES,
    power: float = DEFAULT_POWER,
    clip_min: float = DEFAULT_CLIP_MIN,
    clip_max: float = DEFAULT_CLIP_MAX,
) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop,
        n_mels=n_mels,
        power=power,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = np.clip(mel_db, clip_min, clip_max).T.astype(np.float32)

    if mel_db.shape[0] >= frames:
        return mel_db[:frames]

    pad_rows = frames - mel_db.shape[0]
    pad_value = float(mel_db.min()) if mel_db.size else clip_min
    return np.vstack(
        [mel_db, np.full((pad_rows, n_mels), pad_value, dtype=np.float32)]
    )

def prepare_input_for_model(
    patch: np.ndarray,
    model_input_shape: Sequence | None,
    frames: int = DEFAULT_FRAMES,
    n_mels: int = DEFAULT_N_MELS,
) -> np.ndarray:
    arr = np.asarray(patch, dtype=np.float32)[np.newaxis, :, :]

    if not model_input_shape:
        return arr

    shape = list(model_input_shape)

    if len(shape) == 3:
        s1 = _dim_to_int(shape[1])
        s2 = _dim_to_int(shape[2])
        if (s1 is None or s1 == frames) and (s2 is None or s2 == n_mels):
            return arr
        if (s1 is None or s1 == n_mels) and (s2 is None or s2 == frames):
            return arr.transpose(0, 2, 1)

    if len(shape) == 4:
        s1 = _dim_to_int(shape[1])
        s2 = _dim_to_int(shape[2])
        s3 = _dim_to_int(shape[3])
        if (s1 is None or s1 == frames) and (s2 is None or s2 == n_mels):
            return arr[:, :, :, np.newaxis]
        if (s2 is None or s2 == frames) and (s3 is None or s3 == n_mels):
            return arr[:, np.newaxis, :, :]
        if (s1 is None or s1 == n_mels) and (s2 is None or s2 == frames):
            return arr.transpose(0, 2, 1)[:, :, :, np.newaxis]

    return arr

def prepare_vector_for_model(
    vector: np.ndarray,
    model_input_shape: Sequence | None,
) -> np.ndarray:
    vec = np.asarray(vector, dtype=np.float32).reshape(-1)

    if not model_input_shape:
        return vec[np.newaxis, :]

    shape = list(model_input_shape)

    if len(shape) == 1:
        return vec

    if len(shape) == 2:
        s0 = _dim_to_int(shape[0])
        s1 = _dim_to_int(shape[1])
        if s1 is None or s1 == vec.size:
            return vec[np.newaxis, :]
        if s0 is None or s0 == vec.size:
            return vec[:, np.newaxis]

    return vec[np.newaxis, :]

def standardize_embedding(
    emb: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    floor: float = STD_FLOOR,
) -> np.ndarray:
    emb = np.asarray(emb, dtype=np.float32).reshape(-1)
    mean = np.asarray(mean, dtype=np.float32).reshape(-1)
    std = np.asarray(std, dtype=np.float32).reshape(-1)
    std_safe = np.maximum(std, floor)
    return ((emb - mean) / std_safe).astype(np.float32)
