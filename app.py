"""Uploaded model's preprocessing and CPU inference; no waveform gain normalization."""
from __future__ import annotations

import io
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import librosa
import numpy as np
import soundfile as sf


def load_assets(directory: Path):
    import keras

    config = json.loads((directory / "vehicle_config.json").read_text(encoding="utf-8"))
    model = keras.saving.load_model(
        directory / "vehicle_model.keras", compile=False, safe_mode=True
    )
    expected = (
        int(config["n_mels"]),
        1 + round(config["sample_rate"] * config["seconds"]) // config["hop_length"],
        1,
    )
    if tuple(model.input_shape[1:]) != expected or model.output_shape[-1] != 1:
        raise ValueError(f"모델 입력/출력 구조를 확인하세요: {model.input_shape}, {model.output_shape}")
    return model, config


def prepare_audio(audio_bytes: bytes, config: dict):
    with sf.SoundFile(io.BytesIO(audio_bytes)) as source:
        original_seconds = source.frames / source.samplerate
        if original_seconds <= 0:
            raise ValueError("녹음이 비어 있습니다. 다시 녹음해 주세요.")
        if original_seconds > 120:
            raise ValueError("테스트용 입력은 2분 이하로 준비해 주세요.")
        audio = source.read(dtype="float32", always_2d=True)
        source_rate = source.samplerate
    if not np.isfinite(audio).all():
        raise ValueError("오디오에 잘못된 값이 포함되어 있습니다.")
    audio = audio.mean(axis=1)
    sample_rate = int(config["sample_rate"])
    if source_rate != sample_rate:
        audio = librosa.resample(
            audio, orig_sr=source_rate, target_sr=sample_rate, res_type="soxr_hq"
        )

    target = round(sample_rate * config["seconds"])
    length = len(audio)
    if config["crop"] != "center" or config["padding"] != "end_zeros":
        raise ValueError("지원하지 않는 자르기/패딩 설정입니다.")
    offset = max(0, (length - target) // 2)
    audio = audio[offset:offset + target]
    audio = np.pad(audio, (0, max(0, target - len(audio)))).astype(np.float32)

    # Fields absent in the exported config use explicit librosa defaults.
    # Do not convert the spectrogram into a PNG or normalize each recording.
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sample_rate, n_fft=int(config["n_fft"]),
        win_length=int(config["win_length"]), hop_length=int(config["hop_length"]),
        n_mels=int(config["n_mels"]), fmin=float(config["fmin"]),
        fmax=float(config["fmax"]), power=float(config["power"]),
        window=config.get("window", "hann"), center=config.get("center", True),
        pad_mode=config.get("pad_mode", "constant"),
        htk=config.get("htk", False), norm=config.get("mel_norm", "slaney"),
    )
    mel_db = librosa.power_to_db(
        mel, ref=float(config["db_ref"]), top_db=config["top_db"]
    )
    if float(config["std"]) <= 0:
        raise ValueError("정규화 표준편차는 0보다 커야 합니다.")
    features = ((mel_db - config["mean"]) / config["std"])[None, ..., None]
    info = {
        "original_seconds": original_seconds,
        "cropped": length > target,
        "padded": length < target,
        "start_seconds": offset / sample_rate,
        "sample_rate": sample_rate,
        "silent": bool(np.max(np.abs(audio)) < 1e-6),
    }
    return features.astype(np.float32), audio, mel_db, info


def predict_audio(model, audio_bytes: bytes, config: dict):
    features, waveform, mel_db, info = prepare_audio(audio_bytes, config)
    if tuple(features.shape[1:]) != tuple(model.input_shape[1:]):
        raise ValueError(f"전처리 크기 {features.shape}가 모델 입력 {model.input_shape}와 다릅니다.")
    score = float(np.asarray(model(features, training=False)).reshape(-1)[0])
    if not np.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("모델이 유효한 차량 점수를 반환하지 않았습니다.")
    return {"score": score, "waveform": waveform, "mel_db": mel_db, **info}
