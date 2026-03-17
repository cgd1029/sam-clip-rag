from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import lfilter

NOTE_TO_SEMITONE = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
}


@dataclass
class TimbreProfile:
    lpc: np.ndarray
    sample_rate: int
    rms: float


def note_to_hz(note: str) -> float:
    note = note.strip()
    if len(note) < 2:
        raise ValueError(f"无效音符: {note}")

    if note[1] in ["#", "b"]:
        pitch_class = note[:2]
        octave = int(note[2:])
    else:
        pitch_class = note[0]
        octave = int(note[1:])

    semitone = NOTE_TO_SEMITONE[pitch_class]
    midi = (octave + 1) * 12 + semitone
    return 440.0 * (2 ** ((midi - 69) / 12))


def parse_score(score: str) -> List[Tuple[float, float]]:
    """
    输入格式：C4:0.5, D4:0.5, E4:1.0
    返回: [(freq_hz, duration_sec), ...]
    """
    events: List[Tuple[float, float]] = []
    for token in score.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"乐谱片段缺少时长: {token}")
        note, dur = token.split(":", 1)
        events.append((note_to_hz(note.strip()), float(dur.strip())))

    if not events:
        raise ValueError("乐谱为空")
    return events


def _safe_lpc(frame: np.ndarray, order: int) -> np.ndarray:
    try:
        return librosa.lpc(frame, order=order)
    except np.linalg.LinAlgError:
        return np.array([1.0] + [0.0] * order)


def extract_timbre_profile(audio_path: str, sr: int = 24000, lpc_order: int = 16) -> TimbreProfile:
    wav, _ = librosa.load(audio_path, sr=sr, mono=True)
    if wav.size < sr // 2:
        raise ValueError("示例录音太短，至少需要 0.5 秒")

    wav = librosa.util.normalize(wav)

    frame_len = 1024
    hop = 512
    frames = librosa.util.frame(wav, frame_length=frame_len, hop_length=hop).T

    lpcs = []
    for frame in frames:
        frame = frame * np.hamming(frame_len)
        lpcs.append(_safe_lpc(frame, order=lpc_order))

    lpc_avg = np.mean(np.stack(lpcs, axis=0), axis=0)
    rms = float(np.sqrt(np.mean(wav**2)))

    return TimbreProfile(lpc=lpc_avg, sample_rate=sr, rms=rms)


def _synth_excitation(score_events: List[Tuple[float, float]], sr: int, vibrato_hz: float = 5.0) -> np.ndarray:
    chunks = []
    for f0, dur in score_events:
        n = max(1, int(sr * dur))
        t = np.linspace(0, dur, n, endpoint=False)

        vib = 0.02 * np.sin(2 * np.pi * vibrato_hz * t)
        inst_freq = f0 * (1.0 + vib)
        phase = 2 * np.pi * np.cumsum(inst_freq) / sr

        sig = np.sin(phase) + 0.35 * np.sin(2 * phase) + 0.2 * np.sin(3 * phase)

        attack = int(0.05 * n)
        release = int(0.08 * n)
        env = np.ones(n)
        if attack > 0:
            env[:attack] = np.linspace(0.0, 1.0, attack)
        if release > 0:
            env[-release:] = np.linspace(1.0, 0.0, release)

        chunks.append(sig * env)

    return np.concatenate(chunks)


def synthesize_singing(
    profile: TimbreProfile,
    score: str,
    out_path: str,
    lyrics: str | None = None,
    tempo_scale: float = 1.0,
) -> str:
    del lyrics  # MVP 版本暂未做歌词到发音器映射，后续可扩展到 G2P + 声码器。

    events = parse_score(score)
    if tempo_scale <= 0:
        raise ValueError("tempo_scale 必须 > 0")

    scaled_events = [(f, d / tempo_scale) for f, d in events]
    excitation = _synth_excitation(scaled_events, sr=profile.sample_rate)

    # 全极点滤波: 1 / A(z)
    y = lfilter([1.0], profile.lpc, excitation)
    y = librosa.util.normalize(y)
    y *= min(0.95, max(0.2, profile.rms * 2.5))

    sf.write(out_path, y, profile.sample_rate)
    return out_path
