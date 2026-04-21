from __future__ import annotations

from pathlib import Path

from faster_whisper import WhisperModel


def _fmt_ts(t: float) -> str:
    if t < 0:
        t = 0.0
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    if ms == 1000:
        ms = 0
        s += 1
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def transcribe_to_srt(audio: Path, srt: Path, *, model_size: str = "small") -> str:
    """Run faster-whisper and write an SRT file. Returns detected language."""
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(
        str(audio),
        beam_size=1,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )

    lines: list[str] = []
    seq = 1
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        lines.append(str(seq))
        lines.append(f"{_fmt_ts(seg.start)} --> {_fmt_ts(seg.end)}")
        lines.append(text)
        lines.append("")
        seq += 1

    srt.write_text("\n".join(lines), encoding="utf-8")
    return info.language or ""
