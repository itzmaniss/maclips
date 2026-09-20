"""S0-S14 from PLAN.md §2.1, as stubs.

Every stage body is a placeholder that returns plausible output so the pipeline
runs end to end (build step 1's "done when"). What is *not* stubbed is the
gates: each gate condition from §2.1 is implemented against the stage's output
and the run config, so the gate machinery is real and testable now. Replacing
a stub means deleting its placeholder output and keeping its gate.

Build order (PLAN.md §8): S2-S4 land in step 2, S1 in step 3, S5 in step 4,
S8-S14 in step 5, S6 in step 6, S7 in step 7.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .orchestrator import GateFailure, RunContext, StageOutput, StageSpec

STUB = {"stub": True}

# Classes from PLAN.md §1.4. general-3p is deliberately absent: it is deferred
# until its legal line is written down, and a class that exists in code is a
# class that ships by accident.
CLIP_CLASSES = ("campaign", "general-own")

REJECTED_BRIEF_CATEGORIES = ("crypto", "gambling")

MIN_SOURCE_DURATION_S = 120.0
MIN_ALIGNED_WORD_COVERAGE = 0.97
MIN_SURVIVING_CANDIDATES = 5
FINAL_DURATION_TOLERANCE_S = 0.1


def _cfg(ctx: RunContext, key: str, default: Any = None) -> Any:
    return ctx.config.get(key, default)


# --------------------------------------------------------------------------- #
# S0-S1  ingest and brief
# --------------------------------------------------------------------------- #

def s0_ingest(ctx: RunContext) -> StageOutput:
    """Register the source and probe it with ffprobe."""
    clip_class = _cfg(ctx, "clip_class", "general-own")
    if clip_class not in CLIP_CLASSES:
        raise GateFailure(
            "S0",
            f"clip class {clip_class!r} is not offered. "
            f"Allowed: {', '.join(CLIP_CLASSES)}. "
            "general-3p stays deferred until PLAN.md §9 item 2 is answered.",
        )

    from .ffmpeg import FFmpegError, probe as ffprobe

    record = _cfg(ctx, "source_record") or {}
    try:
        probe = ffprobe(ctx.source)
    except FFmpegError as exc:
        raise GateFailure("S0", f"unreadable container: {exc}") from exc
    probe.update({k: v for k, v in record.items() if k != "path"})

    if not probe["has_video"] or not probe["has_audio"]:
        missing = "video" if not probe["has_video"] else "audio"
        raise GateFailure("S0", f"source has no {missing} stream.")
    floor = float(_cfg(ctx, "min_duration_s", MIN_SOURCE_DURATION_S))
    if probe["duration_s"] < floor:
        raise GateFailure(
            "S0",
            f"source is {probe['duration_s']:.0f}s; the floor is {floor:.0f}s.",
        )
    return {"source_hash": ctx.source_hash, "clip_class": clip_class, **probe}


def s1_brief(ctx: RunContext) -> StageOutput:
    """Campaign only: Haiku extracts the brief, you confirm it in the form."""
    if ctx.output("S0")["clip_class"] != "campaign":
        return {"applicable": False}

    # STUB: build step 3 replaces this with Haiku extraction + the confirm form.
    brief = dict(_cfg(ctx, "brief", {}) or {})
    category = str(brief.get("category", "")).lower()
    if category in REJECTED_BRIEF_CATEGORIES:
        raise GateFailure("S1", f"brief category {category!r} is rejected (PLAN.md §7.2).")
    if not brief.get("confirmed", False):
        raise GateFailure(
            "S1",
            "brief is not confirmed. S5 will not start on an unconfirmed brief.",
        )
    return {"applicable": True, "brief": brief, **STUB}


# --------------------------------------------------------------------------- #
# S2-S4  audio, transcript, speakers
# --------------------------------------------------------------------------- #

def s2_audio_extract(ctx: RunContext) -> StageOutput:
    """ffmpeg to 16 kHz mono WAV. No gate of its own."""
    from .audio import SAMPLE_RATE
    from .ffmpeg import FFmpegError, extract_audio

    out = ctx.workdir / "audio.wav"
    if not out.exists():
        try:
            extract_audio(ctx.source, out, sample_rate=SAMPLE_RATE)
        except FFmpegError as exc:
            raise GateFailure("S2", f"audio extraction failed: {exc}") from exc
    return {"audio_path": str(out), "sample_rate": SAMPLE_RATE,
            "size_bytes": out.stat().st_size}


def s3_transcribe(ctx: RunContext) -> StageOutput:
    """Transcribe and align with whispermlx (MLX backend)."""
    from . import transcribe as s3

    waveform = ctx.shared.get("waveform")
    if waveform is None:
        from .audio import load_wav

        waveform = load_wav(Path(ctx.output("S2")["audio_path"]))
        ctx.shared["waveform"] = waveform

    # Transcription always auto-detects. Forcing Whisper to `expected_language`
    # would make the gate below tautological: the detected language could never
    # differ from the one we told it to use, so a mislabelled or wrong-language
    # source would sail through.
    result = s3.run(
        waveform,
        model_name=_cfg(ctx, "whisper_model", s3.DEFAULT_MODEL),
        language=None,
        on_stage=ctx.shared.get("on_substage"),
    )
    ctx.shared["transcript"] = result
    out = {
        "words": [w.as_dict() for w in result.words],
        "aligned_coverage": result.coverage,
        "language": result.language,
        "word_count": len(result.words),
    }
    if out["aligned_coverage"] < MIN_ALIGNED_WORD_COVERAGE:
        raise GateFailure(
            "S3",
            f"aligned-word coverage {out['aligned_coverage']:.1%} is below "
            f"{MIN_ALIGNED_WORD_COVERAGE:.0%}.",
        )
    expected = _cfg(ctx, "expected_language")
    if expected and out["language"] != expected:
        raise GateFailure(
            "S3",
            f"detected language {out['language']!r} != expected {expected!r}.",
        )
    return out


def s4_diarize(ctx: RunContext) -> StageOutput:
    """pyannote community-1, exclusive mode; merge speaker labels onto words."""
    from . import diarize as s4

    try:
        s4.check_access()  # hard gate before any model download
    except s4.DiarizationAccessError as exc:
        raise GateFailure("S4", str(exc)) from exc

    waveform = ctx.shared.get("waveform")
    if waveform is None:
        from .audio import load_wav

        waveform = load_wav(Path(ctx.output("S2")["audio_path"]))
        ctx.shared["waveform"] = waveform

    result = s4.diarize(
        waveform,
        num_speakers=_cfg(ctx, "expected_speaker_count"),
        min_speakers=_cfg(ctx, "min_speakers"),
        max_speakers=_cfg(ctx, "max_speakers"),
    )
    transcript = ctx.shared.get("transcript")
    if transcript is not None:
        s4.assign_speakers(transcript.words, result)
        ctx.shared["samples"] = s4.sample_labels(transcript.words)
    out = {"speakers": len(result.speakers), **result.as_dict()}
    expected = _cfg(ctx, "expected_speaker_count")
    if expected is not None and out["speakers"] != expected:
        raise GateFailure(
            "S4",
            f"found {out['speakers']} speakers, you expected {expected}. "
            "Confirm the count before continuing.",
        )
    return out


# --------------------------------------------------------------------------- #
# S5  ranking
# --------------------------------------------------------------------------- #

def s5_rank(ctx: RunContext) -> StageOutput:
    """Sonnet ranks candidates under the brief's constraints."""
    if ctx.output("S0")["clip_class"] == "campaign" and not ctx.output("S1").get("brief"):
        raise GateFailure("S5", "no confirmed brief; ranking cannot apply its constraints.")

    # STUB: build step 4 adds the prompt, schema, snapping and filters.
    candidates = list(_cfg(ctx, "stub_candidates", []) or [])
    if _cfg(ctx, "stub_malformed_json", False):
        raise GateFailure("S5", "ranking returned malformed JSON after one retry.")
    if candidates and len(candidates) < MIN_SURVIVING_CANDIDATES:
        raise GateFailure(
            "S5",
            f"only {len(candidates)} candidates survived snapping and duration "
            f"filtering; the floor is {MIN_SURVIVING_CANDIDATES}. "
            "That means a bad source or a broken prompt.",
        )
    return {"candidates": candidates, **STUB}


# --------------------------------------------------------------------------- #
# S6-S7  faces and layout
# --------------------------------------------------------------------------- #

def s6_visual_analysis(ctx: RunContext) -> StageOutput:
    """scdet shots + Apple Vision faces/landmarks at ~5 fps, candidate windows only."""
    # STUB: build step 6. A candidate with no face track is not a run-stopper;
    # it is restricted to letterbox, per §2.1.
    return {"tracks": {}, "shots": [], "letterbox_only": [], **STUB}


def s7_attribution(ctx: RunContext) -> StageOutput:
    """Speaker attribution and layout planning."""
    # STUB: build step 7, and only if the business gate G1 passes.
    return {"plans": {}, "follow_crop_blocked": [], **STUB}


# --------------------------------------------------------------------------- #
# S8-S11  preview, review, commentary, compliance
# --------------------------------------------------------------------------- #

def s8_proxy_render(ctx: RunContext) -> StageOutput:
    """540x960 h264_videotoolbox previews, captions burned, all layouts."""
    return {"previews": {}, **STUB}


def s9_review(ctx: RunContext) -> StageOutput:
    """Human gate by definition: nothing proceeds without approvals."""
    approvals = list(_cfg(ctx, "approvals", []) or [])
    if not approvals:
        raise GateFailure("S9", "awaiting review; no clips approved yet.")
    return {"approved": approvals, **STUB}


def s10_commentary(ctx: RunContext) -> StageOutput:
    """Write / Auto-generate (Haiku) / None. Built when §1.4 (b) is un-deferred."""
    pending = [
        c for c in ctx.output("S9")["approved"]
        if isinstance(c, dict) and c.get("commentary_mode") == "auto"
        and not c.get("commentary_accepted")
    ]
    if pending:
        raise GateFailure(
            "S10",
            f"{len(pending)} clip(s) have an unaccepted auto-generated commentary draft. "
            "Nothing renders with unread text.",
        )
    return {"resolved": {}, **STUB}


def s11_compliance(ctx: RunContext) -> StageOutput:
    """Mechanical checks against the confirmed brief and the class rules."""
    # STUB: build step 8 implements one check per rule in §7.2.
    failures = list(_cfg(ctx, "stub_compliance_failures", []) or [])
    if failures:
        raise GateFailure("S11", f"compliance failures block export: {'; '.join(failures)}")
    return {"passed": True, **STUB}


# --------------------------------------------------------------------------- #
# S12-S14  render, export, track
# --------------------------------------------------------------------------- #

def s12_final_render(ctx: RunContext) -> StageOutput:
    """The only full-quality encode: one ffmpeg pass from the original source."""
    # STUB: build step 5 adds the filter_complex one-pass renderer.
    rendered = list(_cfg(ctx, "stub_rendered", []) or [])
    for clip in rendered:
        drift = abs(clip.get("actual_duration_s", 0) - clip.get("planned_duration_s", 0))
        if drift > FINAL_DURATION_TOLERANCE_S:
            raise GateFailure(
                "S12",
                f"clip {clip.get('id')} duration drifted {drift:.3f}s from plan "
                f"(tolerance {FINAL_DURATION_TOLERANCE_S}s).",
            )
        if not clip.get("has_audio", True):
            raise GateFailure("S12", f"clip {clip.get('id')} rendered without an audio stream.")
    return {"rendered": rendered, **STUB}


def s13_export_bundle(ctx: RunContext) -> StageOutput:
    """Per clip per platform: MP4, caption.txt, checklist.md."""
    return {"bundles": [], **STUB}


def s14_post_track(ctx: RunContext) -> StageOutput:
    """Manual post, then the tracking row. Disclosure is your attestation."""
    rows = list(_cfg(ctx, "post_rows", []) or [])
    is_campaign = ctx.output("S0")["clip_class"] == "campaign"
    for row in rows:
        if is_campaign and not row.get("disclosure_ticked"):
            raise GateFailure(
                "S14",
                f"campaign clip {row.get('clip_id')} cannot be marked submitted "
                "without the paid-promotion checkbox.",
            )
    return {"tracked": rows, **STUB}


STAGES: tuple[StageSpec, ...] = (
    StageSpec("S0", "ingest", "Register + probe the source", s0_ingest,
              params=("clip_class", "min_duration_s", "source_record"),
              gate="No video/audio stream; unreadable container; duration under 2 min"),
    StageSpec("S1", "brief", "Extract + confirm the campaign brief", s1_brief,
              needs=("S0",), params=("brief",),
              gate="Unconfirmed brief blocks S5; crypto/gambling category rejected"),
    StageSpec("S2", "audio-extract", "ffmpeg to 16 kHz mono WAV", s2_audio_extract,
              needs=("S0",), gate=""),
    StageSpec("S3", "transcribe", "whispermlx transcript + alignment", s3_transcribe,
              needs=("S2",), params=("expected_language", "whisper_model"),
              gate="Aligned-word coverage below 97%; language != expected"),
    StageSpec("S4", "diarize", "pyannote community-1 + word merge", s4_diarize,
              needs=("S3",), params=("expected_speaker_count", "min_speakers", "max_speakers"),
              gate="Speaker count differs from the count you gave"),
    StageSpec("S5", "rank", "Sonnet ranks candidates", s5_rank,
              needs=("S1", "S4"), params=("stub_candidates", "stub_malformed_json"),
              gate="Malformed JSON after one retry; fewer than 5 candidates survive"),
    StageSpec("S6", "visual-analysis", "scdet shots + Vision face tracks", s6_visual_analysis,
              needs=("S5",),
              gate="Candidate with no face track is restricted to letterbox"),
    StageSpec("S7", "attribution", "Speaker attribution + layout plans", s7_attribution,
              needs=("S6",),
              gate="Low attribution confidence blocks follow-crop for that clip"),
    StageSpec("S8", "proxy-render", "540x960 videotoolbox previews", s8_proxy_render,
              needs=("S7",), gate=""),
    StageSpec("S9", "review", "Human selection and trimming", s9_review,
              needs=("S8",), params=("approvals",), gate="Human gate by definition"),
    StageSpec("S10", "commentary", "Resolve commentary text", s10_commentary,
              needs=("S9",), gate="Unaccepted auto-generated text blocks final render"),
    StageSpec("S11", "compliance", "Brief + class rule checks", s11_compliance,
              needs=("S10",), params=("stub_compliance_failures",),
              gate="Any mechanical failure blocks the clip"),
    StageSpec("S12", "final-render", "One-pass 1080x1920 libx264 encode", s12_final_render,
              needs=("S11",), params=("stub_rendered",),
              gate="Duration drift > 0.1s; wrong resolution; no audio stream"),
    StageSpec("S13", "export-bundle", "MP4 + caption.txt + checklist.md", s13_export_bundle,
              needs=("S12",), gate=""),
    StageSpec("S14", "post-track", "Log the post URL and disclosure", s14_post_track,
              needs=("S13",), params=("post_rows",),
              gate="Campaign clip without the disclosure tick cannot be submitted"),
)

STAGES_BY_ID = {s.id: s for s in STAGES}
