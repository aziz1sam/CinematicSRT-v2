"""Subtitle Formatting, Timing Sanitization, and Export Engine (.srt, .ass, .vtt)"""
import re
from types import SimpleNamespace
from typing import List, Optional, Tuple, Dict, Any

def make_subtitle_document(
    segments: Optional[List[SimpleNamespace]] = None,
    rejected_blocks: Optional[List[Dict[str, Any]]] = None,
    doc_metadata: Optional[Dict[str, Any]] = None,
    raw_header: Optional[str] = None,
    custom_styles: Optional[List[str]] = None
) -> List[SimpleNamespace]:
    """Factory creating a list-subclass document preserving metadata and diagnostics."""
    class SubtitleDocument(list):
        def __init__(self, segs, rejected, meta, header, styles):
            super().__init__(segs or [])
            self.rejected_blocks = list(rejected or [])
            self.doc_metadata = dict(meta or {})
            self.raw_header = header or ""
            self.custom_styles = list(styles or [])
    return SubtitleDocument(segments, rejected_blocks, doc_metadata, raw_header, custom_styles)

def format_timestamp_srt(seconds: float) -> str:
    """Canonical integer-millisecond timestamp: HH:MM:SS,mmm.
    Uses divmod arithmetic to eliminate the 60-second rollover bug.
    """
    total_ms = max(0, int(round(float(seconds or 0.0) * 1000)))
    total_sec, ms = divmod(total_ms, 1000)
    hours, remainder = divmod(total_sec, 3600)
    minutes, sec = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d},{ms:03d}"

def format_timestamp_ass(seconds: float) -> str:
    """Canonical centisecond timestamp for ASS subtitles: H:MM:SS.cc"""
    total_cs = max(0, int(round(float(seconds or 0.0) * 100)))
    total_sec, cs = divmod(total_cs, 100)
    hours, remainder = divmod(total_sec, 3600)
    minutes, sec = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{sec:02d}.{cs:02d}"

def format_timestamp_vtt(seconds: float) -> str:
    """Canonical millisecond timestamp for WebVTT subtitles: HH:MM:SS.mmm"""
    total_ms = max(0, int(round(float(seconds or 0.0) * 1000)))
    total_sec, ms = divmod(total_ms, 1000)
    hours, remainder = divmod(total_sec, 3600)
    minutes, sec = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}.{ms:03d}"

def wrap_arabic_text(text: str, max_chars: int = 42, max_lines: Optional[int] = None) -> List[str]:
    """Wraps Arabic subtitle text enforcing strict max_chars per line while preserving 100% of content.
    - Zero words or characters are ever discarded.
    - Long unbroken tokens exceeding max_chars are split into chunks of max_chars preserving remainder.
    - If max_chars <= 0, raises ValueError.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be a positive integer")

    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if not cleaned:
        return []

    # Break any single unbroken token that exceeds max_chars, keeping full text
    raw_tokens = cleaned.split(" ")
    tokens = []
    for t in raw_tokens:
        while len(t) > max_chars:
            tokens.append(t[:max_chars])
            t = t[max_chars:]
        if t:
            tokens.append(t)

    # Accumulate tokens into lines strictly respecting max_chars
    lines = []
    current_line = []
    current_len = 0

    for token in tokens:
        token_len = len(token)
        needed = token_len if current_len == 0 else current_len + 1 + token_len
        if needed <= max_chars:
            current_line.append(token)
            current_len = needed
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [token]
            current_len = token_len

    if current_line:
        lines.append(" ".join(current_line))

    # All text is preserved (never truncate lines)
    return lines

def sanitize_timing_without_drift(
    segments: List[SimpleNamespace],
    min_duration: float = 0.8,
    min_gap: float = 0.05,
    max_duration: float = 7.0,
    max_chars: int = 42,
    max_lines: int = 2,
    media_end: Optional[float] = None
) -> List[SimpleNamespace]:
    """Canonical non-drifting bounded timing sanitizer with full QC warning propagation.
    - Start invariance: source starts are immutable.
    - Full content conservation: no text or cues are dropped.
    - True source overlaps are preserved with is_overlap=True and SOURCE_OVERLAP_REVIEW.
    - Overlong speech is preserved with REQUIRES_SOURCE_AWARE_SPLIT.
    - Cues with more lines than max_lines get LAYOUT_OVERFLOW_WARNING.
    - Upstream provider warnings (e.g. PROVIDER_MISSING_TRANSLATION) are strictly preserved.
    """
    if not segments:
        return []

    # Filter out empty or whitespace-only cues first so lookahead is clean
    valid_cues = []
    for seg in segments:
        text = (getattr(seg, "text", "") or "").strip()
        if not text:
            continue

        s_start = float(getattr(seg, "start", 0.0))
        s_end = float(getattr(seg, "end", s_start + 0.1))

        import math
        if math.isnan(s_start) or math.isinf(s_start) or math.isnan(s_end) or math.isinf(s_end):
            raise ValueError(f"Non-finite timestamp: start={s_start}, end={s_end}")
        if s_start < 0.0:
            raise ValueError(f"Invalid negative start time: {s_start}")
        if s_end <= s_start:
            raise ValueError(f"Invalid interval: start={s_start}, end={s_end}")

        cue_warnings = list(getattr(seg, "warnings", ()))
        if media_end is not None:
            if s_start >= media_end:
                raise ValueError(f"Cue start ({s_start}) exceeds media_end ({media_end})")
            if s_end > media_end:
                cue_warnings.append("CUE_TRUNCATED_AT_MEDIA_END")
                s_end = media_end

        valid_cues.append(SimpleNamespace(
            index=getattr(seg, "index", len(valid_cues) + 1),
            start=s_start,
            end=s_end,
            text=text,
            speaker_id=getattr(seg, "speaker_id", None),
            words=getattr(seg, "words", None),
            warnings=tuple(cue_warnings)
        ))

    if not valid_cues:
        return []

    # Sort cues explicitly by start time
    valid_cues.sort(key=lambda c: c.start)
    n = len(valid_cues)

    # Detect true source overlaps
    is_overlap = [False] * n
    active = []
    for i, c in enumerate(valid_cues):
        active = [j for j in active if valid_cues[j].end > c.start]
        for j in active:
            is_overlap[j] = is_overlap[i] = True
        active.append(i)

    sanitized = []
    for i in range(n):
        curr = valid_cues[i]
        src_start = curr.start
        src_end = curr.end
        duration = src_end - src_start
        warnings = list(curr.warnings)

        disp_start = src_start
        disp_end = src_end

        # Independent warning evaluations
        if is_overlap[i]:
            warnings.append("SOURCE_OVERLAP_REVIEW")
            disp_end = src_end
        if duration > max_duration:
            warnings.append("REQUIRES_SOURCE_AWARE_SPLIT")
            disp_end = src_end

        # Timing layout for non-overlapping, normal duration cues
        if not is_overlap[i] and duration <= max_duration:
            upper = disp_start + max_duration
            if i + 1 < n:
                upper = min(upper, valid_cues[i + 1].start - min_gap)
            if media_end is not None:
                upper = min(upper, media_end)

            if upper < src_end:
                warnings.append("GAP_UNSATISFIABLE_WITHOUT_CLIPPING_SPEECH")
                disp_end = src_end
            else:
                disp_end = min(max(src_end, disp_start + min_duration), upper)

        if disp_end - disp_start < min_duration:
            warnings.append("SUB_MINIMUM_DURATION_INTERVAL")

        # Wrap text and check for layout overflow
        wrapped = wrap_arabic_text(curr.text, max_chars=max_chars)
        if len(wrapped) > max_lines:
            warnings.append("LAYOUT_OVERFLOW_WARNING")

        out_cue = SimpleNamespace(
            index=curr.index,
            start=disp_start,
            end=disp_end,
            text=curr.text,
            lines=wrapped,
            is_overlap=is_overlap[i],
            warnings=tuple(warnings)
        )
        if curr.speaker_id is not None:
            out_cue.speaker_id = curr.speaker_id
        if curr.words is not None:
            out_cue.words = curr.words

        sanitized.append(out_cue)

    return sanitized

def build_srt_content(segments: List[SimpleNamespace]) -> str:
    """Generates standard SRT file string preserving all wrapped lines."""
    blocks = []
    for i, seg in enumerate(segments, start=1):
        lines = getattr(seg, "lines", None) or wrap_arabic_text(seg.text)
        text_block = "\n".join(lines)
        start_tc = format_timestamp_srt(seg.start)
        end_tc = format_timestamp_srt(seg.end)
        blocks.append(f"{i}\n{start_tc} --> {end_tc}\n{text_block}\n")
    return "\n".join(blocks)

def build_ass_content(segments: List[SimpleNamespace], title: str = "CinematicSRT") -> str:
    """Generates styled Advanced SubStation Alpha (.ass) subtitle file, preserving custom styles and layers."""
    passed_styles = getattr(segments, "custom_styles", None) or []
    styles_dict = {}
    cinematic_style = "Style: Cinematic,Cairo,52,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,2.5,1.5,2,40,40,45,1"
    styles_dict["Cinematic"] = cinematic_style

    for st_line in passed_styles:
        if ":" in st_line:
            s_name = st_line.split(":", 1)[1].split(",", 1)[0].strip()
            styles_dict[s_name] = st_line

    for seg in segments:
        s_name = getattr(seg, "style", None)
        if s_name and s_name not in styles_dict:
            styles_dict[s_name] = f"Style: {s_name},Cairo,52,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,2.5,1.5,2,40,40,45,1"

    styles_block = "\n".join(styles_dict.values())

    raw_hdr = getattr(segments, "raw_header", None)
    if raw_hdr and "[Script Info]" in raw_hdr:
        script_info_section = raw_hdr.split("[V4+ Styles]")[0].strip()
    else:
        script_info_section = f"""[Script Info]
Title: {title} - Arabic Subtitles
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709
PlayResX: 1920
PlayResY: 1080"""

    header = f"""{script_info_section}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
{styles_block}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    for seg in segments:
        raw_ass = getattr(seg, "raw_ass_text", None)
        clean = getattr(seg, "clean_text", None)

        if raw_ass and (clean is None or seg.text == clean):
            formatted_ass = raw_ass
        else:
            lines = getattr(seg, "lines", None) or wrap_arabic_text(seg.text)
            text_ass = r"\N".join(lines)
            if raw_ass:
                # Without translated span alignment, retain the original ASS
                # event rather than guessing new formatting scopes. seg.text
                # remains translated for the SRT/VTT exporters.
                has_protected_tokens = (
                    re.search(r"\{.*?\}", raw_ass) is not None
                    or r"\N" in raw_ass
                    or r"\n" in raw_ass
                )
                if has_protected_tokens:
                    formatted_ass = raw_ass
                    warns = list(getattr(seg, "warnings", ()))
                    for warning in (
                        "ASS_SOURCE_EVENT_RETAINED_FOR_REVIEW",
                        "ASS_INLINE_SCOPE_TRANSFORMATION_REVIEW",
                    ):
                        if warning not in warns:
                            warns.append(warning)
                    seg.warnings = tuple(warns)
                else:
                    formatted_ass = text_ass
            else:
                ass_tags = getattr(seg, "ass_tags", "") or ""
                formatted_ass = f"{ass_tags}{text_ass}"

        start_tc = format_timestamp_ass(seg.start)
        end_tc = format_timestamp_ass(seg.end)
        speaker = getattr(seg, "speaker_id", "") or getattr(seg, "name", "") or ""
        layer = getattr(seg, "layer", 0)
        style = getattr(seg, "style", "Cinematic") or "Cinematic"
        ml = getattr(seg, "margin_l", 0)
        mr = getattr(seg, "margin_r", 0)
        mv = getattr(seg, "margin_v", 0)
        effect = getattr(seg, "effect", "")
        events.append(f"Dialogue: {layer},{start_tc},{end_tc},{style},{speaker},{ml},{mr},{mv},{effect},{formatted_ass}")

    return header + "\n".join(events) + "\n"

def build_vtt_content(segments: List[SimpleNamespace]) -> str:
    """Generates standard WebVTT (.vtt) format for HTML5 video players."""
    blocks = ["WEBVTT\n"]
    for i, seg in enumerate(segments, start=1):
        lines = getattr(seg, "lines", None) or wrap_arabic_text(seg.text)
        text_block = "\n".join(lines)
        start_tc = format_timestamp_vtt(seg.start)
        end_tc = format_timestamp_vtt(seg.end)
        blocks.append(f"{i}\n{start_tc} --> {end_tc}\n{text_block}\n")
    return "\n".join(blocks)

def validate_imported_cues(segments: List[SimpleNamespace]) -> List[SimpleNamespace]:
    """Validates imported subtitle cues without applying retiming or padding.
    Propagates parser rejection diagnostics and checks finite timestamps.
    """
    if not segments:
        raise ValueError("No subtitle cues found in file")

    rejected = getattr(segments, "rejected_blocks", [])
    doc_meta = getattr(segments, "doc_metadata", {})
    raw_hdr = getattr(segments, "raw_header", "")
    c_styles = getattr(segments, "custom_styles", [])

    validated = []
    for seg in segments:
        s_start = float(getattr(seg, "start", 0.0))
        s_end = float(getattr(seg, "end", 0.0))
        import math
        if math.isnan(s_start) or math.isinf(s_start) or math.isnan(s_end) or math.isinf(s_end):
            raise ValueError(f"Non-finite timestamp in cue: start={s_start}, end={s_end}")
        if s_start < 0.0:
            raise ValueError(f"Negative start timestamp in cue: {s_start}")
        if s_end <= s_start:
            raise ValueError(f"Invalid reversed or zero-duration interval: start={s_start}, end={s_end}")

        if rejected:
            cue_warns = list(getattr(seg, "warnings", ()))
            for r in rejected:
                reason = r.get("reason", "MALFORMED_SOURCE_BLOCK_SKIPPED")
                if reason not in cue_warns:
                    cue_warns.append(reason)
            seg.warnings = tuple(cue_warns)

        validated.append(seg)

    return make_subtitle_document(
        segments=validated,
        rejected_blocks=rejected,
        doc_metadata=doc_meta,
        raw_header=raw_hdr,
        custom_styles=c_styles
    )

def parse_srt_content(content: str) -> List[SimpleNamespace]:
    """Parses SRT format string into standard SimpleNamespace segment objects preserving diagnostics."""
    # Locate blocks before trimming so source diagnostics retain physical line
    # positions across leading, whitespace-only and variable blank separators.
    blocks = []
    block_lines = []
    block_start_line = 1
    for line_number, raw_line in enumerate(content.splitlines(keepends=True), start=1):
        if raw_line.strip():
            if not block_lines:
                block_start_line = line_number
            block_lines.append(raw_line)
        elif block_lines:
            blocks.append((block_start_line, "".join(block_lines).rstrip("\r\n")))
            block_lines = []
    if block_lines:
        blocks.append((block_start_line, "".join(block_lines).rstrip("\r\n")))
    segments = []
    rejected_blocks = []

    for block_idx, (block_start_line, b) in enumerate(blocks, start=1):
        lines = [line.strip() for line in b.splitlines() if line.strip()]

        if not lines:
            continue
        time_line_idx = 1 if (len(lines) >= 2 and re.match(r"^\d+$", lines[0])) else 0
        if time_line_idx >= len(lines):
            rejected_blocks.append({
                "block_index": block_idx,
                "line_number": block_start_line,
                "raw_text": b,
                "reason": "MALFORMED_SRT_BLOCK_STRUCTURE"
            })
            continue

        time_line = lines[time_line_idx]
        text_lines = lines[time_line_idx + 1:]
        m = re.match(r"(\d+):(\d+):(\d+)[,\.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,\.](\d+)", time_line)
        if not m:
            rejected_blocks.append({
                "block_index": block_idx,
                "line_number": block_start_line,
                "raw_text": b,
                "reason": "MALFORMED_TIMESTAMP_FORMAT"
            })
            continue

        s_h, s_m, s_s, s_ms, e_h, e_m, e_s, e_ms = map(int, m.groups())
        start = s_h * 3600 + s_m * 60 + s_s + s_ms / 1000.0
        end = e_h * 3600 + e_m * 60 + e_s + e_ms / 1000.0
        text = " ".join(text_lines)
        segments.append(SimpleNamespace(
            index=len(segments) + 1,
            start=start,
            end=end,
            text=text
        ))

    return make_subtitle_document(segments=segments, rejected_blocks=rejected_blocks)

def parse_ass_content(content: str) -> List[SimpleNamespace]:
    """Parses ASS format string preserving styles, script info, layers, margins, and inline tags."""
    segments = []
    custom_styles = []
    rejected_blocks = []
    raw_header_lines = []
    in_events = False

    for line_no, raw_line in enumerate(content.splitlines(), start=1):
        line = raw_line.strip()
        if not in_events:
            if line.startswith("[Events]"):
                in_events = True
            else:
                raw_header_lines.append(raw_line)
                if line.startswith("Style:"):
                    custom_styles.append(line)
            continue

        if line.startswith("Dialogue:"):
            parts = line.split(",", 9)
            if len(parts) == 10:
                raw_layer = parts[0].split(":", 1)[1].strip() if ":" in parts[0] else "0"
                layer = int(raw_layer) if raw_layer.isdigit() else 0
                start_str = parts[1].strip()
                end_str = parts[2].strip()
                style_str = parts[3].strip()
                speaker = parts[4].strip()
                ml = int(parts[5].strip()) if parts[5].strip().isdigit() else 0
                mr = int(parts[6].strip()) if parts[6].strip().isdigit() else 0
                mv = int(parts[7].strip()) if parts[7].strip().isdigit() else 0
                effect = parts[8].strip()
                text = parts[9].strip()

                def _tc(s):
                    h, m, sec = s.split(":")
                    return int(h) * 3600 + int(m) * 60 + float(sec)

                try:
                    start = _tc(start_str)
                    end = _tc(end_str)
                    tag_m = re.match(r"^(\{.*?\})(.*)$", text)
                    ass_tags = tag_m.group(1) if tag_m else ""
                    clean_text = re.sub(r"\{.*?\}", "", text).replace(r"\N", " ").strip()
                    segments.append(SimpleNamespace(
                        index=len(segments) + 1,
                        start=start,
                        end=end,
                        text=clean_text,
                        speaker_id=speaker if speaker else None,
                        layer=layer,
                        style=style_str,
                        margin_l=ml,
                        margin_r=mr,
                        margin_v=mv,
                        effect=effect,
                        ass_tags=ass_tags,
                        raw_ass_text=text,
                        clean_text=clean_text
                    ))
                except Exception as e:
                    rejected_blocks.append({
                        "line_number": line_no,
                        "raw_text": line,
                        "reason": f"MALFORMED_ASS_TIMESTAMP: {e}"
                    })
            else:
                rejected_blocks.append({
                    "line_number": line_no,
                    "raw_text": line,
                    "reason": "MALFORMED_ASS_DIALOGUE_PARTS"
                })

    return make_subtitle_document(
        segments=segments,
        rejected_blocks=rejected_blocks,
        custom_styles=custom_styles,
        raw_header="\n".join(raw_header_lines)
    )

def parse_vtt_content(content: str) -> List[SimpleNamespace]:
    """Parses WebVTT format string into standard SimpleNamespace segment objects.
    Supports both HH:MM:SS.mmm and MM:SS.mmm, ignoring NOTE blocks and capturing diagnostics.
    """
    blocks = re.split(r"\n\s*\n", content.strip())
    segments = []
    rejected_blocks = []

    for block_idx, b in enumerate(blocks, start=1):
        lines = [line.strip() for line in b.splitlines() if line.strip()]
        if not lines:
            continue
        if lines[0].startswith("WEBVTT") or re.match(r"^NOTE(\s|$)", lines[0]):
            continue
        time_line_idx = 0
        if "-->" not in lines[0] and len(lines) > 1 and "-->" in lines[1]:
            time_line_idx = 1
        if time_line_idx >= len(lines) or "-->" not in lines[time_line_idx]:
            rejected_blocks.append({
                "block_index": block_idx,
                "raw_text": b,
                "reason": "MALFORMED_VTT_BLOCK"
            })
            continue

        time_line = lines[time_line_idx]
        m = re.search(r"((?:(\d+):)?(\d+):(\d+)\.(\d+))\s*-->\s*((?:(\d+):)?(\d+):(\d+)\.(\d+))", time_line)
        if not m:
            rejected_blocks.append({
                "block_index": block_idx,
                "raw_text": b,
                "reason": "MALFORMED_VTT_TIMESTAMP"
            })
            continue

        def _parse_vtt_tc(h_val, m_val, s_val, ms_val):
            h = int(h_val) if h_val else 0
            m_int = int(m_val)
            s_int = int(s_val)
            ms_int = int(ms_val)
            return h * 3600 + m_int * 60 + s_int + ms_int / 1000.0

        s_start = _parse_vtt_tc(m.group(2), m.group(3), m.group(4), m.group(5))
        s_end = _parse_vtt_tc(m.group(7), m.group(8), m.group(9), m.group(10))
        text_lines = lines[time_line_idx + 1:]
        clean_text = " ".join(text_lines).strip()
        clean_text = re.sub(r"<[^>]+>", "", clean_text).strip()
        if clean_text:
            segments.append(SimpleNamespace(
                index=len(segments) + 1,
                start=s_start,
                end=s_end,
                text=clean_text
            ))
        else:
            rejected_blocks.append({
                "block_index": block_idx,
                "raw_text": b,
                "reason": "EMPTY_VTT_TEXT"
            })

    return make_subtitle_document(segments=segments, rejected_blocks=rejected_blocks)
