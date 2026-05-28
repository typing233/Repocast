from __future__ import annotations

import logging
import re
import xml.sax.saxutils as saxutils

import azure.cognitiveservices.speech as speechsdk
from tenacity import retry, stop_after_attempt, wait_exponential

from repocast.config import Config

logger = logging.getLogger(__name__)

MAX_SEGMENT_CHARS = 4000


def _escape_xml(text: str) -> str:
    return saxutils.escape(text, {'"': "&quot;"})


def _process_paragraph(paragraph: str) -> str:
    """Process a single paragraph: escape XML, then wrap English terms in <lang> tags."""
    # First escape all XML-special characters
    escaped = _escape_xml(paragraph)

    # Wrap standalone English terms (3+ chars, letters/digits/dot/underscore/hyphen)
    # Use a regex that won't match inside XML entities (which start with &)
    def wrap_english(m: re.Match) -> str:
        term = m.group(0)
        return f'<lang xml:lang="en-US">{term}</lang>'

    # Match English words: start with letter, followed by letters/digits/dots/underscores/hyphens
    # Negative lookbehind for & to avoid matching XML entity internals like "amp" in "&amp;"
    processed = re.sub(
        r"(?<!&)(?<![A-Za-z])[A-Za-z][A-Za-z0-9_.\-]{2,}(?![A-Za-z;])",
        wrap_english,
        escaped,
    )
    return processed


def _text_to_ssml(text: str, voice_name: str) -> str:
    """Convert a text segment to valid SSML with pauses and prosody."""
    # Split into paragraphs for structured processing
    lines = text.split("\n")
    ssml_body_parts: list[str] = []
    pending_break = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            pending_break = True
            continue

        # Check if this is a section header like 【开场】
        header_match = re.match(r"^【([^】]+)】(.*)$", stripped)

        if pending_break and ssml_body_parts:
            ssml_body_parts.append('<break time="600ms"/>')
            pending_break = False

        if header_match:
            header_title = header_match.group(1)
            rest = header_match.group(2).strip()
            ssml_body_parts.append('<break time="1000ms"/>')
            ssml_body_parts.append(_process_paragraph(f"【{header_title}】"))
            ssml_body_parts.append('<break time="600ms"/>')
            if rest:
                ssml_body_parts.append(_process_paragraph(rest))
        else:
            ssml_body_parts.append(_process_paragraph(stripped))

    body = "\n".join(ssml_body_parts)

    ssml = (
        '<speak version="1.0" '
        'xmlns="http://www.w3.org/2001/10/synthesis" '
        'xmlns:mstts="http://www.w3.org/2001/mstts" '
        'xml:lang="zh-CN">'
        f'<voice name="{voice_name}">'
        '<prosody rate="-5%">'
        f"{body}"
        "</prosody>"
        "</voice>"
        "</speak>"
    )
    return ssml


def _split_into_segments(script: str, max_chars: int = MAX_SEGMENT_CHARS) -> list[str]:
    """Split script into segments at paragraph boundaries."""
    paragraphs = re.split(r"\n\n+", script)
    segments: list[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current) + len(para) + 2 > max_chars and current:
            segments.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para

    if current:
        segments.append(current)

    return segments


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=5, max=30),
    before_sleep=lambda retry_state: logging.getLogger(__name__).warning(
        f"片段合成失败，{retry_state.next_action.sleep:.0f}s后重试 "
        f"(第{retry_state.attempt_number}次): {retry_state.outcome.exception()}"
    ),
)
def _synthesize_segment(
    synthesizer: speechsdk.SpeechSynthesizer, ssml: str
) -> bytes:
    result = synthesizer.speak_ssml_async(ssml).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        return result.audio_data
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation = result.cancellation_details
        if cancellation.reason == speechsdk.CancellationReason.Error:
            raise RuntimeError(
                f"TTS合成错误: {cancellation.error_details}\n"
                "请检查Azure Speech密钥和区域配置。"
            )
        raise RuntimeError(f"TTS合成被取消: {cancellation.reason}")
    else:
        raise RuntimeError(f"TTS合成失败: {result.reason}")


def synthesize_to_mp3(
    script: str,
    output_path: str,
    config: Config,
) -> None:
    speech_config = speechsdk.SpeechConfig(
        subscription=config.azure_speech_key,
        region=config.azure_speech_region,
    )
    speech_config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Audio16Khz128KBitRateMonoMp3
    )

    synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config,
        audio_config=None,
    )

    segments = _split_into_segments(script)
    logger.info(f"脚本已分为 {len(segments)} 个片段进行合成")

    audio_parts: list[bytes] = []

    for i, segment in enumerate(segments):
        logger.info(f"合成片段 {i+1}/{len(segments)} ...")
        ssml = _text_to_ssml(segment, config.azure_voice_name)
        audio_data = _synthesize_segment(synthesizer, ssml)
        audio_parts.append(audio_data)

    with open(output_path, "wb") as f:
        for part in audio_parts:
            f.write(part)

    total_size = sum(len(p) for p in audio_parts)
    logger.info(
        f"音频合成完成: {output_path} "
        f"({total_size / 1024 / 1024:.1f} MB)"
    )
