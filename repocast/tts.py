from __future__ import annotations

import logging
import re

import azure.cognitiveservices.speech as speechsdk
from tenacity import retry, stop_after_attempt, wait_exponential

from repocast.config import Config

logger = logging.getLogger(__name__)

MAX_SSML_CHARS = 4000  # conservative limit per synthesis call


def _text_to_ssml(text: str, voice_name: str) -> str:
    """Convert a text segment to SSML with pauses and prosody."""
    # Escape XML special characters
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    text = text.replace('"', "&quot;")

    # Add pauses for section headers like 【开场】
    text = re.sub(
        r"【([^】]+)】",
        r'<break time="1000ms"/>【\1】<break time="600ms"/>',
        text,
    )

    # Add pauses for paragraph breaks
    text = re.sub(r"\n\n+", '\n<break time="600ms"/>\n', text)

    # Wrap English terms for better pronunciation
    text = re.sub(
        r"\b([A-Za-z][A-Za-z0-9_.]{2,})\b",
        r'<lang xml:lang="en-US">\1</lang>',
        text,
    )

    ssml = (
        '<speak version="1.0" xmlns="http://www.w3.org/2001/Math/MathML" '
        'xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="zh-CN">'
        f'<voice name="{voice_name}">'
        '<prosody rate="-5%">'
        f"{text}"
        "</prosody>"
        "</voice>"
        "</speak>"
    )
    return ssml


def _split_into_segments(script: str, max_chars: int = MAX_SSML_CHARS) -> list[str]:
    """Split script into segments at paragraph boundaries."""
    paragraphs = re.split(r"\n\n+", script)
    segments: list[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current) + len(para) + 2 > max_chars:
            if current:
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

        try:
            audio_data = _synthesize_segment(synthesizer, ssml)
            audio_parts.append(audio_data)
        except RuntimeError as e:
            logger.warning(f"片段 {i+1} 合成失败，跳过: {e}")
            continue

    if not audio_parts:
        raise RuntimeError("所有音频片段合成均失败")

    # Concatenate MP3 segments (same bitrate = safe concatenation)
    with open(output_path, "wb") as f:
        for part in audio_parts:
            f.write(part)

    total_size = sum(len(p) for p in audio_parts)
    logger.info(
        f"音频合成完成: {output_path} "
        f"({total_size / 1024 / 1024:.1f} MB)"
    )
