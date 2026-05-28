from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    gemini_api_key: str
    azure_speech_key: str
    azure_speech_region: str = "eastasia"
    gemini_model: str = "gemini-2.0-flash"
    max_repo_files: int = 80
    max_tokens_per_chunk: int = 800_000
    azure_voice_name: str = "zh-CN-YunxiNeural"
    output_dir: str = "./output"
    clone_dir: str = field(default_factory=lambda: os.path.join("/tmp", "repocast_clones"))

    @classmethod
    def from_env(cls) -> Config:
        gemini_key = os.environ.get("GEMINI_API_KEY", "")
        azure_key = os.environ.get("AZURE_SPEECH_KEY", "")
        azure_region = os.environ.get("AZURE_SPEECH_REGION", "eastasia")

        missing = []
        if not gemini_key:
            missing.append("GEMINI_API_KEY")
        if not azure_key:
            missing.append("AZURE_SPEECH_KEY")

        if missing:
            raise EnvironmentError(
                f"缺少必要的环境变量: {', '.join(missing)}\n"
                "请设置后重试，或参考 .env.example 文件。"
            )

        return cls(
            gemini_api_key=gemini_key,
            azure_speech_key=azure_key,
            azure_speech_region=azure_region,
            gemini_model=os.environ.get("REPOCAST_GEMINI_MODEL", "gemini-2.0-flash"),
            max_repo_files=int(os.environ.get("REPOCAST_MAX_FILES", "80")),
            max_tokens_per_chunk=int(os.environ.get("REPOCAST_MAX_TOKENS_PER_CHUNK", "800000")),
            azure_voice_name=os.environ.get("REPOCAST_VOICE", "zh-CN-YunxiNeural"),
            output_dir=os.environ.get("REPOCAST_OUTPUT_DIR", "./output"),
            clone_dir=os.environ.get("REPOCAST_CLONE_DIR", "/tmp/repocast_clones"),
        )
