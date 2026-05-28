from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel

from repocast.config import Config
from repocast.fetcher import clone_repo, parse_github_url
from repocast.scriptgen import generate_script
from repocast.selector import select_files
from repocast.tts import synthesize_to_mp3
from repocast.utils import cleanup_clone, sanitize_filename, setup_logging

console = Console()
logger = logging.getLogger(__name__)


@click.command()
@click.argument("repo_url")
@click.option("-o", "--output", default=None, help="输出MP3文件路径")
@click.option("-n", "--max-files", default=80, help="最大分析文件数", show_default=True)
@click.option("--voice", default=None, help="Azure TTS语音名称")
@click.option("--branch", default=None, help="指定Git分支")
@click.option("--skip-tts", is_flag=True, help="仅生成脚本，跳过音频合成")
@click.option("--script-out", default=None, help="同时保存脚本到此路径")
@click.option("-v", "--verbose", is_flag=True, help="显示详细日志")
def main(
    repo_url: str,
    output: str | None,
    max_files: int,
    voice: str | None,
    branch: str | None,
    skip_tts: bool,
    script_out: str | None,
    verbose: bool,
) -> None:
    """将GitHub仓库转化为中文播客音频。

    REPO_URL: GitHub仓库地址 (如 https://github.com/owner/repo)
    """
    setup_logging(verbose)

    # Validate URL early
    try:
        owner, repo = parse_github_url(repo_url)
    except ValueError as e:
        console.print(f"[red]错误:[/red] {e}")
        sys.exit(1)

    # Load config
    try:
        config = Config.from_env()
    except EnvironmentError as e:
        console.print(f"[red]配置错误:[/red] {e}")
        sys.exit(1)

    # Override config with CLI options
    if voice:
        config.azure_voice_name = voice
    config.max_repo_files = max_files

    # Determine output path
    if output is None:
        output = f"./{sanitize_filename(repo)}.mp3"

    console.print(Panel(
        f"[bold]Repocast[/bold] - GitHub仓库播客生成器\n\n"
        f"仓库: {owner}/{repo}\n"
        f"输出: {output}",
        title="🎙️ Repocast",
        border_style="blue",
    ))

    repo_info = None
    try:
        # Step 1: Clone
        repo_info = clone_repo(repo_url, config, branch=branch)

        # Step 2: Select files
        files = select_files(
            repo_info.path,
            max_files=config.max_repo_files,
            token_budget=config.max_tokens_per_chunk,
        )
        if not files:
            console.print("[red]未找到可分析的代码文件[/red]")
            sys.exit(1)

        # Step 3: Generate script
        script = generate_script(files, repo_info.name, config)

        # Save script if requested (or if skip-tts)
        if script_out or skip_tts:
            script_path = script_out or output.replace(".mp3", ".txt")
            Path(script_path).parent.mkdir(parents=True, exist_ok=True)
            Path(script_path).write_text(script, encoding="utf-8")
            console.print(f"脚本已保存: [green]{script_path}[/green]")

        if skip_tts:
            console.print("[green]完成！[/green]（已跳过音频合成）")
            return

        # Step 4: Synthesize audio
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        synthesize_to_mp3(script, output, config)

        console.print(f"\n[green bold]完成！[/green bold] 播客音频已保存至: {output}")

    except KeyboardInterrupt:
        console.print("\n[yellow]已取消[/yellow]")
        sys.exit(130)
    except Exception as e:
        logger.debug("详细错误信息:", exc_info=True)
        console.print(f"\n[red]错误:[/red] {e}")
        sys.exit(1)
    finally:
        if repo_info:
            cleanup_clone(repo_info.path)
