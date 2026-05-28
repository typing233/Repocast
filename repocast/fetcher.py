from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from repocast.config import Config

logger = logging.getLogger(__name__)

GITHUB_URL_PATTERN = re.compile(
    r"https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?/?$"
)


@dataclass
class RepoInfo:
    path: Path
    name: str
    owner: str


def parse_github_url(url: str) -> tuple[str, str]:
    m = GITHUB_URL_PATTERN.match(url)
    if not m:
        raise ValueError(
            f"无效的GitHub URL: {url}\n"
            "期望格式: https://github.com/owner/repo"
        )
    return m.group("owner"), m.group("repo")


def clone_repo(url: str, config: Config, branch: str | None = None) -> RepoInfo:
    owner, repo = parse_github_url(url)
    clone_dir = Path(config.clone_dir)
    clone_dir.mkdir(parents=True, exist_ok=True)

    dest = Path(tempfile.mkdtemp(prefix=f"{repo}_", dir=clone_dir))

    cmd = ["git", "clone", "--depth", "1", "--single-branch"]
    if branch:
        cmd.extend(["--branch", branch])
    cmd.extend([url, str(dest)])

    logger.info(f"正在克隆仓库 [bold]{owner}/{repo}[/bold] ...")

    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"克隆超时（120s），请检查网络连接或仓库大小")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"克隆失败: {e.stderr.strip()}\n"
            "请确认URL正确且仓库为公开仓库。"
        )

    logger.info("克隆完成")
    return RepoInfo(path=dest, name=repo, owner=owner)
