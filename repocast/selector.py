from __future__ import annotations

import logging
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from repocast.utils import count_tokens

logger = logging.getLogger(__name__)

SKIP_DIRS = {
    ".git", "node_modules", "vendor", "__pycache__", ".venv", "venv",
    "dist", "build", ".next", "target", ".tox", "coverage", ".idea",
    ".vscode", ".mypy_cache", ".pytest_cache", "env", ".eggs",
    "site-packages", ".gradle", "Pods", ".dart_tool",
}

BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".bmp", ".svg", ".webp",
    ".woff", ".woff2", ".ttf", ".otf", ".eot",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ".exe", ".dll", ".so", ".dylib", ".wasm",
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
    ".mp3", ".mp4", ".wav", ".avi", ".mov",
    ".pyc", ".pyo", ".class", ".o", ".a",
    ".db", ".sqlite", ".sqlite3",
    ".min.js", ".min.css",
}

LOCK_FILES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "Cargo.lock", "poetry.lock", "Gemfile.lock",
    "composer.lock", "Pipfile.lock",
}

LANGUAGE_EXTENSIONS = {
    ".py": "python", ".pyx": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java", ".kt": "kotlin",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp",
    ".cs": "csharp",
    ".swift": "swift",
    ".dart": "dart",
    ".lua": "lua",
    ".zig": "zig",
    ".ex": "elixir", ".exs": "elixir",
    ".scala": "scala",
    ".clj": "clojure",
    ".hs": "haskell",
    ".ml": "ocaml",
    ".vue": "vue",
    ".svelte": "svelte",
}

MANIFEST_FILES = {
    "package.json", "pyproject.toml", "setup.py", "setup.cfg",
    "Cargo.toml", "go.mod", "build.gradle", "build.gradle.kts",
    "pom.xml", "Gemfile", "composer.json", "pubspec.yaml",
    "CMakeLists.txt", "Makefile", "mix.exs", "dune-project",
}

ENTRY_POINT_PATTERNS = [
    "main.py", "app.py", "index.ts", "index.js", "main.go",
    "main.rs", "Main.java", "Program.cs", "main.dart",
    "server.py", "server.ts", "server.js",
]

MAX_FILE_SIZE = 100 * 1024  # 100KB


@dataclass
class SelectedFile:
    path: str
    content: str
    category: str
    score: int


def detect_primary_language(files: list[Path], root: Path) -> str | None:
    ext_counter: Counter[str] = Counter()
    for f in files:
        ext = f.suffix.lower()
        if ext in LANGUAGE_EXTENSIONS:
            ext_counter[ext] += 1

    if not ext_counter:
        return None

    top_ext = ext_counter.most_common(1)[0][0]
    return LANGUAGE_EXTENSIONS[top_ext]


def score_file(file_path: Path, root: Path, primary_lang: str | None) -> tuple[int, str]:
    rel = file_path.relative_to(root)
    name = file_path.name.lower()
    ext = file_path.suffix.lower()
    parts = rel.parts
    rel_str = str(rel)

    # README
    if name.startswith("readme"):
        return 100, "readme"

    # Manifest/package config
    if name in {f.lower() for f in MANIFEST_FILES}:
        return 90, "manifest"

    # Entry points
    if name in {f.lower() for f in ENTRY_POINT_PATTERNS}:
        return 85, "entry_point"
    if len(parts) >= 2 and parts[0] in ("src", "cmd", "app", "lib") and name in {
        "main.py", "main.go", "main.rs", "index.ts", "index.js", "app.py",
        "mod.rs", "lib.rs",
    }:
        return 85, "entry_point"

    # CI/Deploy configs
    if name in ("dockerfile", "docker-compose.yml", "docker-compose.yaml"):
        return 40, "devops"
    if ".github/workflows" in rel_str:
        return 40, "devops"

    # Documentation
    if ext in (".md", ".rst") and name != "changelog.md":
        if "doc" in rel_str.lower() or name in (
            "contributing.md", "architecture.md", "design.md"
        ):
            return 45, "docs"
        return 35, "docs"

    # Config/schema files
    if name in (".env.example", "config.yaml", "config.yml", "config.json"):
        return 50, "config"
    if ext in (".yaml", ".yml", ".toml", ".ini", ".cfg") and "config" in name:
        return 50, "config"
    if "schema" in name:
        return 50, "config"

    # Test files
    if "test" in name or "spec" in name or "tests/" in rel_str or "test/" in rel_str:
        return 30, "test"

    # Source code
    if ext in LANGUAGE_EXTENSIONS:
        score = 60
        lang = LANGUAGE_EXTENSIONS[ext]

        # Primary language boost
        if primary_lang and lang == primary_lang:
            score += 20

        # Location heuristics
        if len(parts) >= 2 and parts[0] in ("src", "lib", "app", "pkg", "internal"):
            score += 15
        if len(parts) <= 2:
            score += 10

        # Name heuristics
        important_names = {"handler", "router", "service", "controller", "model",
                          "schema", "api", "core", "engine", "client"}
        stem = file_path.stem.lower()
        if any(n in stem for n in important_names):
            score += 5

        # Depth penalty
        if len(parts) > 4:
            score -= 10

        # Generated file penalty
        if any(g in name for g in ("generated", "_pb2", ".gen.", "mock_")):
            score -= 15

        return score, "source"

    return 10, "other"


def select_files(
    root: Path,
    max_files: int = 80,
    token_budget: int = 800_000,
) -> list[SelectedFile]:
    logger.info("正在分析仓库文件结构...")

    all_files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune skipped directories
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

        for fname in filenames:
            fpath = Path(dirpath) / fname
            ext = fpath.suffix.lower()

            if fname in LOCK_FILES:
                continue
            if ext in BINARY_EXTENSIONS:
                continue
            if fname.endswith((".min.js", ".min.css")):
                continue

            try:
                size = fpath.stat().st_size
            except OSError:
                continue
            if size > MAX_FILE_SIZE or size == 0:
                continue

            all_files.append(fpath)

    logger.info(f"发现 {len(all_files)} 个候选文件")

    primary_lang = detect_primary_language(all_files, root)
    if primary_lang:
        logger.info(f"检测到主要语言: [bold]{primary_lang}[/bold]")

    # Score and sort
    scored: list[tuple[Path, int, str]] = []
    for f in all_files:
        score, category = score_file(f, root, primary_lang)
        scored.append((f, score, category))

    scored.sort(key=lambda x: x[1], reverse=True)

    # Greedily select within budget
    selected: list[SelectedFile] = []
    total_tokens = 0

    for fpath, score, category in scored:
        if len(selected) >= max_files:
            break

        try:
            content = fpath.read_text(encoding="utf-8", errors="ignore")
        except (OSError, UnicodeDecodeError):
            continue

        if not content.strip():
            continue

        tokens = count_tokens(content)
        if total_tokens + tokens > token_budget:
            continue

        total_tokens += tokens
        rel_path = str(fpath.relative_to(root))
        selected.append(SelectedFile(
            path=rel_path,
            content=content,
            category=category,
            score=score,
        ))

    logger.info(
        f"已选择 {len(selected)} 个文件，"
        f"共 ~{total_tokens:,} tokens"
    )
    return selected
