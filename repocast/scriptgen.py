from __future__ import annotations

import logging
import time

from google import genai
from google.genai.types import GenerateContentConfig
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from repocast.config import Config
from repocast.selector import SelectedFile
from repocast.utils import RateLimiter, count_tokens

logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTION = """\
你是一位专业的中文技术播客主持人。你需要将GitHub代码仓库的内容转化为一期引人入胜的中文播客脚本。

播客风格要求：
- 使用口语化的中文，适合听众收听
- 避免念出具体的代码符号（如括号、分号、花括号等）
- 对于变量名和函数名，用中文解释其含义
- 适当使用类比和比喻帮助理解
- 语气自然，像一位经验丰富的工程师在和朋友聊技术
- 使用"我们"来拉近与听众的距离
"""

SCRIPT_PROMPT_TEMPLATE = """\
基于以下项目信息，生成一期完整的中文技术播客脚本（约3000-5000字）。

项目名称：{repo_name}

{readme_section}

{manifest_section}

代码文件内容：
{code_section}

播客结构要求：
1. 【开场】(约200字) - 引入项目，说明它解决什么问题，为什么值得关注
2. 【项目概览】(约500字) - 技术栈、项目结构、核心依赖
3. 【架构解析】(约800字) - 整体架构设计，模块划分，数据流向
4. 【核心代码解读】(约1500字) - 挑选2-3个最有代表性的模块深入讲解
5. 【设计亮点与不足】(约500字) - 值得学习的设计和可改进之处
6. 【总结】(约300字) - 概括要点，推荐适合的读者群体

注意：
- 全程使用中文
- 用口语化表达，就像在和听众面对面交流
- 对英文技术术语在首次出现时加中文解释
- 不要使用markdown格式符号（如#、*、-列表等）
- 每个段落之间用空行分隔，方便后续语音合成添加停顿
- 段落标题用【】包裹即可
"""

MAP_PROMPT_TEMPLATE = """\
请分析以下代码文件，总结它们的作用、设计模式和关键实现细节。

项目名称：{repo_name}

文件内容：
{code_section}

请输出结构化的中文分析（约1000-2000字），包含：
1. 涉及的模块概述
2. 关键设计决策和使用的设计模式
3. 值得讲解的核心代码逻辑
4. 模块间的依赖关系

注意用口语化中文描述，不要使用markdown格式符号。
"""

REDUCE_PROMPT_TEMPLATE = """\
基于以下项目分析结果，生成一期完整的中文技术播客脚本（约3000-5000字）。

项目名称：{repo_name}

{readme_section}

{manifest_section}

各模块分析汇总：
{analysis}

播客结构要求：
1. 【开场】(约200字) - 引入项目，说明它解决什么问题，为什么值得关注
2. 【项目概览】(约500字) - 技术栈、项目结构、核心依赖
3. 【架构解析】(约800字) - 整体架构设计，模块划分，数据流向
4. 【核心代码解读】(约1500字) - 挑选2-3个最有代表性的模块深入讲解
5. 【设计亮点与不足】(约500字) - 值得学习的设计和可改进之处
6. 【总结】(约300字) - 概括要点，推荐适合的读者群体

注意：
- 全程使用中文
- 用口语化表达，就像在和听众面对面交流
- 对英文技术术语在首次出现时加中文解释
- 不要使用markdown格式符号
- 每个段落之间用空行分隔
- 段落标题用【】包裹
"""


def _format_files_for_prompt(files: list[SelectedFile]) -> str:
    parts = []
    for f in files:
        parts.append(f"--- 文件: {f.path} ---")
        parts.append(f.content)
        parts.append("")
    return "\n".join(parts)


def _build_chunks(
    files: list[SelectedFile], max_tokens: int
) -> list[list[SelectedFile]]:
    chunks: list[list[SelectedFile]] = []
    current_chunk: list[SelectedFile] = []
    current_tokens = 0

    for f in files:
        file_tokens = count_tokens(f.content)
        if current_tokens + file_tokens > max_tokens and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_tokens = 0
        current_chunk.append(f)
        current_tokens += file_tokens

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    retry=retry_if_exception_type((Exception,)),
    before_sleep=lambda retry_state: logger.warning(
        f"API调用失败，{retry_state.next_action.sleep}s后重试... "
        f"(第{retry_state.attempt_number}次)"
    ),
)
def _call_gemini(client: genai.Client, model_name: str, prompt: str) -> str:
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.7,
            max_output_tokens=8192,
        ),
    )
    if not response.text:
        raise RuntimeError("Gemini返回了空响应")
    return response.text


def generate_script(
    files: list[SelectedFile],
    repo_name: str,
    config: Config,
) -> str:
    client = genai.Client(api_key=config.gemini_api_key)
    rate_limiter = RateLimiter()

    # Separate README and manifest from code files
    readme_content = ""
    manifest_content = ""
    code_files: list[SelectedFile] = []

    for f in files:
        if f.category == "readme":
            readme_content = f.content
        elif f.category == "manifest":
            manifest_content = f.content
        else:
            code_files.append(f)

    readme_section = f"README内容：\n{readme_content}" if readme_content else "（无README）"
    manifest_section = f"项目配置文件：\n{manifest_content}" if manifest_content else ""

    # Calculate total tokens of code
    total_code_tokens = sum(count_tokens(f.content) for f in code_files)
    logger.info(f"代码内容共 ~{total_code_tokens:,} tokens")

    if total_code_tokens <= config.max_tokens_per_chunk:
        # Single pass
        logger.info("使用单次生成模式...")
        code_section = _format_files_for_prompt(code_files)
        prompt = SCRIPT_PROMPT_TEMPLATE.format(
            repo_name=repo_name,
            readme_section=readme_section,
            manifest_section=manifest_section,
            code_section=code_section,
        )

        estimated_tokens = count_tokens(prompt)
        rate_limiter.wait_if_needed(estimated_tokens)

        logger.info("正在调用 Gemini 生成播客脚本...")
        script = _call_gemini(client, config.gemini_model, prompt)
        rate_limiter.record_request(estimated_tokens)

    else:
        # Map-reduce mode
        chunks = _build_chunks(code_files, config.max_tokens_per_chunk - 50_000)
        logger.info(f"内容较多，使用分段分析模式（{len(chunks)}段）...")

        summaries = []
        for i, chunk in enumerate(chunks):
            logger.info(f"分析第 {i+1}/{len(chunks)} 段...")
            code_section = _format_files_for_prompt(chunk)
            prompt = MAP_PROMPT_TEMPLATE.format(
                repo_name=repo_name,
                code_section=code_section,
            )

            estimated_tokens = count_tokens(prompt)
            rate_limiter.wait_if_needed(estimated_tokens)

            summary = _call_gemini(client, config.gemini_model, prompt)
            rate_limiter.record_request(estimated_tokens)
            summaries.append(summary)

            # Respect rate limits between calls
            if i < len(chunks) - 1:
                time.sleep(4)

        # Reduce phase
        logger.info("正在合并分析结果，生成最终播客脚本...")
        analysis = "\n\n".join(
            f"=== 第{i+1}部分分析 ===\n{s}" for i, s in enumerate(summaries)
        )
        prompt = REDUCE_PROMPT_TEMPLATE.format(
            repo_name=repo_name,
            readme_section=readme_section,
            manifest_section=manifest_section,
            analysis=analysis,
        )

        estimated_tokens = count_tokens(prompt)
        rate_limiter.wait_if_needed(estimated_tokens)
        script = _call_gemini(client, config.gemini_model, prompt)
        rate_limiter.record_request(estimated_tokens)

    logger.info(f"脚本生成完成，共 {len(script)} 字")
    return script
