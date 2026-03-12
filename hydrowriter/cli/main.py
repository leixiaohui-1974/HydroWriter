"""HydroWriter CLI — multi-engine collaborative writing tool.

Usage:
    hydrowriter write --topic "..." --config engines.yaml
    hydrowriter review --input chapter.md --config engines.yaml
    hydrowriter ppt --topic "..." --slides 10 --output slides.pptx
    hydrowriter info --config engines.yaml
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click


def _run_async(coro):
    """Run async coroutine in sync context."""
    return asyncio.run(coro)


def _load_engines(config_path: str) -> tuple:
    """Load config and create engines."""
    from hydrowriter.config import load_config
    from hydrowriter.engine.factory import create_all_engines

    config = load_config(config_path)
    engines = create_all_engines(config)
    return engines, config


@click.group()
@click.version_option(version="0.1.0", prog_name="hydrowriter")
def cli():
    """HydroWriter — 多引擎协同写作平台"""


@cli.command()
@click.option("--topic", "-t", required=True, help="写作主题")
@click.option("--context", "-c", default="", help="背景上下文")
@click.option("--style", "-s", default="", help="风格指南")
@click.option("--chapter", type=int, default=1, help="章节编号")
@click.option("--config", default="configs/engines.yaml", help="引擎配置文件")
@click.option("--output", "-o", default="", help="输出文件路径")
def write(topic: str, context: str, style: str, chapter: int, config: str, output: str):
    """多引擎协同写作（草稿→评审→修改→编号）"""
    engines, writer_config = _load_engines(config)

    if not engines:
        click.echo("错误：没有可用的引擎。请检查配置文件和 API 密钥。", err=True)
        sys.exit(1)

    click.echo(f"启动写作流水线... 主题: {topic}")
    click.echo(f"可用引擎: {', '.join(engines.keys())}")

    from hydrowriter.pipeline.book_pipeline import BookPipeline

    pipeline = BookPipeline(engines, writer_config)
    result = _run_async(pipeline.run(
        topic=topic, context=context, style_guide=style, chapter_num=chapter,
    ))

    if result.success:
        if output:
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            Path(output).write_text(result.final, encoding="utf-8")
            click.echo(f"输出已保存: {output}")
        else:
            click.echo("\n--- 最终输出 ---\n")
            click.echo(result.final)

        click.echo(f"\n评审平均分: {result.metadata.get('average_score', 'N/A')}")
        click.echo(f"参与引擎: {', '.join(result.metadata.get('draft_engines', []))}")
    else:
        click.echo("写作流水线失败。", err=True)
        sys.exit(1)


@cli.command()
@click.option("--input", "-i", "input_file", required=True, help="待评审文件")
@click.option("--roles", "-r", default="teacher,engineer,reader", help="评审角色（逗号分隔）")
@click.option("--config", default="configs/engines.yaml", help="引擎配置文件")
def review(input_file: str, roles: str, config: str):
    """多引擎多角色评审"""
    engines, writer_config = _load_engines(config)

    if not engines:
        click.echo("错误：没有可用的引擎。", err=True)
        sys.exit(1)

    content = Path(input_file).read_text(encoding="utf-8")
    role_list = [r.strip() for r in roles.split(",")]

    click.echo(f"启动评审... 文件: {input_file}")
    click.echo(f"评审角色: {', '.join(role_list)}")

    from hydrowriter.pipeline.book_pipeline import BookPipeline

    pipeline = BookPipeline(engines, writer_config)
    result = _run_async(pipeline.review_only(content, roles=role_list))

    click.echo("\n--- 评审结果 ---\n")

    consensus = result.get("consensus", [])
    if consensus:
        click.echo("共识问题（必须修改）:")
        for item in consensus:
            click.echo(f"  🔴 {item['issue']} (×{item['count']})")

    divergence = result.get("divergence", [])
    if divergence:
        click.echo("\n分歧建议（酌情采纳）:")
        for item in divergence:
            click.echo(f"  🟡 {item['issue']}")

    avg_score = result.get("average_score", 0)
    click.echo(f"\n综合评分: {avg_score}/10")


@cli.command()
@click.option("--topic", "-t", default="", help="PPT主题（LLM生成模式）")
@click.option("--input", "-i", "input_file", default="", help="Markdown输入文件")
@click.option("--slides", type=int, default=10, help="幻灯片数量")
@click.option("--output", "-o", default="output.pptx", help="输出文件路径")
@click.option("--config", default="configs/engines.yaml", help="引擎配置文件")
def ppt(topic: str, input_file: str, slides: int, output: str, config: str):
    """生成PPT（从Markdown或LLM生成）"""
    from hydrowriter.pipeline.ppt_pipeline import PPTPipeline

    if not topic and not input_file:
        click.echo("错误：需要指定 --topic 或 --input。", err=True)
        sys.exit(1)

    engines, writer_config = _load_engines(config)
    pipeline = PPTPipeline(engines=engines, config=writer_config)

    if input_file:
        content = Path(input_file).read_text(encoding="utf-8")
        click.echo(f"从 Markdown 生成 PPT: {input_file}")
        result = _run_async(pipeline.generate_from_markdown(content, output))
    else:
        click.echo(f"LLM 生成 PPT: {topic} ({slides} slides)")
        result = _run_async(pipeline.generate_with_llm(topic, slides, output))

    if result.success:
        click.echo(f"PPT 已生成: {result.output_path} ({result.slide_count} slides)")
    else:
        click.echo(f"PPT 生成失败: {result.error}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--config", default="configs/engines.yaml", help="引擎配置文件")
def info(config: str):
    """显示引擎配置信息"""
    from hydrowriter.config import load_config

    writer_config = load_config(config)

    click.echo("HydroWriter 引擎配置:")
    click.echo(f"  配置文件: {config}")
    click.echo(f"  默认引擎: {writer_config.default_engine}")
    click.echo(f"  并行草稿: {'是' if writer_config.parallel_drafting else '否'}")
    click.echo(f"  评审角色: {', '.join(writer_config.review_roles)}")
    click.echo(f"  共识阈值: {writer_config.consensus_threshold}")
    click.echo()

    for name, ec in writer_config.engines.items():
        status = "✅" if ec.is_available else "❌"
        click.echo(f"  {status} {name}: {ec.provider}/{ec.model} (role={ec.role})")
        if ec.base_url:
            click.echo(f"     base_url: {ec.base_url}")


if __name__ == "__main__":
    cli()
