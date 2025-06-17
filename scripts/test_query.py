#!/usr/bin/env python
"""测试查询脚本"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import logging
from src.retrieval.searcher import Searcher
from src.indexing.vector_store import VectorStore
from src.indexing.embeddings import EmbeddingModel
from config.settings import settings
import json

logger = logging.getLogger(__name__)


def test_queries():
    """测试一组预定义的查询"""
    # 初始化组件
    vector_store = VectorStore(str(settings.DB_DIR / "chroma"))
    embedding_model = EmbeddingModel(settings.EMBEDDING_MODEL)
    searcher = Searcher(vector_store, embedding_model)

    # 测试查询列表（基于竞赛内容）
    test_queries = [
        "机器人的尺寸要求是什么？",
        "深海观测任务如何完成？",
        "竞赛时间是多少分钟？",
        "启程下潜的任务要求",
        "如何获得额外加分？",
        "重试次数限制是多少？",
        "热液矿床任务怎么做？",
        "参赛队伍人数要求",
        "工程师手册包含什么内容？",
        "评分标准是什么？"
    ]

    results = {}

    for query in test_queries:
        logger.info(f"\n查询: {query}")
        search_results = searcher.search(query, n_results=3)

        results[query] = []
        for i, result in enumerate(search_results):
            logger.info(f"  结果 {i + 1} (分数: {result['score']:.3f}):")
            logger.info(f"    来源: {result['metadata'].get('source', 'unknown')}")
            logger.info(f"    页码: {result['metadata'].get('page_num', 'N/A')}")
            logger.info(f"    内容: {result['content'][:200]}...")

            results[query].append({
                'score': result['score'],
                'source': result['metadata'].get('source', 'unknown'),
                'page': result['metadata'].get('page_num'),
                'content_preview': result['content'][:200]
            })

    # 保存测试结果
    output_path = settings.PROJECT_ROOT / "data" / "test_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"\n测试结果已保存到: {output_path}")


def interactive_query():
    """交互式查询测试"""
    # 初始化组件
    vector_store = VectorStore(str(settings.DB_DIR / "chroma"))
    embedding_model = EmbeddingModel(settings.EMBEDDING_MODEL)
    searcher = Searcher(vector_store, embedding_model)

    print("\n=== PDF知识库问答系统 ===")
    print("输入问题进行查询，输入 'quit' 退出\n")

    while True:
        query = input("请输入问题: ").strip()

        if query.lower() in ['quit', 'exit', 'q']:
            break

        if not query:
            continue

        # 执行搜索
        results = searcher.search(query, n_results=3)

        if results:
            print(f"\n找到 {len(results)} 个相关结果:\n")
            for i, result in enumerate(results):
                print(f"--- 结果 {i + 1} (相似度: {result['score']:.3f}) ---")
                print(f"来源: {result['metadata'].get('source', 'unknown')}, "
                      f"页码: {result['metadata'].get('page_num', 'N/A')}")
                print(f"内容:\n{result['content']}\n")
        else:
            print("\n未找到相关内容\n")

        # 询问是否需要解释
        explain = input("是否需要解释相关性？(y/n): ").strip().lower()
        if explain == 'y' and results:
            explanation = searcher.explain_search(query, results[0])
            print("\n相关性解释:")
            print(f"查询关键词: {explanation['query_keywords']}")
            print(f"结果关键词: {explanation['result_keywords'][:10]}")
            print(f"共同关键词: {explanation['common_keywords']}")
            print(f"关键词重叠率: {explanation['keyword_overlap_ratio']:.2%}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="测试PDF知识库查询")
    parser.add_argument(
        "--mode",
        choices=["batch", "interactive"],
        default="interactive",
        help="测试模式：batch-批量测试，interactive-交互式查询"
    )

    args = parser.parse_args()

    if args.mode == "batch":
        test_queries()
    else:
        interactive_query()