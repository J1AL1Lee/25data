#!/usr/bin/env python
"""构建PDF知识库的主脚本"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import logging
from src.extractors.pdf_extractor import PDFExtractor
from src.processors.text_splitter import TextSplitter
from src.processors.preprocessor import TextPreprocessor
from src.indexing.embeddings import EmbeddingModel
from src.indexing.vector_store import VectorStore
from config.settings import settings
import json
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """主构建流程"""
    start_time = time.time()
    logger.info("开始构建PDF知识库...")

    # 1. 初始化组件
    logger.info("初始化组件...")
    pdf_extractor = PDFExtractor()
    text_splitter = TextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP
    )
    preprocessor = TextPreprocessor()
    embedding_model = EmbeddingModel(settings.EMBEDDING_MODEL)
    vector_store = VectorStore(str(settings.DB_DIR / "chroma"))

    # 2. 提取PDF文本
    logger.info(f"从 {settings.PDF_DIR} 提取PDF文本...")
    pdf_results = pdf_extractor.extract_batch(str(settings.PDF_DIR))

    if not pdf_results:
        logger.error("未找到PDF文件!")
        return

    logger.info(f"成功提取 {len(pdf_results)} 个PDF文件")

    # 3. 处理和分块文本
    logger.info("处理和分块文本...")
    all_chunks = []

    for pdf_data in pdf_results:
        if not pdf_data['extraction_success']:
            logger.warning(f"跳过提取失败的文件: {pdf_data['metadata']['filename']}")
            continue

        # 预处理文本
        processed = preprocessor.preprocess(pdf_data['full_text'])

        # 分块
        metadata = {
            'source': pdf_data['metadata']['filename'],
            'title': pdf_data['metadata'].get('title', ''),
            'total_pages': pdf_data['metadata']['pages']
        }

        # 按页分块
        for page_data in pdf_data['pages']:
            if page_data['content'].strip():  # 跳过空页
                page_chunks = text_splitter.split_text(
                    page_data['content'],
                    metadata={
                        **metadata,
                        'page_num': page_data['page_num']
                    }
                )
                all_chunks.extend(page_chunks)

    logger.info(f"生成了 {len(all_chunks)} 个文本块")

    # 4. 向量化
    logger.info("开始向量化文本块...")
    texts = [chunk['text'] for chunk in all_chunks]
    metadatas = [chunk['metadata'] for chunk in all_chunks]

    # 批量向量化
    embeddings = embedding_model.encode(texts, batch_size=32)

    # 5. 存储到向量数据库
    logger.info("存储到向量数据库...")
    success = vector_store.add_documents(
        texts=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

    if success:
        # 保存构建元数据
        build_metadata = {
            'build_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'pdf_count': len(pdf_results),
            'chunk_count': len(all_chunks),
            'embedding_model': settings.EMBEDDING_MODEL,
            'chunk_size': settings.CHUNK_SIZE,
            'time_elapsed': time.time() - start_time
        }

        metadata_path = settings.PROJECT_ROOT / "data" / "build_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(build_metadata, f, ensure_ascii=False, indent=2)

        logger.info(f"知识库构建完成! 耗时: {build_metadata['time_elapsed']:.2f}秒")
        logger.info(f"构建统计: {build_metadata}")
    else:
        logger.error("存储向量失败!")


if __name__ == "__main__":
    main()