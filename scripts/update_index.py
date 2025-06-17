#!/usr/bin/env python
"""更新索引脚本 - 用于增量更新"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import logging
import json
from datetime import datetime
from src.extractors.pdf_extractor import PDFExtractor
from src.processors.text_splitter import TextSplitter
from src.indexing.embeddings import EmbeddingModel
from src.indexing.vector_store import VectorStore
from config.settings import settings

logger = logging.getLogger(__name__)


def get_new_pdfs(last_build_time: str) -> list:
    """获取上次构建后的新PDF文件"""
    new_files = []
    last_build = datetime.fromisoformat(last_build_time)

    for pdf_file in settings.PDF_DIR.glob("*.pdf"):
        file_mtime = datetime.fromtimestamp(pdf_file.stat().st_mtime)
        if file_mtime > last_build:
            new_files.append(str(pdf_file))

    return new_files


def main():
    """增量更新主流程"""
    logger.info("开始增量更新索引...")

    # 读取上次构建信息
    metadata_path = settings.PROJECT_ROOT / "data" / "build_metadata.json"
    if not metadata_path.exists():
        logger.error("未找到构建元数据，请先运行完整构建!")
        return

    with open(metadata_path, 'r') as f:
        last_build_info = json.load(f)

    # 查找新文件
    new_pdfs = get_new_pdfs(last_build_info['build_time'])

    if not new_pdfs:
        logger.info("没有新的PDF文件需要处理")
        return

    logger.info(f"找到 {len(new_pdfs)} 个新PDF文件")

    # 处理新文件
    pdf_extractor = PDFExtractor()
    text_splitter = TextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP
    )
    embedding_model = EmbeddingModel(settings.EMBEDDING_MODEL)
    vector_store = VectorStore(str(settings.DB_DIR / "chroma"))

    # 提取和处理新文件
    all_chunks = []
    for pdf_path in new_pdfs:
        logger.info(f"处理: {Path(pdf_path).name}")
        pdf_data = pdf_extractor.extract_text(pdf_path)

        if pdf_data['extraction_success']:
            for page_data in pdf_data['pages']:
                if page_data['content'].strip():
                    chunks = text_splitter.split_text(
                        page_data['content'],
                        metadata={
                            'source': pdf_data['metadata']['filename'],
                            'page_num': page_data['page_num'],
                            'update_time': datetime.now().isoformat()
                        }
                    )
                    all_chunks.extend(chunks)

    # 向量化和存储
    if all_chunks:
        texts = [chunk['text'] for chunk in all_chunks]
        metadatas = [chunk['metadata'] for chunk in all_chunks]
        embeddings = embedding_model.encode(texts)

        vector_store.add_documents(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

        logger.info(f"成功更新 {len(all_chunks)} 个文本块")

        # 更新构建元数据
        last_build_info['last_update'] = datetime.now().isoformat()
        last_build_info['update_count'] = last_build_info.get('update_count', 0) + 1
        last_build_info['new_chunks'] = len(all_chunks)

        with open(metadata_path, 'w') as f:
            json.dump(last_build_info, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()