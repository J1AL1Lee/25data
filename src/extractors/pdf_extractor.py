import fitz  # PyMuPDF
import os
from typing import List, Dict, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PDFExtractor:
    """PDF文本提取器"""

    def __init__(self):
        self.supported_extensions = ['.pdf']

    def extract_text(self, pdf_path: str) -> Dict[str, any]:
        """提取单个PDF的文本和元数据"""
        try:
            doc = fitz.open(pdf_path)

            # 提取元数据
            metadata = {
                'title': doc.metadata.get('title', ''),
                'author': doc.metadata.get('author', ''),
                'subject': doc.metadata.get('subject', ''),
                'keywords': doc.metadata.get('keywords', ''),
                'pages': doc.page_count,
                'filename': os.path.basename(pdf_path),
                'filepath': pdf_path
            }

            # 提取全文
            full_text = ""
            pages_content = []

            for page_num in range(doc.page_count):
                page = doc[page_num]
                page_text = page.get_text()
                full_text += page_text

                # 保存每页内容
                pages_content.append({
                    'page_num': page_num + 1,
                    'content': page_text,
                    'char_count': len(page_text)
                })

            doc.close()

            return {
                'metadata': metadata,
                'full_text': full_text,
                'pages': pages_content,
                'total_chars': len(full_text),
                'extraction_success': True
            }

        except Exception as e:
            logger.error(f"提取PDF失败 {pdf_path}: {str(e)}")
            return {
                'metadata': {'filename': os.path.basename(pdf_path)},
                'full_text': '',
                'pages': [],
                'total_chars': 0,
                'extraction_success': False,
                'error': str(e)
            }

    def extract_batch(self, pdf_folder: str) -> List[Dict]:
        """批量提取PDF文件夹中的所有PDF"""
        results = []
        pdf_files = []

        # 查找所有PDF文件
        for root, dirs, files in os.walk(pdf_folder):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))

        logger.info(f"找到 {len(pdf_files)} 个PDF文件")

        # 批量处理
        for i, pdf_path in enumerate(pdf_files):
            logger.info(f"处理进度: {i + 1}/{len(pdf_files)} - {os.path.basename(pdf_path)}")
            result = self.extract_text(pdf_path)
            results.append(result)

        return results