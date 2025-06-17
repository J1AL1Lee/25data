import re
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class TextSplitter:
    """文本分块器"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """将长文本分割成小块"""
        # 清理文本
        text = self._clean_text(text)

        # 按句子分割
        sentences = self._split_sentences(text)

        # 组合成chunks
        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence_size = len(sentence)

            if current_size + sentence_size > self.chunk_size and current_chunk:
                # 创建chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'size': len(chunk_text),
                    'metadata': metadata
                })

                # 保留重叠部分
                overlap_size = 0
                overlap_sentences = []
                for s in reversed(current_chunk):
                    overlap_size += len(s)
                    overlap_sentences.insert(0, s)
                    if overlap_size >= self.chunk_overlap:
                        break

                current_chunk = overlap_sentences
                current_size = overlap_size

            current_chunk.append(sentence)
            current_size += sentence_size

        # 处理最后的chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'size': len(chunk_text),
                'metadata': metadata
            })

        return chunks

    def _clean_text(self, text: str) -> str:
        """清理文本"""
        # 移除多余的空白
        text = re.sub(r'\s+', ' ', text)
        # 移除特殊字符
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        return text.strip()

    def _split_sentences(self, text: str) -> List[str]:
        """按句子分割文本"""
        # 中文句子分割
        chinese_sentences = re.split(r'[。！？；]', text)
        # 英文句子分割
        english_sentences = re.split(r'[.!?;]', text)

        # 选择分割效果更好的
        if len(chinese_sentences) > len(english_sentences):
            sentences = chinese_sentences
        else:
            sentences = english_sentences

        # 过滤空句子
        return [s.strip() for s in sentences if s.strip()]