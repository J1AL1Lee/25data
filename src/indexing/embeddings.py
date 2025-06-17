import os
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
import logging
import torch

logger = logging.getLogger(__name__)

class EmbeddingModel:
    """文本向量化模型，离线从本地缓存加载"""

    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        self.model_name = model_name
        logger.info(f"加载向量化模型（离线）: {model_name}")

        # 设备判断
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 默认 HuggingFace 缓存路径
        cache_root = os.path.expanduser("~/.cache/huggingface/hub")
        # 在缓存目录下，寻找以 models--<model_name> 开头的文件夹
        candidates = [d for d in os.listdir(cache_root)
                      if d.startswith(f"models--sentence-transformers--{model_name}")]
        if not candidates:
            raise FileNotFoundError(
                f"本地模型目录未找到，请检查是否已把模型放到 {cache_root} 下，"
                f"并确保目录名以 models--sentence-transformers--{model_name} 开头。"
            )
        # 取第一个匹配
        model_folder = os.path.join(cache_root, candidates[0])

        # 最终用本地路径初始化
        self.model = SentenceTransformer(model_folder, device=self.device)

        # 获取向量维度
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"模型维度: {self.dimension}, 设备: {self.device}")

    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        logger.info(f"向量化 {len(texts)} 条文本…")
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )

    def encode_queries(self, queries: Union[str, List[str]]) -> np.ndarray:
        if isinstance(queries, str):
            queries = [queries]
        prefixed = [f"查询: {q}" for q in queries]
        return self.encode(prefixed)
