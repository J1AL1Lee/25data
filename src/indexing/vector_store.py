import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Optional
import logging
import uuid

logger = logging.getLogger(__name__)


class VectorStore:
    """向量存储管理器"""

    def __init__(self, persist_directory: str = "./data/db/chroma"):
        self.persist_directory = persist_directory

        # 初始化ChromaDB客户端
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # 创建或获取集合
        self.collection_name = "pdf_knowledge_base"
        self._init_collection()

    def _init_collection(self):
        """初始化集合"""
        try:
            self.collection = self.client.get_collection(self.collection_name)
            logger.info(f"加载现有集合: {self.collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"创建新集合: {self.collection_name}")

    def add_documents(self,
                      texts: List[str],
                      embeddings: np.ndarray,
                      metadatas: List[Dict] = None,
                      ids: List[str] = None) -> bool:
        """添加文档到向量库"""
        try:
            # 生成ID
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in range(len(texts))]

            # 确保metadata不为None
            if metadatas is None:
                metadatas = [{} for _ in range(len(texts))]

            # 批量添加
            batch_size = 100
            for i in range(0, len(texts), batch_size):
                end_idx = min(i + batch_size, len(texts))

                self.collection.add(
                    documents=texts[i:end_idx],
                    embeddings=embeddings[i:end_idx].tolist(),
                    metadatas=metadatas[i:end_idx],
                    ids=ids[i:end_idx]
                )

            logger.info(f"成功添加 {len(texts)} 个文档到向量库")
            return True

        except Exception as e:
            logger.error(f"添加文档失败: {str(e)}")
            return False

    def search(self,
               query_embeddings: np.ndarray,
               n_results: int = 5,
               filter_dict: Dict = None) -> Dict:
        """向量搜索"""
        try:
            results = self.collection.query(
                query_embeddings=query_embeddings.tolist(),
                n_results=n_results,
                where=filter_dict
            )

            return results

        except Exception as e:
            logger.error(f"搜索失败: {str(e)}")
            return {"documents": [], "metadatas": [], "distances": []}

    def get_collection_stats(self) -> Dict:
        """获取集合统计信息"""
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "persist_directory": self.persist_directory
        }

    def reset_collection(self):
        """重置集合（清空所有数据）"""
        try:
            self.client.delete_collection(self.collection_name)
            self._init_collection()
            logger.info("集合已重置")
        except Exception as e:
            logger.error(f"重置集合失败: {str(e)}")