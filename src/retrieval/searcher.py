from typing import List, Dict, Optional
import logging
from ..indexing.embeddings import EmbeddingModel
from ..indexing.vector_store import VectorStore
from ..processors.preprocessor import TextPreprocessor

logger = logging.getLogger(__name__)


class Searcher:
    """统一的搜索接口"""

    def __init__(self, vector_store: VectorStore, embedding_model: EmbeddingModel):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.preprocessor = TextPreprocessor()

    def search(self,
               query: str,
               n_results: int = 5,
               score_threshold: float = 0.7) -> List[Dict]:
        """执行搜索"""
        # 预处理查询
        processed = self.preprocessor.preprocess(query)
        logger.info(f"查询关键词: {processed['keywords'][:5]}")

        # 向量化查询
        query_embedding = self.embedding_model.encode_queries(query)

        # 执行向量搜索
        results = self.vector_store.search(
            query_embeddings=query_embedding,
            n_results=n_results * 2  # 获取更多结果用于后续筛选
        )

        # 格式化结果
        formatted_results = []

        for i in range(len(results['documents'][0])):
            distance = results['distances'][0][i]
            score = 1 - distance  # 将距离转换为相似度分数

            if score >= score_threshold:
                formatted_results.append({
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'score': score,
                    'distance': distance
                })

        # 按分数排序
        formatted_results.sort(key=lambda x: x['score'], reverse=True)

        # 返回top n结果
        return formatted_results[:n_results]

    def search_with_filter(self,
                           query: str,
                           filter_dict: Dict,
                           n_results: int = 5) -> List[Dict]:
        """带过滤条件的搜索"""
        query_embedding = self.embedding_model.encode_queries(query)

        results = self.vector_store.search(
            query_embeddings=query_embedding,
            n_results=n_results,
            filter_dict=filter_dict
        )

        # 格式化结果
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'score': 1 - results['distances'][0][i]
            })

        return formatted_results

    def explain_search(self, query: str, result: Dict) -> Dict:
        """解释搜索结果的相关性"""
        # 提取查询关键词
        query_keywords = set(word for word, weight in
                             self.preprocessor.preprocess(query)['keywords'])

        # 提取结果关键词
        result_keywords = set(word for word, weight in
                              self.preprocessor.preprocess(result['content'])['keywords'])

        # 计算关键词重叠
        common_keywords = query_keywords.intersection(result_keywords)

        return {
            'query_keywords': list(query_keywords)[:10],
            'result_keywords': list(result_keywords)[:10],
            'common_keywords': list(common_keywords),
            'keyword_overlap_ratio': len(common_keywords) / len(query_keywords) if query_keywords else 0,
            'relevance_score': result['score']
        }