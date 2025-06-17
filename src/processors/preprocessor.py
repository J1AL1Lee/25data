import jieba
import jieba.analyse
import re
from typing import List, Dict, Set
import logging

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """文本预处理器"""

    def __init__(self):
        # 加载自定义词典（竞赛相关术语）
        self.custom_words = [
            "极地资源勘探", "深海观测", "锰矿发掘", "热液矿床",
            "深海潜标", "区域迷航", "固体燃料", "生物资源",
            "火山避险", "英雄归来", "遥控方式", "程控方式"
        ]
        for word in self.custom_words:
            jieba.add_word(word)

        # 停用词
        self.stop_words = self._load_stop_words()

    def preprocess(self, text: str) -> Dict:
        """预处理文本"""
        # 基础清理
        cleaned_text = self._clean_text(text)

        # 分词
        tokens = list(jieba.cut(cleaned_text))

        # 去除停用词
        filtered_tokens = [t for t in tokens if t not in self.stop_words and len(t) > 1]

        # 提取关键词
        keywords = jieba.analyse.extract_tags(cleaned_text, topK=20, withWeight=True)

        # 提取实体（简单规则）
        entities = self._extract_entities(cleaned_text)

        return {
            'cleaned_text': cleaned_text,
            'tokens': filtered_tokens,
            'keywords': keywords,
            'entities': entities,
            'word_count': len(filtered_tokens)
        }

    def _clean_text(self, text: str) -> str:
        """清理文本"""
        # 统一换行符
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text)
        # 移除特殊字符但保留中文标点
        text = re.sub(r'[^\w\s\u4e00-\u9fa5。！？，、；：""''（）《》【】]', ' ', text)
        return text.strip()

    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """提取命名实体"""
        entities = {
            'tasks': [],  # 任务名称
            'components': [],  # 组件名称
            'requirements': []  # 要求规格
        }

        # 任务名称模式
        task_patterns = [
            r'(启程下潜|深海观测|锰矿发掘|热液矿床|深海潜标|区域迷航|英雄归来)',
            r'(固体燃料|生物资源|火山避险)'
        ]

        for pattern in task_patterns:
            matches = re.findall(pattern, text)
            entities['tasks'].extend(matches)

        # 组件规格模式
        component_patterns = [
            r'(\d+cm×\d+cm×\d+cm)',  # 尺寸
            r'(\d+V)',  # 电压
            r'(\d+分钟)',  # 时间
        ]

        for pattern in component_patterns:
            matches = re.findall(pattern, text)
            entities['requirements'].extend(matches)

        # 去重
        for key in entities:
            entities[key] = list(set(entities[key]))

        return entities

    def _load_stop_words(self) -> Set[str]:
        """加载停用词"""
        # 这里使用简单的停用词列表，实际项目中应该从文件加载
        return set([
            '的', '了', '和', '是', '在', '有', '我', '你', '他', '她', '它',
            '这', '那', '哪', '什么', '怎么', '如果', '但是', '因为', '所以',
            '，', '。', '！', '？', '、', '；', '：', '"', '"', ''', ''',
            '（', '）', '《', '》', '【', '】', ' ', '\n', '\t'
        ])