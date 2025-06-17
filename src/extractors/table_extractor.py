import pdfplumber
import pandas as pd
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class TableExtractor:
    """PDF表格提取器"""

    def extract_tables(self, pdf_path: str) -> List[Dict]:
        """提取PDF中的所有表格"""
        tables_data = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    tables = page.extract_tables()

                    for table_num, table in enumerate(tables):
                        if table:  # 确保表格不为空
                            # 转换为DataFrame便于处理
                            df = pd.DataFrame(table[1:], columns=table[0])

                            tables_data.append({
                                'page_num': page_num + 1,
                                'table_num': table_num + 1,
                                'dataframe': df,
                                'rows': len(df),
                                'columns': len(df.columns),
                                'data': table
                            })

            logger.info(f"从 {pdf_path} 提取了 {len(tables_data)} 个表格")
            return tables_data

        except Exception as e:
            logger.error(f"提取表格失败 {pdf_path}: {str(e)}")
            return []