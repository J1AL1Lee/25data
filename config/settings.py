from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # 路径配置
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    PDF_DIR: Path = PROJECT_ROOT / "data" / "pdfs"
    DB_DIR: Path = PROJECT_ROOT / "data" / "db"
    LOG_DIR: Path = PROJECT_ROOT / "logs"

    # 处理参数
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # 模型配置
    EMBEDDING_MODEL: str = "paraphrase-multilingual-MiniLM-L12-v2"

    # API配置
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    class Config:
        env_file = ".env"


settings = Settings()