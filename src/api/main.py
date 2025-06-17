from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import time
import logging
from typing import Dict
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from .models import QueryRequest, QueryResponse, HealthResponse, BuildStatusResponse
from ..retrieval.searcher import Searcher
from ..indexing.vector_store import VectorStore
from ..indexing.embeddings import EmbeddingModel
from config.settings import settings

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="PDF知识库问答系统",
    description="基于向量搜索的PDF文档问答系统",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化组件
vector_store = VectorStore(str(settings.DB_DIR / "chroma"))
embedding_model = EmbeddingModel(settings.EMBEDDING_MODEL)
searcher = Searcher(vector_store, embedding_model)

# 全局状态
build_status = {
    "is_building": False,
    "processed_files": 0,
    "total_chunks": 0,
    "message": "系统就绪"
}


@app.get("/", response_model=HealthResponse)
async def health_check():
    """健康检查接口"""
    stats = vector_store.get_collection_stats()
    return HealthResponse(
        status="healthy",
        message="PDF知识库问答系统运行正常",
        vector_store_stats=stats
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """问答查询接口"""
    start_time = time.time()

    try:
        # 执行搜索
        results = searcher.search(
            query=request.question,
            n_results=request.n_results,
            score_threshold=request.score_threshold
        )

        # 格式化响应
        answers = []
        for result in results:
            answers.append({
                "content": result['content'],
                "score": result['score'],
                "source": result['metadata'].get('source', 'unknown'),
                "page": result['metadata'].get('page_num', None)
            })

        processing_time = time.time() - start_time

        return QueryResponse(
            question=request.question,
            answers=answers,
            total_results=len(answers),
            processing_time=processing_time
        )

    except Exception as e:
        logger.error(f"查询错误: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/build/status", response_model=BuildStatusResponse)
async def get_build_status():
    """获取知识库构建状态"""
    return BuildStatusResponse(**build_status)


@app.post("/build/start")
async def start_build(background_tasks: BackgroundTasks):
    """启动知识库构建"""
    if build_status["is_building"]:
        raise HTTPException(status_code=400, detail="构建正在进行中")

    # 在后台启动构建任务
    background_tasks.add_task(build_knowledge_base)

    return {"message": "知识库构建已启动"}


async def build_knowledge_base():
    """后台构建知识库任务"""
    global build_status

    build_status["is_building"] = True
    build_status["message"] = "正在构建知识库..."

    try:
        # 这里调用实际的构建逻辑
        from scripts.build_knowledge_base import main as build_main
        build_main()

        build_status["message"] = "知识库构建完成"
    except Exception as e:
        build_status["message"] = f"构建失败: {str(e)}"
        logger.error(f"构建失败: {str(e)}")
    finally:
        build_status["is_building"] = False


@app.get("/stats")
async def get_statistics():
    """获取系统统计信息"""
    stats = vector_store.get_collection_stats()
    return {
        "vector_store": stats,
        "embedding_model": {
            "name": embedding_model.model_name,
            "dimension": embedding_model.dimension,
            "device": embedding_model.device
        }
    }


# 添加更多业务相关的接口
@app.get("/search/tasks/{task_name}")
async def search_by_task(task_name: str, n_results: int = 5):
    """根据任务名称搜索相关内容"""
    # 任务名称映射
    task_mapping = {
        "startup": "启程下潜",
        "observation": "深海观测",
        "mining": "锰矿发掘",
        "thermal": "热液矿床",
        "beacon": "深海潜标",
        "navigation": "区域迷航",
        "return": "英雄归来"
    }

    chinese_task = task_mapping.get(task_name, task_name)

    results = searcher.search(
        query=chinese_task,
        n_results=n_results
    )

    return {
        "task": chinese_task,
        "results": results
    }


@app.get("/search/requirements/{category}")
async def search_requirements(category: str):
    """搜索特定类别的要求"""
    category_queries = {
        "robot": "机器人要求 尺寸 控制器",
        "competition": "竞赛流程 时间 规则",
        "scoring": "评分 分数 标准"
    }

    query = category_queries.get(category, category)
    results = searcher.search(query=query, n_results=10)

    return {
        "category": category,
        "results": results
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)