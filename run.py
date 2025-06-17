#!/usr/bin/env python
"""主启动文件"""
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PDF知识库问答系统")
    parser.add_argument("command", choices=["build", "serve", "update"],
                        help="执行命令：build-构建知识库, serve-启动服务, update-更新索引")

    args = parser.parse_args()

    if args.command == "build":
        from scripts.build_knowledge_base import main

        main()
    elif args.command == "serve":
        import uvicorn
        from config.settings import settings

        uvicorn.run(
            "src.api.main:app",
            host=settings.API_HOST,
            port=settings.API_PORT,
            reload=True
        )
    elif args.command == "update":
        from scripts.update_index import main

        main()