"""Streamlit Web界面"""
import streamlit as st
import requests
import json
import time

# API配置
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="PDF知识库问答系统",
    page_icon="📚",
    layout="wide"
)

st.title("📚 PDF知识库问答系统")
st.markdown("基于第七届全国青少年人工智能创新挑战赛文档构建的智能问答系统")

# 侧边栏
with st.sidebar:
    st.header("系统状态")

    # 获取系统状态
    try:
        response = requests.get(f"{API_URL}/")
        if response.status_code == 200:
            health_data = response.json()
            st.success(f"状态: {health_data['status']}")
            stats = health_data['vector_store_stats']
            st.info(f"文档数量: {stats['document_count']}")
    except:
        st.error("无法连接到后端服务")

    st.divider()

    # 查询设置
    st.header("查询设置")
    n_results = st.slider("返回结果数", 1, 10, 5)
    score_threshold = st.slider("相似度阈值", 0.0, 1.0, 0.7, 0.05)

# 主界面
tab1, tab2, tab3 = st.tabs(["💬 问答", "🔍 任务查询", "📊 统计"])

with tab1:
    st.header("智能问答")

    # 查询输入
    query = st.text_input("请输入您的问题:", placeholder="例如：机器人的尺寸要求是什么？")

    if st.button("🔍 查询", type="primary"):
        if query:
            with st.spinner("正在搜索..."):
                try:
                    response = requests.post(
                        f"{API_URL}/query",
                        json={
                            "question": query,
                            "n_results": n_results,
                            "score_threshold": score_threshold
                        }
                    )

                    if response.status_code == 200:
                        data = response.json()

                        st.success(f"找到 {data['total_results']} 个相关结果 "
                                   f"(耗时: {data['processing_time']:.2f}秒)")

                        for i, answer in enumerate(data['answers']):
                            with st.expander(
                                    f"结果 {i + 1} - {answer['source']} "
                                    f"(相似度: {answer['score']:.3f})"
                            ):
                                if answer['page']:
                                    st.caption(f"页码: {answer['page']}")
                                st.write(answer['content'])
                    else:
                        st.error("查询失败")
                except Exception as e:
                    st.error(f"错误: {str(e)}")
        else:
            st.warning("请输入问题")

with tab2:
    st.header("任务相关查询")

    # 任务列表
    tasks = {
        "startup": "启程下潜",
        "observation": "深海观测",
        "mining": "锰矿发掘",
        "thermal": "热液矿床",
        "beacon": "深海潜标",
        "navigation": "区域迷航",
        "return": "英雄归来"
    }

    selected_task = st.selectbox("选择任务:", options=list(tasks.keys()),
                                 format_func=lambda x: tasks[x])

    if st.button("查询任务详情"):
        with st.spinner("正在查询..."):
            try:
                response = requests.get(f"{API_URL}/search/tasks/{selected_task}")
                if response.status_code == 200:
                    data = response.json()
                    st.subheader(f"任务: {data['task']}")

                    for result in data['results']:
                        with st.expander(f"相关内容 (分数: {result['score']:.3f})"):
                            st.write(result['content'])
            except Exception as e:
                st.error(f"错误: {str(e)}")

with tab3:
    st.header("系统统计")

    if st.button("刷新统计"):
        try:
            response = requests.get(f"{API_URL}/stats")
            if response.status_code == 200:
                stats = response.json()

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("向量存储统计")
                    st.json(stats['vector_store'])

                with col2:
                    st.subheader("嵌入模型信息")
                    st.json(stats['embedding_model'])
        except Exception as e:
            st.error(f"错误: {str(e)}")

# 页脚
st.divider()
st.caption("PDF知识库问答系统 v1.0.0")