@echo off
echo ========================================
echo   RAG + Agent 智能助手 - 环境安装
echo ========================================
echo.
echo [1/3] 安装 Python 依赖...
pip install -r requirements.txt
echo.
echo [2/3] 下载对话模型 qwen2.5:3b...
ollama pull qwen2.5:3b
echo.
echo [3/3] 下载嵌入模型 nomic-embed-text...
ollama pull nomic-embed-text
echo.
echo ========================================
echo   安装完成！运行 python agent_with_rag.py
echo ========================================
pause
