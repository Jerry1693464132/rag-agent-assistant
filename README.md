# RAG + Agent 智能办公助手

基于 LangChain + Chroma + Ollama 构建的 AI 智能助手。

## 能做什么

- 📄 上传公司文档，自动切片并存入向量数据库
- 🔍 自然语言提问，AI 自动检索文档并回答
- 📧 根据文档内容自动撰写邮件并真实发送（SMTP）

## 技术栈

Python | LangChain | ChromaDB | Ollama | SMTP

## 怎么跑起来

1. 安装 Ollama：https://ollama.com/download
2. 下载模型：ollama pull qwen2.5:3b  可调用api deepseek r1
ollama pull nomic-embed-text
3. 安装依赖：pip install -r requirements.txt
4. 
## 项目结构

- agent_with_rag.py — 主程序
- requirements.txt — Python 依赖
- setup.bat / setup.sh — 一键安装脚本
- chroma_db/ — 向量数据库（自动生成）
