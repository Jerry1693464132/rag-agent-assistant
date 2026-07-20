import streamlit as st
from agent_with_rag import run_agent  # 导入你的 Agent 函数

st.set_page_config(page_title="智能办公助手", page_icon="🤖")
st.title("🤖 RAG + Agent 智能办公助手")

# 初始化聊天记录
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 用户输入
if prompt := st.chat_input("请输入你的问题..."):
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 调用 Agent
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            response = run_agent(prompt)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
