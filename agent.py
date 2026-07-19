from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ==================== 1. 准备文档，建向量数据库 ====================
long_doc = """
第一部分：公司简介
我们公司成立于2010年，是一家专注于人工智能解决方案的科技企业。总部位于北京，在上海、深圳设有分公司。公司目前拥有员工500余人。

第二部分：考勤制度
上班时间为弹性工作制，核心工作时间是上午10:00到下午16:00。员工每日工作时间不少于8小时。迟到或早退累计超过3次，每次扣款100元。

第三部分：休假制度
员工入职满一年后，享有每年5天带薪年假。入职满三年，年假增加至10天。入职满十年，年假增加至15天。年假可分次使用，每次不少于半天。

第四部分：报销制度
出差交通费实报实销，住宿标准为一线城市400元/晚，其他城市250元/晚。加班餐补为每人每次30元，需提供餐饮发票。

第五部分：培训与发展
公司每年为每位员工提供不低于2000元的培训预算。员工可自主选择线上或线下课程，需提前向直属领导申请。公司内部每月举办一次技术分享会。
"""

# 切片
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, chunk_overlap=50,
    separators=["\n\n", "\n", "。", "，", " "]
)
chunks = text_splitter.split_text(long_doc)

# 向量化 + 存入数据库
embeddings = OllamaEmbeddings(model="nomic-embed-text")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="company_docs")

for i, chunk in enumerate(chunks):
    vector = embeddings.embed_query(chunk)
    collection.add(ids=[f"chunk_{i}"], documents=[chunk], embeddings=[vector])

print(f"✅ 已加载 {len(chunks)} 个文档块到向量数据库\n")


# ==================== 2. 工具：搜索公司文档 ====================
def search_company_docs(query):
    """搜索公司内部文档，根据用户问题返回相关政策内容"""
    question_vector = embeddings.embed_query(query)
    results = collection.query(query_embeddings=[question_vector], n_results=2)

    documents_list = results.get("documents")
    if documents_list is None or len(documents_list) == 0 or len(documents_list[0]) == 0:
        return "未找到相关内容"

    return "\n".join(documents_list[0])


def send_email(to, subject, body):
    """真实发送邮件，自动把中文名转成邮箱地址"""
    print(f"🔍 DEBUG: to = '{to}', subject = '{subject}'")  # ← 加这行

    # 通讯录：中文名 → 邮箱地址（按需修改）
    email_book = {
        "清风": "3753409971@qq.com",
        "王经理": "wang@company.com",
        "小明": "xiaoming@qq.com",
        "醉孤一": "468068115@qq.com",
    }

    # 如果 to 在通讯录里，自动替换成邮箱
    if to in email_book:
        to = email_book[to]

    # QQ 邮箱配置
    smtp_server = "smtp.qq.com"
    smtp_port = 587
    sender_email = "1693464132@qq.com"  # ← 改成你的
    password = "mnnxlvhihjpsbaje"  # ← 改成你的

    # 构建邮件
    msg = MIMEMultipart()
    msg["From"] = sender_email # 寄件人：你的邮箱
    msg["To"] = to             # 收件人：对方的邮箱
    msg["Subject"] = subject   # 主题：这封信的标题
    msg.attach(MIMEText(body, "plain", "utf-8"))

    # 发送
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, to, msg.as_string())
        server.quit()
        print(f"📧 邮件已发送给 {to}")
        return f"邮件已成功发送给{to}"
    except Exception as e:
        return f"邮件发送失败: {str(e)}"


TOOLS = {
    "search_company_docs": search_company_docs,
    "send_email": send_email
}

# ==================== 3. 模型 + 提示词 ====================
model = ChatOllama(model="qwen2.5:3b", temperature=0)

SYSTEM_PROMPT = """你是一个智能助手，你可以使用以下工具来完成用户的任务：

工具1：search_company_docs(query) — 搜索公司内部文档，参数query是用户的自然语言问题（如"年假有几天""迟到怎么扣钱"）
工具2：send_email(to, subject, body) — 发送邮件。to必须是用户明确指定的人名或邮箱（如"张三"或"abc@qq.com"），subject是邮件主题，body是邮件正文。绝对不要把to填成"收件人"或"收件人邮箱地址"这类描述文字。

使用规则：
- 用户问公司制度相关的问题，必须先用 search_company_docs 搜索文档
- 用户让你写邮件，先搜文档确认相关信息，再发邮件
- 需要调工具时，只输出：TOOL_CALL: 工具名(参数1, 参数2, ...)
- 不需要调工具时，直接回复"""


# ==================== 4. Agent 循环 ====================
def run_agent(user_input, max_steps=5):
    messages: list = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_input)
    ]

    for step in range(max_steps):
        print(f"\n{'=' * 50}")
        print(f"第 {step + 1} 步：AI 思考中...")

        response = model.invoke(messages)
        ai_text = response.content
        if isinstance(ai_text, list):
            ai_text = str(ai_text[0]) if ai_text else ""
        ai_text = str(ai_text).strip()

        messages.append(AIMessage(content=ai_text))

        if ai_text.startswith("TOOL_CALL:"):
            tool_call = ai_text.replace("TOOL_CALL:", "").strip()
            print(f"🔧 AI 调用工具: {tool_call}")

            match = re.match(r"(\w+)\((.*)\)", tool_call)
            if match:
                tool_name = match.group(1)
                args_str = match.group(2)
                parts = args_str.split(",", 2)  # 最多切 2 次，得到最多 3 段
                args = [p.strip() for p in parts]

                if tool_name in TOOLS:
                    try:
                        result = TOOLS[tool_name](*args)
                        print(f"✅ 结果: {result[:100]}...")
                    except Exception as e:
                        result = f"出错: {str(e)}"
                        print(f"❌ {result}")
                else:
                    result = f"未知工具: {tool_name}"

                messages.append(HumanMessage(content=f"[工具返回] {result}"))
                continue
        else:
            print(f"💬 AI: {ai_text}")
            return ai_text

    return "Agent 达到最大步数。"


# ==================== 5. 交互对话 ====================
print("╔══════════════════════════════════╗")
print("║   RAG + Agent 智能助手           ║")
print("║   输入 '退出' 结束对话           ║")
print("╚══════════════════════════════════╝")

while True:
    user_msg = input("\n🧑 你: ")
    if user_msg.lower() in ["退出", "quit", "exit", "q"]:
        print("👋 再见！")
        break
    if user_msg.strip() == "":
        continue
    run_agent(user_msg)