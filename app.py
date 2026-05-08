import streamlit as st
import os
import base64
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# ── 1. تحميل اللوجو (المسار الجديد) ─────────────────────────────────────────
def get_logo_base64(path="logo.png"):
    try:
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        return f"data:image/jpeg;base64,{data}"
    except FileNotFoundError:
        return None

logo_src = get_logo_base64()
logo_html = f'<img src="{logo_src}" style="height:80px; margin-bottom:10px;">' if logo_src else "🌱"

# ── 2. إعدادات الصفحة و CSS (نفس كودك الأصلي بالظبط) ────────────────────────
st.set_page_config(page_title="AGRIRA - Intelligent Agriculture RAG", page_icon="🌱")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    @import url('https://googleapis.com');
    html, body, [data-testid="stAppViewContainer"] { font-family: 'Cairo', sans-serif; background-color: #f2f8f5; }
    .stChatMessage { background-color: #fafdfb !important; border: 1px solid #e0ebe4 !important; border-radius: 15px !important; padding: 15px !important; margin-bottom: 10px !important; }
    .custom-header { background: linear-gradient(135deg, #1b4f31 0%, #2b7a8a 100%); color: white; padding: 20px; border-radius: 15px; text-align: center; margin-bottom: 25px; }
    .welcome-list-item { list-style: none; padding-left: 25px; position: relative; margin-bottom: 10px; }
    .welcome-list-item::before { content: "🌱"; position: absolute; left: 0; }
</style>
""", unsafe_allow_html=True)

st.markdown(f'<div class="custom-header">{logo_html}<h2>AGRIRA</h2><p>Intelligent Agriculture RAG 🌿</p></div>', unsafe_allow_html=True)

# ── 3. وظائف المراجع (APA Citation) اللي كانت في كودك ────────────────────────
def build_apa_citation(metadata):
    author = metadata.get("author", "Unknown Author")
    year = metadata.get("year", "n.d.")
    title = metadata.get("title", "Untitled")
    page = metadata.get("page", None)
    citation = f"{author}. ({year}). *{title}*."
    if page:
        citation += f" p. {int(page) + 1}"
    return citation

# ── 4. بناء الـ RAG Chain ───────────────────────────────────────────────
@st.cache_resource
def build_rag_chain():
    # سحب المفتاح للأمان (أو استبدليه بمفتاحك هنا مؤقتاً)
    api_key = os.environ.get("GOOGLE_API_KEY", "حط_مفتاحك_هنا_لو_عايز_تجرب")
    
    embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    vector_store = Chroma(persist_directory="VDB", embedding_function=embedding_model)
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 4})
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2, google_api_key=api_key)
    
    system_prompt = (
        "You are AGRIRA, a professional Agriculture Assistant. "
        "Use the retrieved context about agriculture to answer the user's question. "
        "If the answer is not in the context, say that you don't know.\n\nContext: {context}"
    )
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, combine_docs_chain)

rag_chain = build_rag_chain()

# ── 5. إدارة المحادثة ────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# رسالة الترحيب
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown("""
        Hello I am **AGRIRA**<br>
        Your smart assistant in climate-smart agriculture 🌾<br><br>
        I can help you with:<br>
        <div class="welcome-list-item">Choosing suitable crops based on climate</div>
        <div class="welcome-list-item">Optimizing water consumption</div>
        """, unsafe_allow_html=True)

# عرض التاريخ
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# إدخال المستخدم
query = st.chat_input("Ask about agriculture topics...")
if query:
    with st.chat_message("user"): st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})
    
    with st.spinner("AGRIRA is thinking..."):
        result = rag_chain.invoke({"input": query})
        answer = result["answer"]
    
    with st.chat_message("assistant"):
        st.markdown(answer)
        st.markdown("---")
        st.markdown("<small>📚 <b>References</b></small>", unsafe_allow_html=True)
        seen_citations = set()
        for doc in result["context"]:
            citation = build_apa_citation(doc.metadata)
            if citation not in seen_citations:
                seen_citations.add(citation)
                st.caption(citation)
    
    st.session_state.messages.append({"role": "assistant", "content": answer})
