import streamlit as st
from rag_utility import build_qa_chain, ask

st.set_page_config(page_title="Ask About Rebin", page_icon="🤖")
st.title("🤖 Ask About Rebin")
st.caption("Ask me anything about Muhammed Rebin Najeeb's resume!")

# Build the RAG once (cached, so it doesn't rebuild on every message)
@st.cache_resource
def load_chain():
    return build_qa_chain(st.secrets["GROQ_API_KEY"])

qa_chain = load_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

user_q = st.chat_input("Ask about the resume...")
if user_q:
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.write(user_q)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = ask(qa_chain, user_q)
        st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})