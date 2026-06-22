import os
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

RESUME_PDF = "Muhammed_Rebin_Najeeb_Resume.pdf"   # your resume file
INDEX_NAME = "resume-rag"                          # Pinecone index (auto-created)


def build_qa_chain(groq_api_key: str, pinecone_api_key: str):
    os.environ["GROQ_API_KEY"] = groq_api_key
    os.environ["PINECONE_API_KEY"] = pinecone_api_key

    # 1. Load the resume PDF
    documents = PyPDFLoader(RESUME_PDF).load()

    # 2. Split it into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    # 3. Turn chunks into vectors + store them in Pinecone
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    pc = Pinecone(api_key=pinecone_api_key)
    if not pc.has_index(INDEX_NAME):
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,                # all-MiniLM-L6-v2 outputs 384-dim vectors
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(1)

    # store chunks only the first time (avoids duplicates on every redeploy)
    if pc.Index(INDEX_NAME).describe_index_stats().total_vector_count == 0:
        vector_store = PineconeVectorStore.from_documents(
            documents=chunks, embedding=embedding, index_name=INDEX_NAME
        )
    else:
        vector_store = PineconeVectorStore(index_name=INDEX_NAME, embedding=embedding)

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # 4. The LLM (Groq)
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.0)

    # 5. The rule: answer ONLY from the resume, else refuse
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are an assistant that answers questions ONLY about "
            "Muhammed Rebin Najeeb's resume.\n\n"
            "Use ONLY the resume context below. If the answer is not in the "
            "context, reply EXACTLY with:\n"
            "\"I can only answer questions about Muhammed Rebin Najeeb's resume.\"\n"
            "Do not use any outside knowledge.\n\n"
            "Resume context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        ),
    )

    # 6. The RAG chain (retriever + LLM + rule)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )


def ask(qa_chain, question: str) -> str:
    result = qa_chain.invoke({"query": question})
    return result["result"]
