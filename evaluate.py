import re
from rag_utility import build_qa_chain, ask
from langchain_groq import ChatGroq

GROQ_API_KEY = "gsk_your_real_key_here"          # same key as your bot
PINECONE_API_KEY = "pcsk_your_real_key_here"     # same Pinecone key as your bot

# --- Load the RAG (the "student") + the Judge (the "teacher") ---
qa_chain = build_qa_chain(GROQ_API_KEY, PINECONE_API_KEY)
judge_llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.0)

# --- Golden dataset: questions + the CORRECT answers (the answer key) ---
golden_dataset = [
    {"question": "What is Muhammed's current job role?",
     "ground_truth": "QA Automation Engineer at Cognizant Technology Solutions."},
    {"question": "Which certification does he hold?",
     "ground_truth": "Anthropic Claude Certified Architect."},
    {"question": "Name one AI project he built.",
     "ground_truth": "The AI Mock Interview Coach (FastAPI + Streamlit + Claude + Docker)."},
    # add 1-2 more if you like
]

# --- The Judge: an LLM that grades an answer 1-5 ---
def extract_score(text):
    m = re.search(r"[1-5]", text)
    return int(m.group()) if m else None

def judge_correctness(question, reference, ai_answer):
    grading_prompt = f"""You are a strict grader.
Compare the AI Answer to the Reference Answer for the question.
Score how CORRECT and complete the AI Answer is, from 1 (wrong) to 5 (perfect).
Reply with ONLY a single number from 1 to 5.

Question: {question}
Reference Answer: {reference}
AI Answer: {ai_answer}

Score (1-5):"""
    reply = judge_llm.invoke(grading_prompt)
    return extract_score(reply.content)

# --- Run the RAG on every question, then judge each answer ---
results = []
for item in golden_dataset:
    answer = ask(qa_chain, item["question"])
    score = judge_correctness(item["question"], item["ground_truth"], answer)
    results.append({**item, "rag_answer": answer, "score": score})
    print("Q:  ", item["question"])
    print("RAG:", answer)
    print("Score:", score, "/5")
    print("-" * 70)

# --- Scorecard ---
scores = [r["score"] for r in results if r["score"] is not None]
average = sum(scores) / len(scores)
print("\n=== SCORECARD ===")
for r in results:
    print(f"{r['score']}/5 : {r['question']}")
print(f"\n⭐ Average correctness: {average:.2f} / 5")