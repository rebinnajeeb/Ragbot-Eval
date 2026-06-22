import os
from rag_utility import build_qa_chain, ask
from deepeval.models import AnthropicModel
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

ANTHROPIC_API_KEY = "sk-ant-your-real-key-here"
PINECONE_API_KEY = "pcsk_your_real_key_here"

# DeepEval's judge reads the key from the environment
os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY

# 1. Build the RAG bot (the STUDENT)
qa_chain = build_qa_chain(ANTHROPIC_API_KEY, PINECONE_API_KEY)

# 2. The JUDGE = Claude, wrapped in DeepEval's AnthropicModel
judge = AnthropicModel(model="claude-sonnet-4-6", temperature=0)

# 3. The metric: Correctness, scored by G-Eval (LLM-as-a-judge with a criteria)
correctness = GEval(
    name="Correctness",
    criteria="Determine if the 'actual output' is factually correct and complete "
             "based on the 'expected output'.",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    model=judge,
    threshold=0.5,
)

# 4. Golden dataset: questions + correct answers (keep it small: 3-4)
golden_dataset = [
    {"question": "What is Muhammed's current job role?",
     "expected": "QA Automation Engineer at Cognizant Technology Solutions."},
    {"question": "Which certification does he hold?",
     "expected": "Anthropic Claude Certified Architect."},
    {"question": "Name one AI project he built.",
     "expected": "The AI Mock Interview Coach (FastAPI + Streamlit + Claude + Docker)."},
]

# 5. Run RAG -> build a test case -> let DeepEval grade it
print("Running DeepEval...\n")
scores = []
for item in golden_dataset:
    rag_answer = ask(qa_chain, item["question"])

    test_case = LLMTestCase(
        input=item["question"],
        actual_output=rag_answer,
        expected_output=item["expected"],
    )

    correctness.measure(test_case)          # DeepEval scores it (0.0 - 1.0)
    scores.append(correctness.score)

    print("Q:     ", item["question"])
    print("RAG:   ", rag_answer)
    print(f"Score:  {correctness.score:.2f}  (pass >= 0.5)")
    print("Reason:", correctness.reason)    # ← DeepEval EXPLAINS the score!
    print("-" * 70)

# 6. Scorecard
average = sum(scores) / len(scores)
print("\n=== SCORECARD ===")
print(f"⭐ Average correctness: {average:.2f} / 1.00")
