import os

# Model used for the experiment.
# Change this if you want to test another supported model.
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

# Number of documents returned by the retriever.
TOP_K = 3

# Minimum retrieval similarity required before documents are considered relevant.
RETRIEVAL_THRESHOLD = 0.10

# Location of the experiment files.
KNOWLEDGE_BASE_PATH = "documents/knowledge_base.txt"
QUESTIONS_PATH = "data/questions.json"
RESULTS_PATH = "results/results.csv"

# OpenAI API key is read from the environment.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print(
        "WARNING: OPENAI_API_KEY is not set. "
        "Set it before running the experiment."
    )
