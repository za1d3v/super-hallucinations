# super-hallucinations
I conducted an experimental comparison of four RAG configurations to investigate whether retrieval actually reduces hallucination. I evaluated answer correctness, groundedness, citation accuracy, retrieval quality, and abstention behavior, including adversarial cases.

Design the experiment and create the 50–100 question dataset.
Build the basic RAG application.
Run the four experiments and generate the actual results/charts.
Write the technical paper around our actual findings, including the failures.

If RAG performs poorly in one experiment, that's potentially the most interesting finding.

Can RAG Really Prevent Hallucinations?
An Experimental Evaluation of Grounded Generation

This project investigates whether Retrieval-Augmented Generation (RAG) actually reduces hallucinations in Large Language Models (LLMs).

Rather than assuming that RAG eliminates hallucinations, this experiment compares four different approaches:

LLM Only
Basic RAG
RAG + Citations
RAG + Abstention

The experiment evaluates answer correctness, groundedness, citation behavior, and the ability of the system to abstain when sufficient information is unavailable.

Research Questions
Does RAG reduce hallucination compared with using an LLM without retrieval?
Does requiring citations improve factual grounding?
Does explicit abstention reduce unsupported answers?
How does retrieval quality affect hallucination?
Can adversarial questions cause a RAG system to produce unsupported information?

Experimental Design

The experiment uses a small synthetic knowledge base containing factual information about a fictional company called Acme Corporation.

The evaluation dataset contains several types of questions:

Answerable questions
Unanswerable questions
Ambiguous questions
Adversarial questions

Four systems are evaluated against the same questions.

Configuration A — LLM Only

The question is sent directly to the language model.

Question → LLM → Answer

Configuration B — Basic RAG

Relevant documents are retrieved and provided to the LLM.

Question
   ↓
Retriever
   ↓
Relevant Documents
   ↓
LLM
   ↓
Answer

Configuration C — RAG + Citations

The model must identify the source documents supporting its answer.

Configuration D — RAG + Abstention

The model must refuse to answer when the retrieved evidence does not contain enough information.

Metrics

The experiment records:

Correctness

Whether the answer agrees with the expected answer.

Grounded-ness

Whether the answer is supported by the retrieved context.

Citation Accuracy

Whether cited information actually supports the answer.

Abstention Behavior

Whether the system correctly refuses to answer when sufficient evidence is unavailable.

Hallucination Rate

The percentage of responses containing unsupported information.



The purpose of the experiment is to determine whether RAG improves factual reliability under different conditions.

Running the Experiment

Install dependencies:

pip install -r requirements.txt

Set your API key.

Linux/macOS:

export OPENAI_API_KEY="your-api-key"

Windows PowerShell:

$env:OPENAI_API_KEY="your-api-key"

Then run:

python rag_experiment.py

Results will be written to:

results/results.csv
Experimental Hypothesis

The initial hypothesis is:

RAG should reduce hallucination when relevant information exists in the knowledge base, but retrieval alone will not eliminate hallucinations when information is missing, ambiguous, incorrectly retrieved, or adversarially presented.

The experiment is designed to test this hypothesis rather than assume it is correct.

Future Work

Potential extensions include:

Vector databases
Semantic embeddings
Larger evaluation datasets
Multiple LLM providers
Reranking
RAG poisoning attacks
Prompt injection attacks
Automated LLM-as-a-judge evaluation
Human evaluation
Retrieval Precision@K
Recall@K
MRR
NDCG
Production monitoring
Security-specific RAG benchmarks

Author

Independent experimental research into LLM reliability, RAG security, and AI safety.
