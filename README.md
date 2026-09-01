# Multi-Agent LLM Output Evaluation & Arbitration System

A multi-agent LLM evaluation system that independently evaluates generated answers for **accuracy, logical consistency, and completeness**, detects evaluator disagreement, and invokes an adjudicator when the evaluators disagree.

The system produces a final **verdict, score, confidence, reasoning, and identified issues** instead of relying on a single LLM judgment.

---

## Overview

Large Language Models can produce answers that sound convincing while containing factual errors, logical mistakes, or missing information.

A single evaluator can also make mistakes.

This project addresses that problem by using multiple independent evaluation agents and a second-stage arbitration process.

### Evaluation Pipeline

```text
                    User Evaluation
                           |
                           v
                    LangGraph Flow
                           |
          +----------------+----------------+
          |                |                |
          v                v                v
      Accuracy          Logic         Completeness
       Agent            Agent             Agent
          |                |                |
          +----------------+----------------+
                           |
                           v
                      Aggregation
                           |
                           v
                  Disagreement Detection
                           |
                    +------+------+
                    |             |
                 Agreement    Disagreement
                    |             |
                    |             v
                    |        Adjudicator
                    |             |
                    +-------------+
                           |
                           v
                     Final Decision
              Verdict / Score / Confidence
                    / Reasoning / Issues
```

---

## Key Features

- **Multi-agent evaluation**
  - Accuracy evaluator
  - Logic evaluator
  - Completeness evaluator
- **Independent judgments** from each evaluator.
- **Disagreement detection** based on evaluator results.
- **LLM adjudication** when meaningful disagreement is detected.
- **Confidence-aware final decisions**.
- Structured evaluator outputs containing criterion, score, confidence, reasoning, and issues.
- **LangGraph** workflow orchestration.
- **FastAPI** REST API.
- **Streamlit** interactive interface.
- **Ollama** local-first LLM inference.
- **Docker** containerization.
- **AWS ECS Express Mode + Amazon ECR** deployment.

---

## Evaluation Agents

### Accuracy Agent

Evaluates whether the candidate response is factually consistent with the question and reference answer.

### Logic Agent

Evaluates logical correctness and identifies reasoning errors, including invalid inference patterns such as reversing a one-way implication.

### Completeness Agent

Evaluates whether the response sufficiently addresses the question and contains the important information required by the reference answer.

---

## Disagreement Detection

Evaluator outputs are aggregated and analyzed for disagreement.

The disagreement result contains:

```text
has_disagreement
severity
score_spread
reasons
```

When significant disagreement is detected, the system can invoke the adjudicator rather than relying only on an aggregate score.

---

## Adjudication

The adjudicator provides a second-level judgment for conflicting or difficult evaluations.

It produces:

```text
final_verdict
final_score
confidence
reasoning
issues
```

The resulting decision is then exposed as the final system result.

---

## Final Result

The final result contains:

```text
verdict
score
confidence
reasoning
issues
```

This gives the user both a structured decision and an explanation of why that decision was reached.

---

# Architecture

```text
Question
Response
Reference Answer
        |
        v
+----------------------+
|    LangGraph Flow    |
+----------+-----------+
           |
     +-----+-----+-----+
     |           |     |
     v           v     v
 Accuracy      Logic  Completeness
 Evaluator   Evaluator Evaluator
     |           |     |
     +-----------+-----+
                 |
                 v
            Aggregation
                 |
                 v
       Disagreement Detection
                 |
          +------+------+
          |             |
          v             v
       No Major     Adjudicator
     Disagreement       |
          |             |
          +------+------+
                 |
                 v
          Final Result
```

---

# Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| LLM Runtime | Ollama |
| LLM Model | Qwen3 1.7B |
| Workflow Orchestration | LangGraph |
| API | FastAPI |
| Validation | Pydantic |
| UI | Streamlit |
| HTTP Client | Requests |
| Containerization | Docker |
| Cloud Deployment | AWS ECS Express Mode |
| Container Registry | Amazon ECR |
| Testing | Pytest |

---

# Project Structure

```text
multi-agent-llm-output-evaluation/
|
+-- src/
|   +-- agents/
|   |   +-- accuracy.py
|   |   +-- logic.py
|   |   +-- completeness.py
|   |   +-- adjudicator.py
|   |
|   +-- api/
|   |   +-- schemas.py
|   |
|   +-- graph/
|   |   +-- graph.py
|   |   +-- nodes.py
|   |   +-- state.py
|   |
|   +-- llm/
|   |   +-- client.py
|   |
|   +-- models/
|   |   +-- inputs.py
|   |
|   +-- ui/
|   |   +-- app.py
|   |
|   +-- main.py
|
+-- tests/
|
+-- .dockerignore
+-- .gitignore
+-- Dockerfile
+-- requirements.txt
+-- README.md
```

---

# Local Setup

## 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd multi-agent-llm-output-evaluation
```

## 2. Create a virtual environment

### Windows

```cmd
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# Ollama Setup

The project uses Ollama for local LLM inference.

Default Ollama endpoint:

```text
http://localhost:11434
```

Pull the configured model:

```bash
ollama pull qwen3:1.7b
```

The model can be configured using:

```text
LLM_MODEL
```

The Ollama endpoint can be configured using:

```text
OLLAMA_HOST
```

Example:

```cmd
set LLM_MODEL=qwen3:1.7b
set OLLAMA_HOST=http://localhost:11434
```

Make sure Ollama is running before starting the evaluation API.

---

# Running the FastAPI Backend

Start the API:

```bash
uvicorn src.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{
  "status": "ok"
}
```

---

# Running the Streamlit UI

With FastAPI running, open another terminal:

```bash
streamlit run src/ui/app.py
```

The interface allows users to:

1. Enter a question.
2. Enter a candidate response.
3. Enter a reference answer.
4. Run the evaluation pipeline.
5. Inspect all three independent evaluator results.
6. View disagreement detection.
7. Inspect adjudication when triggered.
8. View the final verdict, score, confidence, reasoning, and issues.

---

# API Usage

## Endpoint

```text
POST /evaluate
```

### Request

```json
{
  "question": "What is the capital of France?",
  "response": "The capital of France is Paris.",
  "reference_answer": "Paris is the capital of France."
}
```

### Response Structure

```json
{
  "evaluations": [
    {
      "criterion": "accuracy",
      "score": 10,
      "confidence": 0.98,
      "reasoning": "...",
      "issues": []
    }
  ],
  "disagreement": {
    "has_disagreement": false,
    "severity": "low",
    "score_spread": 0,
    "reasons": []
  },
  "adjudication": null,
  "final_result": {
    "verdict": "...",
    "score": 10,
    "confidence": 0.97,
    "reasoning": "...",
    "issues": []
  }
}
```

The exact scores and reasoning depend on the model evaluation.

---

# Disagreement Example

The following example tests whether the system can detect an invalid logical inference.

### Question

```text
If all employees who complete security training receive a certificate,
and Rahul received a certificate, can we conclude Rahul completed the training?
```

### Reference Answer

```text
We cannot conclude that Rahul completed the training solely because
he received a certificate.
```

### Candidate Response

```text
Rahul received a certificate, so Rahul completed the security training.
```

The candidate response reverses the direction of the implication.

The system should identify the logical issue and use disagreement detection and adjudication when evaluator judgments differ sufficiently.

---

# Testing

The project uses Pytest for automated testing.

Run the full test suite:

```bash
pytest -q
```

The final development test suite contains **39 passing tests**.

The tests cover evaluator components, aggregation and disagreement logic, graph workflow, API behavior, and UI-related functionality.

---

# Docker

The FastAPI backend is containerized using the included Dockerfile.

Build the image:

```bash
docker build -t project3-api .
```

Run the container:

```bash
docker run -p 8000:8000 project3-api
```

The API will be available at:

```text
http://localhost:8000
```

### Important

The Docker image contains the FastAPI application and Python dependencies.

Ollama is **not bundled inside the image**.

The default configuration expects an Ollama server at:

```text
http://localhost:11434
```

When running the container, `OLLAMA_HOST` must point to an Ollama server reachable from the container.

---

# AWS Deployment

The FastAPI application was containerized and deployed to AWS as a deployment exercise.

Deployment flow:

```text
Docker Image
     |
     v
Amazon ECR
     |
     v
AWS ECS Express Mode
     |
     v
Containerized FastAPI API
```

The deployment demonstrated:

- Docker image creation
- Amazon ECR image publishing
- ECS service deployment
- Cloud-hosted API infrastructure
- AWS networking and service configuration
- CloudWatch logging

### Inference Architecture

The primary inference runtime is local Ollama.

The AWS deployment therefore validated the **containerization and cloud deployment path**, while the current LLM inference configuration remains local-first.

A production cloud inference setup would require a remotely reachable model provider or a separately deployed inference service.

---

# Design Decisions

## Why multiple evaluators?

A single LLM evaluator can produce inconsistent or biased judgments.

Independent evaluators provide multiple perspectives:

```text
Accuracy
Logic
Completeness
```

Their outputs can then be compared before the final decision is produced.

## Why disagreement detection?

Simply averaging evaluator scores can hide important conflicts.

The disagreement stage explicitly identifies cases where evaluators differ and can trigger additional reasoning.

## Why an adjudicator?

The adjudicator provides a second-level reasoning stage for difficult or conflicting cases.

The overall decision process is:

```text
Independent Evaluation
        |
        v
Aggregation
        |
        v
Disagreement Detection
        |
        v
Conditional Adjudication
        |
        v
Final Decision
```

## Why Ollama?

Ollama provides a local inference option that allows development and experimentation without requiring a paid cloud LLM API for every evaluation.

The LLM connection is isolated behind a small client abstraction, with the model and endpoint configurable through environment variables.

---

# Limitations

- Evaluation quality depends on the selected LLM.
- LLM-based evaluation is not guaranteed to be objectively correct.
- Local Ollama inference requires the model to be available on the host system.
- The current project does not provide persistent evaluation history.
- The AWS deployment does not include a remotely hosted Ollama inference service.
- Evaluation latency increases because multiple LLM judgments may be executed before the final result.

---

# Future Improvements

- Support additional LLM providers.
- Add model routing and fallback strategies.
- Persist evaluation results.
- Add evaluation history and regression tracking.
- Improve disagreement metrics.
- Add human-in-the-loop review.
- Add batch evaluation.
- Add benchmark datasets and reporting.
- Add authentication and rate limiting.
- Add production cloud inference.
- Add CI/CD automation.

---

# What This Project Demonstrates

This project demonstrates practical experience with:

- Multi-agent LLM system design
- LLM-as-a-judge evaluation
- Structured LLM outputs
- Prompt engineering for evaluation
- Independent evaluator design
- Consensus and disagreement analysis
- LLM adjudication
- Confidence-aware decision making
- LangGraph workflow orchestration
- FastAPI API development
- Streamlit application development
- Docker containerization
- Ollama local inference
- Automated testing with Pytest
- AWS container deployment
- Debugging local-vs-cloud inference issues

---

## License

This project is intended as a portfolio and learning project.
