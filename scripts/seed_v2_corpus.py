from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS_DIR = ROOT / "data" / "v2" / "documents"

DOCUMENTS = {
    "01_rag.txt": """
Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation is a system design that combines information retrieval with language generation. Instead of asking a language model to answer a question using only information encoded in its parameters, a RAG system first retrieves relevant documents or passages from an external knowledge source. The retrieved information is then supplied to the language model as context for generating the answer.

A typical RAG pipeline contains several stages. Documents are collected and divided into smaller chunks. Each chunk can be converted into an embedding and stored in a vector index. When a user asks a question, the question is embedded and compared with stored representations to retrieve relevant chunks. The retrieved chunks are passed to a language model together with the original question.

RAG can improve factual reliability because the generated response can be grounded in information retrieved from a specified knowledge source. It can also make the information used by the generation process easier to inspect. However, retrieval does not automatically guarantee a correct answer. If the retrieved evidence is irrelevant, incomplete, outdated, or contradictory, the resulting answer can still be unreliable.

The quality of a RAG system therefore depends on both retrieval and generation. Good retrieval should return evidence that is relevant to the question, while generation should use that evidence accurately rather than introducing unsupported information.
""",

    "02_large_language_models.txt": """
Large Language Models

Large language models (LLMs) are machine learning models trained on large collections of text. Many modern LLMs use transformer-based neural network architectures. During pretraining, a model learns statistical patterns in text and develops the ability to predict tokens based on surrounding context.

An LLM does not normally retrieve a specific document from its training data when answering a question. Instead, it generates text using patterns represented in its learned parameters and the context supplied at inference time. Additional training techniques can adapt a model to particular tasks or behaviors.

The context provided to an LLM can include the user's question, instructions, previous conversation, retrieved documents, and other application-specific information. The model uses this context when producing its response. The amount of information that can be supplied is constrained by the model's context window.

LLMs are capable of producing fluent and useful responses, but fluency does not guarantee factual correctness. A model can produce statements that sound plausible while being unsupported by available evidence. This behavior is commonly associated with hallucination.

For applications where factual reliability matters, LLM output should therefore be evaluated rather than judged only by how natural or confident the response sounds. External evidence, structured evaluation, and explicit validation can provide additional reliability controls.
""",

    "03_embeddings.txt": """
Text Embeddings

An embedding is a numerical representation of information such as text. An embedding model converts an input text into a vector containing numerical values. Texts with related meanings can often produce vectors that are closer together in an embedding space than texts with unrelated meanings.

Embeddings are commonly used for semantic search and information retrieval. A collection of documents can be divided into chunks, and each chunk can be converted into an embedding. These vectors can then be stored in a vector index. When a user submits a query, the query is embedded using the same or a compatible embedding model. The system compares the query vector with stored document vectors and returns nearby results.

Similarity or distance measures are used to determine which vectors are close to one another. Cosine similarity is one commonly used measure, while some vector databases expose distance values directly. The exact interpretation depends on the embedding model and retrieval system.

Embeddings are useful because semantic retrieval does not require the query and document to contain exactly the same words. However, an embedding similarity score is not proof that a document contains the information required to answer a question. Retrieval results still need to be inspected for relevance and evidentiary support.

Chunk size and chunk boundaries can also affect retrieval quality. Very small chunks may lose context, while very large chunks may contain unrelated information.
""",

    "04_vector_databases.txt": """
Vector Databases

A vector database is a system designed to store and retrieve numerical vector representations efficiently. In retrieval applications, text documents are commonly divided into chunks, converted into embeddings, and stored together with identifiers and metadata.

A vector search begins with a query embedding. The database compares that vector with stored vectors and returns results according to a similarity or distance measure. Applications commonly request the top-k nearest results, where k specifies how many candidate chunks should be returned.

Vector databases can also store metadata associated with each vector. Metadata may include document identifiers, chunk identifiers, source names, timestamps, or other application-specific attributes. Metadata can support filtering and help an application trace a retrieved result back to its original source.

Approximate nearest-neighbor indexing techniques are often used to make vector search efficient at larger scales. These methods trade some aspects of exact search for faster retrieval.

A vector database does not determine whether retrieved information is true. It performs retrieval according to the representations and search configuration provided to it. A high similarity score means that an item is close according to the selected vector representation and distance function; it does not by itself establish factual correctness.

Reliable retrieval systems therefore combine vector search with appropriate chunking, metadata, source information, and downstream evaluation.
""",

    "05_machine_learning.txt": """
Machine Learning

Machine learning is a field of computing in which algorithms learn patterns from data and use those patterns to make predictions or decisions. In supervised learning, a model is trained using examples that contain input features and target labels or values. Classification predicts categories, while regression predicts numerical values.

Unsupervised learning works with data without explicitly provided target labels. Examples include clustering and dimensionality reduction. Other learning settings use feedback or interaction to improve a model's behavior.

A machine learning workflow commonly separates data into training, validation, and test sets. The training data is used to fit model parameters. Validation data can be used to select models or tune hyperparameters. A held-out test set provides an estimate of performance on unseen examples.

Overfitting occurs when a model learns patterns that work well on its training data but do not generalize well to new data. Regularization, appropriate model complexity, sufficient data, and careful validation can help reduce overfitting.

Model evaluation should use metrics appropriate to the task. Classification may use accuracy, precision, recall, F1 score, or other measures. Regression may use metrics such as mean absolute error or root mean squared error.

Machine learning systems can also experience distribution shift when the characteristics of new data differ from the training data. Consequently, good performance on a test set does not guarantee reliable behavior in every future environment.
""",

    "06_deep_learning.txt": """
Deep Learning

Deep learning is a branch of machine learning that uses neural networks with multiple computational layers. These networks can learn increasingly complex representations from data. Parameters are typically adjusted during training using gradient-based optimization and backpropagation.

Different neural network architectures are suited to different types of information. Convolutional neural networks have been widely used for image-related tasks. Recurrent architectures have been used for sequential data. Transformer architectures use attention mechanisms and have become important for language processing and many other domains.

Deep learning models can contain large numbers of parameters and may require substantial computational resources for training and inference. Modern hardware accelerators can make these computations practical at larger scales.

Training a deep learning model involves selecting an architecture, preparing data, defining an objective function, optimizing model parameters, and evaluating performance. As with other machine learning systems, a model may overfit its training data and fail to generalize.

Transfer learning can reduce the amount of task-specific training required by starting from a model that has already learned useful representations. Fine-tuning can then adapt the model to a particular task or domain.

Deep learning models can produce highly capable predictions and generated outputs, but their complexity does not guarantee correctness. Evaluation remains necessary, particularly when the model is used in applications where errors have significant consequences.
""",

    "07_llm_hallucinations.txt": """
LLM Hallucinations

An LLM hallucination is a generated statement that is unsupported by the available evidence or is factually incorrect. Hallucinated text can sometimes appear fluent, detailed, and confident, which makes it difficult to detect through surface-level inspection alone.

Hallucinations can occur for several reasons. A language model generates text based on learned statistical patterns rather than directly verifying every statement against an authoritative source. Ambiguous questions, missing context, incomplete retrieval, or insufficiently represented information can increase the risk of unsupported generation.

Retrieval-augmented generation can reduce some hallucinations by providing relevant external information to the model. However, retrieval alone does not eliminate the problem. A model may ignore retrieved evidence, misunderstand it, combine unrelated information, or generate claims that are not present in the retrieved material.

One useful mitigation strategy is evidence grounding. Generated claims can be compared with retrieved evidence to determine whether the evidence actually supports them. Claim-level evaluation can distinguish supported, partially supported, and unsupported statements.

Other reliability measures include using clear prompts, providing appropriate context, validating generated references, evaluating multiple dimensions of the response, and escalating cases where evaluators strongly disagree.

The goal of hallucination detection is not simply to identify unusual wording. The important question is whether the claims made by a response are supported by reliable information and whether the response accurately represents that information.
""",

    "08_llm_evaluation.txt": """
LLM Output Evaluation

Evaluating an LLM-generated answer involves determining whether the response satisfies the question and whether its claims are reliable. A single score may hide different types of errors, so evaluation can be separated into multiple dimensions.

Accuracy measures whether the claims in a response are factually correct relative to an appropriate reference or evidence. Completeness measures whether important information required to answer the question has been included. Logical consistency examines whether the response is internally coherent and whether its claims conflict with one another.

Evidence grounding examines whether claims are actually supported by the available evidence. A response may be factually plausible while still being ungrounded if the supplied evidence does not support its statements.

Claim-level evaluation can make these dimensions more transparent. A response can be decomposed into individual claims, and each claim can be associated with supporting evidence where available. This makes it possible to identify exactly which parts of an answer are supported or problematic.

Independent evaluators can assess different dimensions separately. Their results can then be aggregated into an overall evaluation. If evaluator scores differ substantially, the system can detect disagreement and invoke an additional adjudication step.

Evidence quality is another useful signal. Retrieved evidence can be assessed using factors such as relevance, source quality, freshness, and consistency. High-quality evidence provides stronger support for grounded evaluation, while weak evidence should reduce confidence in conclusions.

A reliable evaluation system should therefore consider multiple dimensions, explicit evidence, claim-level errors, evaluator confidence, and disagreement rather than relying only on whether an answer sounds convincing.
""",
}


def main() -> None:
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    for filename, content in DOCUMENTS.items():
        path = DOCUMENTS_DIR / filename
        path.write_text(content.strip() + "\n", encoding="utf-8")
        print(f"Created: {path.relative_to(ROOT)}")

    print()
    print(f"Corpus directory: {DOCUMENTS_DIR}")
    print(f"Documents created: {len(DOCUMENTS)}")
    print("Existing demo_ai.txt was preserved.")


if __name__ == "__main__":
    main()