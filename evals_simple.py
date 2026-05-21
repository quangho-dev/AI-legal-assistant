import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI

from ragas import Dataset, experiment
from ragas.llms import llm_factory
from ragas.metrics import DiscreteMetric
from src.graph_builder.agentic_rag_builder import AgenticGraphBuilder
from src.graph_builder.graph_builder import GraphBuilder
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder
from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall, NoiseSensitivity
from ragas.embeddings.base import embedding_factory
from dotenv import load_dotenv
import getpass
from langchain_groq import ChatGroq
import mlflow
from src.graph_builder.agentic_rag_builder import AgenticGraphBuilder
from src.tools.agentic_rag import retrieve_docs

import sys
from pathlib import Path

load_dotenv()

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Add the current directory to the path so we can import rag module when run as a script
sys.path.insert(0, str(Path(__file__).parent))
# from rag import default_rag_client

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_and_save_dataset() -> Path:
    dataset_path = Path("datasets/sample_civil_law_qa_eval.csv")
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    if dataset_path.exists():
        logger.info(f"Dataset already exists at {dataset_path}")
        return dataset_path

    return dataset_path


def load_dataset(dataset_path: Path) -> Dataset:
    dataset = Dataset(
        name="civil_law_qa_eval",
        backend="local/csv",
        root_dir="evals",
    )

    import pandas as pd
    df = pd.read_csv(dataset_path)
    
    for _, row in df.iterrows():
        dataset.append({"question": row["question"], "expected_answer": row["expected_answer"]})

    # make sure to save it
    dataset.save()
    return dataset


# Define correctness metric
correctness_metric = DiscreteMetric(
    name="correctness",
    prompt="""Compare the model response to the expected answer and determine if it's correct.
    
    Consider the response correct if it:
    1. Contains the key information from the expected answer
    2. Is factually accurate based on the provided context
    3. Adequately addresses the question asked

    Return 'pass' if the response is correct, 'fail' if it's incorrect.

    Question: {question}
    Expected Answer: {expected_answer}
    Model Response: {response}

    Evaluation:""",
    allowed_values=["pass", "fail"],
)

@experiment()
async def run_experiment(row: Dict[str, Any]):
    try:
        llmGroq = ChatGroq(
            model_name="llama-3.1-8b-instant",
            temperature=0.7
            )
        llm = Config.get_llm()
        doc_processor = DocumentProcessor(
                   chunk_size=Config.CHUNK_SIZE,
                   chunk_overlap=Config.CHUNK_OVERLAP
              )
        vector_store = VectorStore()

        # Use default URLs
        urls = Config.DEFAULT_URLS
        
        documents = doc_processor.process_urls(urls)
        # Load the index
        vector_store.create_vectorstore(documents)
        
        # Build graph
        graph_builder = GraphBuilder(
                retriever=vector_store.get_retriever(),
                llm=llmGroq
            )
        graph_builder.build()

    except Exception as e:
                print(f"Error initializing RAG system: {e}")
                return None, 0
        
    logger.info("Naive RAG system initialized!")
    
    # Query the RAG system
    question = row['question']

    # Query the RAG system
    rag_response = await graph_builder.arun(question)
    model_response = rag_response.get('answer')

    # Evaluate correctness asynchronously
    score = await correctness_metric.ascore(
        question=question,
        expected_answer=row["expected_answer"],
        response=model_response,
        llm=llm_factory('gpt-4o-mini', client=openai_client)
    )

    scorerContextPrecision = ContextPrecision(llm=llm_factory('gpt-4o-mini', client=openai_client))

    scoreContextPrecision = await scorerContextPrecision.ascore(
        user_input=question,
        reference=row["expected_answer"],
        retrieved_contexts=[doc.page_content for doc in rag_response.get("retrieved_docs", [])]
    )

    # Create metric
    scorerContextRecall = ContextRecall(llm=llm_factory('gpt-4o-mini', client=openai_client))

    # Evaluate
    scoreContextRecall = await scorerContextRecall.ascore(
        user_input=question,
        retrieved_contexts=[doc.page_content for doc in rag_response.get("retrieved_docs", [])],
        reference=row["expected_answer"]
    )

    scorerFaithfulness = Faithfulness(llm=llm_factory('gpt-4o-mini', client=openai_client))

    scoreFaithfulness = await scorerFaithfulness.ascore(
        user_input=question,
        response=model_response,
        retrieved_contexts=[doc.page_content for doc in rag_response.get("retrieved_docs", [])]
    )

    embeddings = embedding_factory("openai", model="text-embedding-3-small", client=openai_client)
    # Create metric
    scorerAnswerRelevancy = AnswerRelevancy(llm=llm_factory('gpt-4o-mini', client=openai_client), embeddings=embeddings)

    scoreAnswerRelevancy = await scorerAnswerRelevancy.ascore(
        user_input=question,
        response=model_response,
    )

    scorerNoiseSensitivity = NoiseSensitivity(llm=llm_factory('gpt-4o-mini', client=openai_client))

    # Evaluate
    scoreNoiseSensitivity = await scorerNoiseSensitivity.ascore(
        user_input=question,
        response=model_response,
        reference=row["expected_answer"],
        retrieved_contexts=[doc.page_content for doc in rag_response.get("retrieved_docs", [])]
    )

    experiment_view = {
        **row,
        "response": model_response,
        "correctness": 1 if score.value == "pass" else 0,
        "context_precision": scoreContextPrecision.value,
        "faithfulness": scoreFaithfulness.value,
        "context_Recall": scoreContextRecall.value,
        "answer_relevancy": scoreAnswerRelevancy.value,
        "noise_sensitivity": scoreNoiseSensitivity.value
    }
    return experiment_view


async def main():
    dataset = load_dataset(download_and_save_dataset())
    print("dataset loaded successfully", dataset)
    experiment_results = await run_experiment.arun(dataset)
    print("Experiment completed successfully!")
    print("Experiment results:", experiment_results)

    # Save experiment results to CSV
    experiment_results.save()
    csv_path = Path(".") / "experiments" / f"{experiment_results.name}.csv"
    print(f"\nExperiment results saved to: {csv_path.resolve()}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
