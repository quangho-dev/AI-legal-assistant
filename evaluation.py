from langchain_openai import ChatOpenAI
from langsmith import Client, traceable
from typing_extensions import Annotated, TypedDict
from dotenv import load_dotenv
import os
from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from data.examples_for_eval import EXAMPLES_FOR_EVAL

load_dotenv()

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

doc_processor = DocumentProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )

vector_store = VectorStore()
        
        # Use default URLs
urls = Config.DEFAULT_URLS
        
        # Process documents
documents = doc_processor.process_urls(urls)
        
        # Create vector store
vector_store.create_vectorstore(documents)

retriever = vector_store.get_retriever()

llm = ChatOpenAI(model="gpt-5.4", temperature=1)

# Add decorator so this function is traced in LangSmith
@traceable()
def rag_bot(question: str) -> dict:
    # langchain Retriever will be automatically traced
    docs = retriever.invoke(question)
    docs_string = "".join(doc.page_content for doc in docs)
    instructions = f"""You are a helpful assistant who is good at analyzing source information and answering questions.
       Use the following source documents to answer the user's questions.
       If you don't know the answer, just say that you don't know.
       Use three sentences maximum and keep the answer concise.

Documents:
{docs_string}"""
    # langchain ChatModel will be automatically traced
    ai_msg = llm.invoke([
            {"role": "system", "content": instructions},
            {"role": "user", "content": question},
        ],
    )
    return {"answer": ai_msg.content, "documents": docs}

client = Client()

# Define the examples for the dataset
examples = [
  {
    "inputs": {
      "question": "Những nguyên tắc cơ bản của pháp luật dân sự được quy định trong Bộ luật dân sự 2015 là gì?"
    },
    "outputs": {
      "answer": "Các nguyên tắc gồm: (1) Bình đẳng, không phân biệt đối xử; (2) Tự do, tự nguyện cam kết, thỏa thuận; (3) Thiện chí, trung thực; (4) Không xâm phạm lợi ích quốc gia, công cộng và quyền lợi người khác; (5) Tự chịu trách nhiệm về nghĩa vụ dân sự."
    }
  },
  {
    "inputs": {
      "question": "Việc áp dụng Bộ luật dân sự được quy định như thế nào trong Bộ luật dân sự năm 2015?"
    },
    "outputs": {
      "answer": "Bộ luật dân sự là luật chung điều chỉnh quan hệ dân sự; luật khác không được trái nguyên tắc cơ bản; nếu không có quy định hoặc trái nguyên tắc thì áp dụng BLDS; nếu có điều ước quốc tế thì áp dụng điều ước quốc tế."
    }
  },
  {
    "inputs": {
      "question": "X cho Y thuê nhà nhưng Y không trả sau khi hết hạn. X có thể làm gì?"
    },
    "outputs": {
      "answer": "X có thể tự bảo vệ hoặc yêu cầu cơ quan có thẩm quyền: buộc trả nhà, chấm dứt vi phạm, bồi thường thiệt hại, hoặc giải quyết qua thương lượng, hòa giải, trọng tài hoặc tòa án."
    }
  },
  {
    "inputs": {
      "question": "Người nghiện ma túy có thể bị hạn chế năng lực hành vi dân sự không?"
    },
    "outputs": {
      "answer": "Có. Nếu nghiện ma túy dẫn đến phá tán tài sản thì theo yêu cầu của người liên quan, Tòa án có thể tuyên bố hạn chế năng lực hành vi dân sự."
    }
  },
  {
    "inputs": {
      "question": "Người bị mờ mắt có bị mất năng lực hành vi dân sự không?"
    },
    "outputs": {
      "answer": "Không. Chỉ người bị bệnh tâm thần hoặc không làm chủ hành vi mới bị coi là mất năng lực hành vi dân sự. Người bị mờ mắt vẫn có thể thực hiện giao dịch."
    }
  },
  {
    "inputs": {
      "question": "Có thể thay đổi dân tộc không?"
    },
    "outputs": {
      "answer": "Có. Cá nhân có quyền xác định lại dân tộc theo cha hoặc mẹ theo quy định pháp luật và phải thực hiện thủ tục tại cơ quan có thẩm quyền."
    }
  },
  {
    "inputs": {
      "question": "Việc bóc thư người khác có hợp pháp không?"
    },
    "outputs": {
      "answer": "Không. Thư tín là bí mật cá nhân được pháp luật bảo vệ. Việc bóc thư khi chưa được phép là vi phạm pháp luật."
    }
  },
  {
    "inputs": {
      "question": "Ai là người giám hộ cho người chưa thành niên không còn cha mẹ?"
    },
    "outputs": {
      "answer": "Theo thứ tự: anh/chị ruột, ông bà, cô dì chú bác. Nếu không có thì UBND cấp xã cử người giám hộ."
    }
  },
  {
    "inputs": {
      "question": "Ai có trách nhiệm nuôi dưỡng người mất năng lực hành vi dân sự?"
    },
    "outputs": {
      "answer": "Nếu là vợ/chồng thì người còn lại là giám hộ. Nếu không đủ điều kiện thì xét đến con hoặc cha mẹ theo quy định pháp luật."
    }
  },
  {
    "inputs": {
      "question": "Nếu cây nhà hàng xóm gây thiệt hại thì có phải bồi thường không?"
    },
    "outputs": {
      "answer": "Có. Chủ sở hữu cây phải chịu trách nhiệm chặt bỏ nếu nguy hiểm và bồi thường nếu gây thiệt hại."
    }
  }
]

# Create the dataset and examples in LangSmith
dataset_name = "Q&A luật dân sự 2015"
if not client.has_dataset(dataset_name=dataset_name):
    dataset = client.create_dataset(dataset_name=dataset_name)
    client.create_examples(
        dataset_id=dataset.id,
        examples=EXAMPLES_FOR_EVAL
    )

# Grade output schema
class CorrectnessGrade(TypedDict):
    # Note that the order in the fields are defined is the order in which the model will generate them.
    # It is useful to put explanations before responses because it forces the model to think through
    # its final response before generating it:
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    correct: Annotated[bool, ..., "True if the answer is correct, False otherwise."]

# Grade prompt
correctness_instructions = """You are a teacher grading a quiz. You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER. Here is the grade criteria to follow:
(1) Grade the student answers based ONLY on their factual accuracy relative to the ground truth answer. (2) Ensure that the student answer does not contain any conflicting statements.
(3) It is OK if the student answer contains more information than the ground truth answer, as long as it is factually accurate relative to the  ground truth answer.

Correctness:
A correctness value of True means that the student's answer meets all of the criteria.
A correctness value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

# Grader LLM
grader_llm = ChatOpenAI(model="gpt-5.4", temperature=0).with_structured_output(
    CorrectnessGrade, method="json_schema", strict=True
)

def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    """An evaluator for RAG answer accuracy"""
    answers = f"""\
QUESTION: {inputs['question']}
GROUND TRUTH ANSWER: {reference_outputs['answer']}
STUDENT ANSWER: {outputs['answer']}"""
    # Run evaluator
    grade = grader_llm.invoke([
            {"role": "system", "content": correctness_instructions},
            {"role": "user", "content": answers},
        ]
    )
    return grade["correct"]

# Grade output schema
class RelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[
        bool, ..., "Provide the score on whether the answer addresses the question"
    ]

# Grade prompt
relevance_instructions = """You are a teacher grading a quiz. You will be given a QUESTION and a STUDENT ANSWER. Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is concise and relevant to the QUESTION
(2) Ensure the STUDENT ANSWER helps to answer the QUESTION

Relevance:
A relevance value of True means that the student's answer meets all of the criteria.
A relevance value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

# Grader LLM
relevance_llm = ChatOpenAI(model="gpt-5.4", temperature=0).with_structured_output(
    RelevanceGrade, method="json_schema", strict=True
)

# Evaluator
def relevance(inputs: dict, outputs: dict) -> bool:
    """A simple evaluator for RAG answer helpfulness."""
    answer = f"QUESTION: {inputs['question']}\nSTUDENT ANSWER: {outputs['answer']}"
    grade = relevance_llm.invoke([
            {"role": "system", "content": relevance_instructions},
            {"role": "user", "content": answer},
        ]
    )
    return grade["relevant"]

# Grade output schema
class GroundedGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    grounded: Annotated[
        bool, ..., "Provide the score on if the answer hallucinates from the documents"
    ]

# Grade prompt
grounded_instructions = """You are a teacher grading a quiz. You will be given FACTS and a STUDENT ANSWER. Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is grounded in the FACTS. (2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

Grounded:
A grounded value of True means that the student's answer meets all of the criteria.
A grounded value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

# Grader LLM
grounded_llm = ChatOpenAI(model="gpt-5.4", temperature=0).with_structured_output(
    GroundedGrade, method="json_schema", strict=True
)

# Evaluator
def groundedness(inputs: dict, outputs: dict) -> bool:
    """A simple evaluator for RAG answer groundedness."""
    doc_string = "\n\n".join(doc.page_content for doc in outputs["documents"])
    answer = f"FACTS: {doc_string}\nSTUDENT ANSWER: {outputs['answer']}"
    grade = grounded_llm.invoke([
            {"role": "system", "content": grounded_instructions},
            {"role": "user", "content": answer},
        ]
    )
    return grade["grounded"]

# Grade output schema
class RetrievalRelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[
        bool,
        ...,
        "True if the retrieved documents are relevant to the question, False otherwise",
    ]

# Grade prompt
retrieval_relevance_instructions = """You are a teacher grading a quiz. You will be given a QUESTION and a set of FACTS provided by the student. Here is the grade criteria to follow:
(1) You goal is to identify FACTS that are completely unrelated to the QUESTION
(2) If the facts contain ANY keywords or semantic meaning related to the question, consider them relevant
(3) It is OK if the facts have SOME information that is unrelated to the question as long as (2) is met

Relevance:
A relevance value of True means that the FACTS contain ANY keywords or semantic meaning related to the QUESTION and are therefore relevant.
A relevance value of False means that the FACTS are completely unrelated to the QUESTION.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset."""

# Grader LLM
retrieval_relevance_llm = ChatOpenAI(
    model="gpt-5.4", temperature=0
).with_structured_output(RetrievalRelevanceGrade, method="json_schema", strict=True)

def retrieval_relevance(inputs: dict, outputs: dict) -> bool:
    """An evaluator for document relevance"""
    doc_string = "\n\n".join(doc.page_content for doc in outputs["documents"])
    answer = f"FACTS: {doc_string}\nQUESTION: {inputs['question']}"
    # Run evaluator
    grade = retrieval_relevance_llm.invoke([
            {"role": "system", "content": retrieval_relevance_instructions},
            {"role": "user", "content": answer},
        ]
    )
    return grade["relevant"]

def target(inputs: dict) -> dict:
    return rag_bot(inputs["question"])

experiment_results = client.evaluate(
    target,
    data=dataset_name,
    evaluators=[correctness, groundedness, relevance, retrieval_relevance],
    experiment_prefix="rag-doc-relevance",
    metadata={"version": "LCEL context, gpt-4-0125-preview"},
)

# Explore results locally as a dataframe if you have pandas installed
# experiment_results.to_pandas()