#==================================================
#  import env file
#==================================================
from dotenv import load_dotenv
import os
load_dotenv()
PDF_FOLDER_PATH= os.getenv("pdf_folder_path")
VECTOR_DB = os.getenv("vector_db_path")
CONTEXT_JSON_PATH = os.getenv("context_json_path")
api_key=os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is not set in environment variables")
print("==================================================")
print("Groq API key loaded successfully")
print("==================================================")

#==================================================
#  import libraries
#==================================================
import os
import json
import warnings
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter # type: ignore
from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore
from langchain_community.vectorstores import FAISS  # type: ignore
from langchain.chains import RetrievalQA # pyright: ignore[reportMissingImports]
from langchain.prompts import PromptTemplate # pyright: ignore[reportMissingImports]
from langchain_groq import ChatGroq

from agent_tools import apply_all_filters

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

#---------------------------------------------------------
# Load environment variables
#---------------------------------------------------------
load_dotenv()


#---------------------------------------------------------
# Load Agent Context (JSON)
#---------------------------------------------------------
with open(CONTEXT_JSON_PATH, "r", encoding="utf-8") as f: # type: ignore
    AGENT_CONTEXT = json.load(f)


def format_agent_context(context_json: dict) -> str:
    """Convert structured JSON context into LLM-readable text."""
    lines = []

    lines.append(f"Agent Role: {context_json.get('agent_role', 'Contract Data Extraction Agent')}")

    if "rules" in context_json:
        lines.append("\nGlobal Rules:")
        for rule in context_json["rules"]:
            lines.append(f"- {rule}")

    if "absence_message" in context_json:
        lines.append("\nAbsence Message:")
        lines.append(context_json["absence_message"])

    if "date_rules" in context_json:
        lines.append("\nDate Rules:")
        for fmt in context_json["date_rules"].get("formats", []):
            lines.append(f"- {fmt}")

    return "\n".join(lines)


AGENT_CONTEXT_TEXT = format_agent_context(AGENT_CONTEXT)

#---------------------------------------------------------
# Load PDFs
#---------------------------------------------------------
loader = PyPDFDirectoryLoader(PDF_FOLDER_PATH) # type: ignore
documents = loader.load()

for doc in documents:
    doc.metadata["source_file"] = doc.metadata.get("source", "unknown")

#---------------------------------------------------------
# Chunking
#---------------------------------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)

#---------------------------------------------------------
# Embeddings
#---------------------------------------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

#---------------------------------------------------------
# Vector Store
#---------------------------------------------------------
vector_db = FAISS.from_documents(chunks, embedding_model)
vector_db.save_local(VECTOR_DB_PATH) # type: ignore

#---------------------------------------------------------
# Retriever
#---------------------------------------------------------
retriever = vector_db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

#---------------------------------------------------------
# LLM (Groq)
#---------------------------------------------------------
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),  # type: ignore
    model_name="llama-3.1-8b-instant",  # type: ignore
    temperature=0
)

#---------------------------------------------------------
# Prompt Template (Prompt Engineering)
#---------------------------------------------------------
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""

==================================================================================================================
Role : 
==================================================================================================================
You are a Contract Data Extraction Agent specialized in extracting and analyzing data from contract papers.
You are strictly instructed to follow the instructions provided below for data understanding, extraction, analysis
and output formatting.



================ AGENT CONTEXT ================
{context}

================ USER QUESTION ================
{question}

================ ANSWER ================
"""
)

#---------------------------------------------------------
# Extraction Function
#---------------------------------------------------------
def extract_data(user_query: str, filename: str | None = None) -> str:

    filtered_docs = apply_all_filters(
        documents=chunks,
        retriever=retriever,
        query=user_query,
        vectorstore=vector_db,
        filename=filename,
        top_k=5,
        alpha=0.5
    )

    if not filtered_docs:
        return AGENT_CONTEXT.get(
            "absence_message",
            "The information or the data you are asking for is not available in any contract papers."
        )

    contract_text = "\n".join(doc.page_content for doc in filtered_docs)

    full_context = (
        AGENT_CONTEXT_TEXT
        + "\n\n================ CONTRACT TEXT ================\n\n"
        + contract_text
    )

    prompt_input = prompt.format(
        context=full_context,
        question=user_query
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain.run(prompt_input)

#---------------------------------------------------------
# Interactive Client
#---------------------------------------------------------
if __name__ == "__main__":

    print("======================================================")
    print("Contract Data Extraction Agent (Refactored) is ready")
    print("======================================================")

    while True:
        query = input("Enter your query (or type 'exit'): ")
        if query.lower() == "exit":
            break

        filename = None
        q = query.lower()
        if "paper 1" in q:
            filename = "Paper-1.pdf"
        elif "paper 2" in q:
            filename = "Paper-2.pdf"
        elif "paper 3" in q:
            filename = "Paper-3.pdf"

        result = extract_data(query, filename)
        print("\nResult:\n", result)
        print("======================================================")
