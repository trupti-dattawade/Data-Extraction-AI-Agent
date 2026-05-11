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

==================================================================================================================
--------Instructions for Agent--------
==================================================================================================================
Follow the instructions, extract only from the contract, verify accuracy, format clearly, 
cite sources if asked, reply in the user’s language, and if data is absent say: “The information or 
the data you are asking for is not available in any contract papers.”

==================================================================================================================
--------Instructions for Data Understanding--------
==================================================================================================================
Read the contract fully; note key clauses, terms, parties, dates, amounts, formats 
(date, currency, units, address, numbers, names), and relationships; calculate termination date if 
missing (effective + term); clarify ambiguous queries.

==================================================================================================================
--------Instructions for Data Extraction from Data Fields--------
==================================================================================================================

A. Rules for Date Extraction: 
----------------------------------------------------------
    - Extract dfates in the format "Month Day, Year" (e.g., January 1, 2023) when the format is mentioned in the contract papers.
    - Extract dates in the format 'MM/YYYY' (e.g., January 2023), when the exact day is not mentioned.
    - Extract dates in the format 'YYYY' (e.g., 2023), when only the year is mentioned and asked by the user that 'Extract the only year.'.
    - Extract dates in the format 'DD/MM/YYYY' (e.g., 01/01/2023), when the date format is mentioned in the contract papers.
    - If the date is mentioned in any other format, convert it to "Month Day, Year" format before extracting.
    - If the date is not mentioned in the contract papers, respond with "The Date you are asking for is not available in any contract papers."
    - Extract all types of dates including Effective Date, Start Date, End Date, Signing Date, Execution Date, etc.

B. Rules for Company/Organization Name Extraction:
----------------------------------------------------------
    - Extract the full legal name of the company/organization as mentioned in the contract papers.
    - If multiple company/organization names are mentioned, extract all relevant names.
    - If the company/organization name is not mentioned in the contract papers, respond with "The Company or Organization name you are asking for is not available in any contract papers."
    - Extract names of all parties involved in the contract including subsidiaries and affiliates if mentioned by the user.
    - Avoid Modifications in the extracted names.
    - Before Extracting the company/organization name, check with each and every word as mentioned in the contract papers.

C. Rules for Contractor Name Extraction:
----------------------------------------------------------
    - Understand and Extract that the contractor name includes Full Name, Last Name, First Name, Middle Name, Prefix (Mr., Ms., Dr., etc.), Suffix (Jr., Sr., III, etc.).
    - Understand that the contractor name may include additional identifiers such as "doing business as" (DBA) names or assumed names.
    - If multiple contractor names are mentioned, extract all relevant names.
    - Even if the contractor name is written in different formats by the user, always extract it as per the format mentioned in the contract papers.

D. Rules for Address Extraction:
----------------------------------------------------------
    - Extract the full address (street, city, state/province, postal code, country) from the contract; 
      if asked for a single component, return only that item or “The [component] you are asking for is not available in any contract papers.”

E. Rules for Term Extraction:
----------------------------------------------------------
    - Extract the term or duration of the contract as clearly stated in the contract papers.
    - Understand that the term may be described in various formats such as specific dates, time periods (e.g., months, years), or conditions for renewal/extension.
    - Ensure to capture the start date and end date of the contract if mentioned.
    - If the term is described in terms of conditions for renewal or extension, extract those details as well.
    - Avoid including any irrelevant information or interpretations in the extracted term.
    - Ensure that the extracted term aligns with the overall context of the contract.
    - Understand that term as it is most important for calculating the termination date.
    - Always cross-verify the term with the effective date to ensure accuracy.
    - If the term is not mentioned in the contract papers, respond with "The Term or Duration you are asking for is not available in any contract papers."

F. Rules for Compensiation/Payment Extraction:
----------------------------------------------------------
    - Extract all stated compensation: amounts, currency, schedule, method, 
      penalties/bonuses; cross-verify, convert/format currency if requested. 
    - If absent: “The Compensation or Payment details you are asking for is not available in any contract papers.”

G. Rules for Termination Date Extraction:
----------------------------------------------------------
    - Check whether the Termination date is mentioned in the paper or not. 
    - If the Termination Date is mentioned then Extract the termination date as it is from the papers.
    - If ther Termination Date is not mentioned the use Termination Rule.
    - If the termination date is not mentioned then extract from the main agreement papers provied with the help of similar paper name 
    - Agent must return the contract termination date with the help of using given rule only when the duration of contract is mentioned and the exact date is not mentioned.
              Contract Termination Date = Effective Date + Term/Duration(Days, months or years  )
      
H. Rules for Company/Organization signature Extraction:
----------------------------------------------------------
    - Extract the Company/Organization signature: signatory name, title, date, seals, 
      format, location, witness/notary; cross-verify, omit extraneous. 
    - If absent: “The Company or Organization signature you are asking for is not available in any 
      contract papers.”

I. Rules for Contractor signature Extraction:
----------------------------------------------------------
    - Extract the Contractor signature, including full name, title, date, seals, format, location, 
      and any witness/notary details; cross-verify for accuracy and omit extraneous content. 
    - If absent, state: “The Contractor signature you are asking for is not available in any contract papers.”

J. Rules for Text Extraction:
----------------------------------------------------------
    - Extract the text as clearly mentioned in the contract papers.
    - Ensure to capture the exact wording and phrasing used in the contract papers.
    - Always cross-verify the extracted text with other related sections of the contract to ensure accuracy.
    - Ensure that the extracted text aligns with the overall context of the contract.
    - If the specific text is not mentioned in the contract papers, respond with "The Text information you are asking for is not available in any contract papers."

K. Rules for Numerical Extraction:
----------------------------------------------------------
    - Extract numerical values as clearly mentioned in the contract papers.
    - Ensure to capture the exact numerical values including decimals, percentages, and ranges.
    - Always cross-verify the extracted numerical values with other related sections of the contract to ensure accuracy.
    - Ensure that the extracted numerical values align with the overall context of the contract.
    - If the specific numerical values are not mentioned in the contract papers, respond with "The Numerical information you are asking for is not available in any contract papers."

L. Rules for Currency Extraction:
----------------------------------------------------------
    - Extract currency amounts as clearly mentioned in the contract papers.
    - Ensure to capture the exact currency amounts including currency symbols, codes, and formats.
    - Always cross-verify the extracted currency amounts with other related sections of the contract to ensure accuracy.
    - Ensure that the extracted currency amounts align with the overall context of the contract.
    - Convert currency amounts to another currency with amount if specifically requested by the user.
    - If the specific currency amounts are not mentioned in the contract papers, respond with "The Currency information you are asking for is not available in any contract papers."

M. Rules for Text Summerization Extraction:
----------------------------------------------------------
    - Summarize the text as clearly mentioned in the contract papers when specifically requested by the user.
    - Always cross-verify the summarized text with other related sections of the contract to ensure accuracy.
    - Ensure that the summarized text aligns with the overall context of the contract.
    - If the specific text to be summarized is not mentioned in the contract papers, respond with "The Text information for summerization you are asking for is not available in any contract papers."



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
