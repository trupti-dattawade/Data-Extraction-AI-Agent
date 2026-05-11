#---------------------------------------------------------
#                     agent_tools.py
#---------------------------------------------------------
import re
import numpy as np
from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer

#---------------------------------------------------------
# 1. Filename-Based Filtering
#---------------------------------------------------------

def filter_by_filename(documents: List, filename: Optional[str] = None) -> List:
    """Filter documents by filename metadata."""
    if filename:
        return [
            doc for doc in documents
            if filename.lower() in doc.metadata.get("source_file", "").lower()
        ]
    return documents

#---------------------------------------------------------
# 2. Keyword Presence Filtering
#---------------------------------------------------------

def filter_by_keywords(documents: List, query: str) -> List:
    """Filter documents by keyword presence."""
    query_keywords = [word.lower() for word in query.split() if len(word) > 2]
    filtered_docs = []

    for doc in documents:
        text = doc.page_content.lower()
        if any(keyword in text for keyword in query_keywords):
            filtered_docs.append(doc)

    return filtered_docs if filtered_docs else documents

#---------------------------------------------------------
# 3. Date and Number Pattern Filtering
#---------------------------------------------------------

def filter_by_date_number_pattern(documents: List) -> List:
    """Filter documents containing dates or year patterns."""
    pattern = r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\b\d{4}\b"
    filtered_docs = [
        doc for doc in documents
        if re.search(pattern, doc.page_content)
    ]
    return filtered_docs if filtered_docs else documents

#---------------------------------------------------------
# 4. Exact Phrase Matching
#---------------------------------------------------------

def filter_by_exact_phrase(documents: List, query: str) -> List:
    """Filter documents by exact quoted phrase matching."""
    exact_phrases = [
        phrase.lower()
        for phrase in query.split('"')
        if phrase.strip()
    ]

    if not exact_phrases:
        return documents

    filtered_docs = []
    for doc in documents:
        text = doc.page_content.lower()
        if any(phrase in text for phrase in exact_phrases):
            filtered_docs.append(doc)

    return filtered_docs if filtered_docs else documents

#---------------------------------------------------------
# Sparse Retrieval (TF-IDF)
#---------------------------------------------------------

def sparse_retrieval(documents: List, query: str, top_k: int = 5) -> List:
    """Keyword-based sparse retrieval using TF-IDF."""
    if not documents:
        return []

    texts = [doc.page_content for doc in documents]
    vectorizer = TfidfVectorizer(stop_words="english")

    doc_vectors = vectorizer.fit_transform(texts)
    query_vector = vectorizer.transform([query])

    scores = (doc_vectors @ query_vector.T).toarray().ravel() # type: ignore
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [
        (documents[i], float(scores[i]))
        for i in top_indices
        if scores[i] > 0
    ]

#---------------------------------------------------------
# Dense Retrieval (Embedding-based)
#---------------------------------------------------------

def dense_retrieval(vectorstore, query: str, top_k: int = 5) -> List:
    """Dense semantic retrieval from vector store."""
    if vectorstore is None:
        return []
    return vectorstore.similarity_search_with_score(query, k=top_k)

#---------------------------------------------------------
# Hybrid Retrieval
#---------------------------------------------------------

def hybrid_retrieval(
    documents: List,
    vectorstore,
    query: str,
    top_k: int = 5,
    alpha: float = 0.5
) -> List:
    """Hybrid sparse + dense retrieval."""
    sparse_results = sparse_retrieval(documents, query, top_k)
    dense_results = dense_retrieval(vectorstore, query, top_k)

    score_map = {}

    for doc, score in sparse_results:
        score_map[id(doc)] = {"doc": doc, "sparse": score, "dense": 0.0}

    for doc, score in dense_results:
        if id(doc) not in score_map:
            score_map[id(doc)] = {"doc": doc, "sparse": 0.0, "dense": score}
        else:
            score_map[id(doc)]["dense"] = score

    combined = []
    for v in score_map.values():
        combined_score = alpha * v["dense"] + (1 - alpha) * v["sparse"]
        combined.append((v["doc"], combined_score))

    combined.sort(key=lambda x: x[1], reverse=True)
    return combined[:top_k]

#---------------------------------------------------------
# Re-ranking
#---------------------------------------------------------

def rerank_documents(documents: List, query: str, top_k: int = 5) -> List:
    """Second-stage TF-IDF reranking."""
    if not documents:
        return []

    texts = [doc.page_content for doc in documents]
    vectorizer = TfidfVectorizer(stop_words="english")

    doc_vectors = vectorizer.fit_transform(texts)
    query_vector = vectorizer.transform([query])

    scores = (doc_vectors @ query_vector.T).toarray().ravel() # type: ignore
    ranked_indices = np.argsort(scores)[::-1][:top_k]

    return [
        (documents[i], float(scores[i]))
        for i in ranked_indices
        if scores[i] > 0
    ]

#---------------------------------------------------------
# Apply All Filters (Orchestrator)
#---------------------------------------------------------
def apply_all_filters(
    documents: List,
    retriever,
    query: str,
    vectorstore=None,
    filename: Optional[str] = None,
    top_k: int = 5,
    similarity_threshold: float = 0.75,
    alpha: float = 0.6
) -> List:
    """End-to-end document filtering and ranking pipeline."""

    docs = filter_by_filename(documents, filename)
    docs = filter_by_keywords(docs, query)
    docs = filter_by_date_number_pattern(docs)
    docs = filter_by_exact_phrase(docs, query)

    if vectorstore:
        hybrid_docs = hybrid_retrieval(
            documents=docs,
            vectorstore=vectorstore,
            query=query,
            top_k=top_k,
            alpha=alpha
        )
        docs = [doc for doc, _ in hybrid_docs]

    reranked = rerank_documents(docs, query, top_k)
    return [doc for doc, _ in reranked]

#---------------------------------------------------------
# End of agent_tools.py
#---------------------------------------------------------
