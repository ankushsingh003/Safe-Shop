
"""
ml/rag/fraud_rag.py
────────────────────
RAG (Retrieval Augmented Generation) Layer for Safe-Shop.

What it does:
  1. Every time LangGraph agent completes an investigation → store the
     full case (order details + evidence + reasoning + decision) as a
     document in ChromaDB with semantic embeddings.

  2. When an analyst calls POST /ask with a natural language question,
     ChromaDB finds the most semantically similar past cases, then
     GPT-4o-mini synthesizes a human-readable answer grounded in
     those real cases.

Example queries the analyst can ask:
  - "Show all cases where velocity triggered a block"
  - "Which fraud pattern is most common in Electronics?"
  - "Find cases where the agent approved but score was above 0.6"
  - "What reasoning did the agent use for stolen card cases?"

Environment variables required:
  OPENAI_API_KEY=sk-...   ← same key used by fraud_investigator.py
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma

logger = logging.getLogger("fraud_rag")

# ── CONFIG ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY   = os.environ.get("OPENAI_API_KEY", "")
CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME  = "fraud_investigations"
TOP_K_RESULTS    = 5    # how many similar cases to retrieve per query


# ── INITIALIZE CHROMADB ───────────────────────────────────────────────────────
def get_chroma_client() -> chromadb.PersistentClient:
    """
    Returns a persistent ChromaDB client.
    Data is saved to disk at CHROMA_PERSIST_DIR so it survives restarts.
    """
    return chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False)
    )


def get_collection() -> chromadb.Collection:
    """Get or create the fraud investigations collection."""
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}   # cosine similarity for text embeddings
    )


# ── EMBEDDING FUNCTION ────────────────────────────────────────────────────────
def get_embeddings():
    """
    Returns OpenAI text-embedding-3-small.
    Falls back to a simple hash-based mock if no API key set.
    Cost: ~$0.00002 per document — essentially free.
    """
    if OPENAI_API_KEY:
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=OPENAI_API_KEY
        )
    return None


# ── STORE A FRAUD CASE ────────────────────────────────────────────────────────
def store_fraud_case(
    order_id: str,
    order_amount: float,
    category: str,
    device_type: str,
    location_mismatch: bool,
    orders_per_user_1m: int,
    champion_score: float,
    gnn_score: float,
    decision: str,
    reasoning: str,
    evidence: List[str],
    risk_level: str,
    fraud_pattern: Optional[str] = None,
) -> bool:
    """
    Called automatically by FastAPI after every LangGraph investigation.
    Converts the fraud case into a rich text document and stores it
    in ChromaDB with metadata for filtering.

    Returns True on success, False on failure (non-blocking).
    """
    try:
        collection = get_collection()
        embeddings = get_embeddings()

        # ── BUILD DOCUMENT TEXT ───────────────────────────────────────────
        # This is what gets embedded — rich, descriptive, searchable text
        evidence_text = " | ".join(evidence) if evidence else "No evidence collected"

        document = f"""
Fraud Investigation Case
Order ID: {order_id}
Decision: {decision}
Risk Level: {risk_level}
Fraud Pattern: {fraud_pattern or 'UNKNOWN'}

Order Details:
- Amount: ${order_amount:.2f}
- Category: {category}
- Device: {device_type}
- Location Mismatch: {location_mismatch}
- Velocity (orders/min): {orders_per_user_1m}

Model Scores:
- Ensemble Champion Score: {champion_score:.4f}
- GNN Ring Detection Score: {gnn_score:.4f}

Evidence Signals:
{evidence_text}

Agent Reasoning:
{reasoning}
""".strip()

        # ── METADATA (for filtered queries) ──────────────────────────────
        metadata = {
            "order_id":           order_id,
            "decision":           decision,
            "risk_level":         risk_level,
            "fraud_pattern":      fraud_pattern or "UNKNOWN",
            "champion_score":     round(champion_score, 4),
            "gnn_score":          round(gnn_score, 4),
            "order_amount":       round(order_amount, 2),
            "category":           category,
            "device_type":        device_type,
            "location_mismatch":  str(location_mismatch),
            "orders_per_user_1m": orders_per_user_1m,
            "timestamp":          datetime.utcnow().isoformat(),
        }

        # ── EMBED AND STORE ───────────────────────────────────────────────
        if embeddings:
            # Use OpenAI embeddings for semantic search
            embedding_vector = embeddings.embed_query(document)
            collection.add(
                ids=[order_id],
                documents=[document],
                metadatas=[metadata],
                embeddings=[embedding_vector]
            )
        else:
            # ChromaDB's built-in embeddings (no API key needed)
            collection.add(
                ids=[order_id],
                documents=[document],
                metadatas=[metadata]
            )

        logger.info(f"✅ RAG: Stored fraud case {order_id} in ChromaDB")
        return True

    except Exception as e:
        logger.error(f"❌ RAG: Failed to store case {order_id}: {e}")
        return False


# ── QUERY FRAUD CASES (RAG RETRIEVAL) ────────────────────────────────────────
def retrieve_similar_cases(
    query: str,
    top_k: int = TOP_K_RESULTS,
    filter_decision: Optional[str] = None,
    filter_risk_level: Optional[str] = None,
    filter_pattern: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Semantic search over all stored fraud investigations.
    Returns the top_k most semantically similar cases.

    Optional metadata filters:
      filter_decision   → 'FINAL_BLOCK', 'FINAL_APPROVE', 'HUMAN_REVIEW'
      filter_risk_level → 'CRITICAL', 'HIGH', 'LOW'
      filter_pattern    → 'BOT_ACTIVITY', 'STOLEN_CARD', etc.
    """
    try:
        collection = get_collection()
        embeddings = get_embeddings()

        # Build ChromaDB where clause for metadata filters
        where_clause = {}
        if filter_decision:
            where_clause["decision"] = filter_decision
        if filter_risk_level:
            where_clause["risk_level"] = filter_risk_level
        if filter_pattern:
            where_clause["fraud_pattern"] = filter_pattern

        query_kwargs = {
            "query_texts": [query],
            "n_results": min(top_k, max(1, collection.count())),
            "include": ["documents", "metadatas", "distances"]
        }

        if where_clause:
            query_kwargs["where"] = where_clause

        if embeddings:
            # Use same OpenAI embedding model for query
            query_embedding = embeddings.embed_query(query)
            query_kwargs.pop("query_texts")
            query_kwargs["query_embeddings"] = [query_embedding]

        results = collection.query(**query_kwargs)

        # Format results into clean dicts
        cases = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ):
                cases.append({
                    "document":   doc,
                    "metadata":   meta,
                    "similarity": round(1 - dist, 4)  # convert distance to similarity
                })

        logger.info(f"✅ RAG: Retrieved {len(cases)} cases for query: '{query[:50]}...'")
        return cases

    except Exception as e:
        logger.error(f"❌ RAG: Retrieval failed: {e}")
        return []


# ── GENERATE ANSWER (RAG SYNTHESIS) ──────────────────────────────────────────
RAG_SYSTEM_PROMPT = """You are a fraud intelligence analyst at Safe-Shop.
You have access to a database of past fraud investigations from the AI agent.
Answer the analyst's question based ONLY on the fraud cases provided.

Rules:
- Be specific and reference actual case data (order IDs, scores, patterns)
- If you see trends across cases, highlight them
- If the cases don't contain enough information, say so clearly
- Format your answer in clear sections if it's a complex query
- Keep the answer concise but complete"""


def answer_fraud_query(
    question: str,
    top_k: int = TOP_K_RESULTS,
    filter_decision: Optional[str] = None,
    filter_risk_level: Optional[str] = None,
    filter_pattern: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Full RAG pipeline:
      1. Retrieve semantically similar fraud cases from ChromaDB
      2. Pass cases + question to GPT-4o-mini
      3. Return synthesized answer + source cases

    This is what POST /ask calls.
    """
    # Step 1: Retrieve relevant cases
    cases = retrieve_similar_cases(
        query=question,
        top_k=top_k,
        filter_decision=filter_decision,
        filter_risk_level=filter_risk_level,
        filter_pattern=filter_pattern,
    )

    if not cases:
        return {
            "answer": (
                "No fraud investigation cases found in the database yet. "
                "Cases are stored automatically after each LangGraph agent investigation. "
                "Run the fraud detection pipeline first to populate the knowledge base."
            ),
            "cases_retrieved": 0,
            "source_cases": [],
        }

    # Step 2: Build context from retrieved cases
    context_blocks = []
    for i, case in enumerate(cases, 1):
        context_blocks.append(
            f"--- Case {i} (Similarity: {case['similarity']:.2f}) ---\n"
            f"{case['document']}"
        )
    context = "\n\n".join(context_blocks)

    # Step 3: Call GPT-4o-mini for synthesis
    if not OPENAI_API_KEY:
        # Fallback: return raw cases without LLM synthesis
        summary = f"Found {len(cases)} relevant cases (LLM synthesis unavailable — set OPENAI_API_KEY):\n\n"
        for case in cases:
            meta = case["metadata"]
            summary += (
                f"• Order {meta['order_id']}: {meta['decision']} | "
                f"Pattern: {meta['fraud_pattern']} | "
                f"Score: {meta['champion_score']} | "
                f"Risk: {meta['risk_level']}\n"
            )
        return {
            "answer": summary,
            "cases_retrieved": len(cases),
            "source_cases": [c["metadata"] for c in cases],
        }

    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=500,
            api_key=OPENAI_API_KEY,
            timeout=15,
        )

        user_message = f"""Analyst Question: {question}

Retrieved Fraud Cases:
{context}

Please answer the analyst's question based on these cases."""

        response = llm.invoke([
            SystemMessage(content=RAG_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ])

        return {
            "answer":          response.content.strip(),
            "cases_retrieved": len(cases),
            "source_cases":    [c["metadata"] for c in cases],
        }

    except Exception as e:
        logger.error(f"❌ RAG: LLM synthesis failed: {e}")
        return {
            "answer":          f"Retrieved {len(cases)} cases but LLM synthesis failed: {e}",
            "cases_retrieved": len(cases),
            "source_cases":    [c["metadata"] for c in cases],
        }


# ── STATS ─────────────────────────────────────────────────────────────────────
def get_rag_stats() -> Dict[str, Any]:
    """Returns stats about the ChromaDB collection. Called by GET /health."""
    try:
        collection = get_collection()
        count = collection.count()
        return {
            "total_cases_stored": count,
            "collection_name":    COLLECTION_NAME,
            "persist_directory":  CHROMA_PERSIST_DIR,
            "embeddings_model":   "text-embedding-3-small" if OPENAI_API_KEY else "chroma-default",
            "status":             "healthy"
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}