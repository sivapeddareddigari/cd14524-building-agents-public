# UdaPlay tools 
from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from tavily import TavilyClient

from lib.tooling import tool
from lib.udaplay_vector_store import UdaPlayVectorStore


_vector_store: Optional[UdaPlayVectorStore] = None


def configure_udaplay_tools(vector_store: UdaPlayVectorStore) -> None:
    
    global _vector_store
    _vector_store = vector_store


def _require_vector_store() -> UdaPlayVectorStore:
    
    if _vector_store is None:
        raise RuntimeError(
            "UdaPlay tools are not configured. "
            "Call configure_udaplay_tools(vector_store) before using the tools."
        )
    return _vector_store


@tool
def retrieve_game(query: str, top_k: int = 5) -> Dict[str, Any]:
    
    store = _require_vector_store()
    results = store.search_games(query=query, top_k=top_k)

    return {
        "tool": "retrieve_game",
        "query": query,
        "top_k": top_k,
        "results": results,
        "source": "local_vector_db",
    }


@tool
def evaluate_retrieval(query: str, retrieved_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    
    query_lower = query.lower()

    time_sensitive_terms = [
        "right now",
        "currently",
        "latest",
        "recent",
        "working on",
        "upcoming",
        "announced",
        "today",
        "this year",
    ]

    if any(term in query_lower for term in time_sensitive_terms):
        return {
            "tool": "evaluate_retrieval",
            "is_sufficient": False,
            "confidence": "low",
            "reason": "The query appears time-sensitive, so local static data may be outdated.",
            "missing_information": ["current or recent information"],
        }

    if not retrieved_results:
        return {
            "tool": "evaluate_retrieval",
            "is_sufficient": False,
            "confidence": "low",
            "reason": "No local retrieval results were found.",
            "missing_information": ["game information"],
        }

    top_result = retrieved_results[0]
    similarity = float(top_result.get("similarity", 0.0))
    document = top_result.get("document", "").lower()
    metadata = top_result.get("metadata", {})

    requested_fields = []

    if "developed" in query_lower or "developer" in query_lower:
        requested_fields.append("developer")
    if "released" in query_lower or "release date" in query_lower or "when was" in query_lower:
        requested_fields.append("release_date")
    if "platform" in query_lower or "launched on" in query_lower:
        requested_fields.append("platforms")
    if "publisher" in query_lower or "published" in query_lower:
        requested_fields.append("publisher")
    if "genre" in query_lower:
        requested_fields.append("genre")

    missing_fields = []
    for field in requested_fields:
        value = metadata.get(field, "")
        if not value:
            missing_fields.append(field)

    if similarity >= 0.45 and not missing_fields:
        return {
            "tool": "evaluate_retrieval",
            "is_sufficient": True,
            "confidence": "high",
            "reason": "The top local result appears relevant and contains the requested information.",
            "top_similarity": similarity,
            "missing_information": [],
        }

    if similarity >= 0.30 and not missing_fields:
        return {
            "tool": "evaluate_retrieval",
            "is_sufficient": True,
            "confidence": "medium",
            "reason": "The local result is moderately relevant and contains the requested information.",
            "top_similarity": similarity,
            "missing_information": [],
        }

    return {
        "tool": "evaluate_retrieval",
        "is_sufficient": False,
        "confidence": "low",
        "reason": "The retrieved result was weak or missing the requested field.",
        "top_similarity": similarity,
        "missing_information": missing_fields or ["reliable matching local answer"],
    }


@tool
def game_web_search(query: str, search_depth: str = "advanced") -> Dict[str, Any]:
    
    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        return {
            "tool": "game_web_search",
            "query": query,
            "answer": "",
            "results": [],
            "source": "tavily_web_search",
            "error": "TAVILY_API_KEY is missing. Add it to your .env file.",
        }

    client = TavilyClient(api_key=api_key)

    search_result = client.search(
        query=query,
        search_depth=search_depth,
        include_answer=True,
        include_raw_content=False,
        include_images=False,
    )

    return {
        "tool": "game_web_search",
        "query": query,
        "answer": search_result.get("answer", ""),
        "results": search_result.get("results", []),
        "source": "tavily_web_search",
        "retrieved_at": datetime.now().isoformat(),
    }


def persist_web_result_to_memory(query: str, web_result: Dict[str, Any]) -> Optional[str]:

    store = _require_vector_store()

    answer = web_result.get("answer", "")
    sources = web_result.get("results", [])

    if not answer and not sources:
        return None

    return store.add_web_memory(
        query=query,
        answer=answer,
        sources=sources,
    )


def get_udaplay_tools():
    
    return [
        retrieve_game,
        evaluate_retrieval,
        game_web_search,
    ]