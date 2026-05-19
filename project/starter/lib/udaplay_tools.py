# lib/udaplay_tools.py

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
    
    global _vector_store

    if _vector_store is None:
        _vector_store = UdaPlayVectorStore(
            persist_dir="./chroma_db/udaplay_games",
            collection_name="udaplay_games",
            reset_collection=False,
        )

    return _vector_store


@tool
@tool
def retrieve_game(query: str, top_k: int = 3) -> Dict[str, Any]:
    
    store = _require_vector_store()
    raw_results = store.search_games(query=query, top_k=top_k)

    compact_results = []

    for r in raw_results:
        metadata = r.get("metadata", {}) or {}

        compact_results.append(
            {
                "title": metadata.get("title", ""),
                "developer": metadata.get("developer", ""),
                "publisher": metadata.get("publisher", ""),
                "release_date": metadata.get("release_date", ""),
                "platforms": metadata.get("platforms", ""),
                "genre": metadata.get("genre", ""),
                "source": metadata.get("source", "local_game_dataset"),
                "similarity": r.get("similarity", 0.0),
                "citation": f"Local dataset: {metadata.get('title', '')}",
            }
        )

    return {
        "tool": "retrieve_game",
        "query": query,
        "top_k": top_k,
        "results": compact_results,
        "source": "local_vector_db",
    }
@tool
def evaluate_retrieval(
    query: str,
    retrieved_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    
    query_lower = query.lower()

    time_sensitive_terms = [
        "right now",
        "currently",
        "latest",
        "recent",
        "working on",
        "ongoing",
        "upcoming",
        "announced",
        "today",
        "this year",
        "now",
    ]

    if any(term in query_lower for term in time_sensitive_terms):
        return {
            "tool": "evaluate_retrieval",
            "is_sufficient": False,
            "confidence": "low",
            "reason": (
                "The query asks for current or time-sensitive information. "
                "The local vector database may be outdated, so web fallback is required."
            ),
            "missing_information": ["current web information"],
        }

    if not retrieved_results:
        return {
            "tool": "evaluate_retrieval",
            "is_sufficient": False,
            "confidence": "low",
            "reason": "No local retrieval results were found.",
            "missing_information": ["local game information"],
        }

    top_result = retrieved_results[0]
    similarity = float(top_result.get("similarity", 0.0))
    
    missing_fields = []

for field in requested_fields:
    value = top_result.get(field)
    if value is None or str(value).strip() == "":
        missing_fields.append(field)

    if similarity >= 0.45 and not missing_fields:
        return {
            "tool": "evaluate_retrieval",
            "is_sufficient": True,
            "confidence": "high",
            "reason": (
                "The top local result is relevant and contains the requested information."
            ),
            "top_similarity": similarity,
            "missing_information": [],
        }

    if similarity >= 0.30 and not missing_fields:
        return {
            "tool": "evaluate_retrieval",
            "is_sufficient": True,
            "confidence": "medium",
            "reason": (
                "The local result is moderately relevant and contains the requested information."
            ),
            "top_similarity": similarity,
            "missing_information": [],
        }

    return {
        "tool": "evaluate_retrieval",
        "is_sufficient": False,
        "confidence": "low",
        "reason": (
            "The retrieved result was weak or did not contain the requested information."
        ),
        "top_similarity": similarity,
        "missing_information": missing_fields or ["reliable matching answer"],
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


def get_udaplay_tools() -> List[Any]:
    
    return [
        retrieve_game,
        evaluate_retrieval,
        game_web_search,
    ]