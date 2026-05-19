#UdaPlay vector store manager for the RAG pipeline

# UdaPlay vector store manager for the RAG pipeline

from __future__ import annotations
import sys

try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.utils import embedding_functions


class UdaPlayVectorStore:
    def __init__(
        self,
        persist_dir: str = "./chroma_db/udaplay_games",
        collection_name: str = "udaplay_games",
        embedding_model: str = "all-MiniLM-L6-v2",
        reset_collection: bool = False,
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=persist_dir)

        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )

        if reset_collection:
            try:
                self.client.delete_collection(collection_name)
            except Exception:
                pass

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _safe_list(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, list):
            return ", ".join(str(v) for v in value)
        return str(value)

    @staticmethod
    def make_id(prefix: str, text: str) -> str:
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
        return f"{prefix}_{digest}"

    def load_games_from_json(self, file_path: str | Path) -> List[Dict[str, Any]]:
        file_path = Path(file_path)
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Game JSON must contain a list of game records.")
        return data

    def format_game_document(self, game: Dict[str, Any]) -> str:
        return (
            f"Title: {game.get('title', '')}\n"
            f"Developer: {game.get('developer', '')}\n"
            f"Publisher: {game.get('publisher', '')}\n"
            f"Release Date: {game.get('release_date', '')}\n"
            f"Platforms: {self._safe_list(game.get('platforms', []))}\n"
            f"Genre: {game.get('genre', '')}\n"
            f"Description: {game.get('description', '')}"
        )

    def game_metadata(self, game: Dict[str, Any], source_file: str = "") -> Dict[str, Any]:
        
        return {
            "record_type": "game",
            "title": str(game.get("title", "")),
            "developer": str(game.get("developer", "")),
            "publisher": str(game.get("publisher", "")),
            "release_date": str(game.get("release_date", "")),
            "platforms": self._safe_list(game.get("platforms", [])),
            "genre": str(game.get("genre", "")),
            "source": source_file or "local_game_dataset",
        }

    def add_games(self, games: List[Dict[str, Any]], source_file: str = "games.json") -> int:
        documents, metadatas, ids = [], [], []

        for game in games:
            doc = self.format_game_document(game)
            title = str(game.get("title", "unknown_game"))
            doc_id = self.make_id("game", title.lower())
            documents.append(doc)
            metadatas.append(self.game_metadata(game, source_file=source_file))
            ids.append(doc_id)

        if not documents:
            return 0
        self.collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
        return len(documents)

    def add_games_from_file(self, file_path: str | Path) -> int:
        games = self.load_games_from_json(file_path)
        return self.add_games(games, source_file=str(file_path))

    def search_games(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        result = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        rows: List[Dict[str, Any]] = []
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        ids = result.get("ids", [[]])[0]

        for doc_id, doc, meta, dist in zip(ids, docs, metas, dists):
            rows.append({
                "id": doc_id,
                "document": doc,
                "metadata": meta or {},
                "distance": float(dist),
                "similarity": round(1 - float(dist), 4),
            })
        return rows

    def add_web_memory(
        self,
        query: str,
        answer: str,
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        sources = sources or []
        source_text = "\n".join(
            f"- {s.get('title', 'Untitled')}: {s.get('url', '')}" for s in sources[:5]
        )
        doc = (
            f"Web-discovered UdaPlay memory\n"
            f"Query: {query}\n"
            f"Answer: {answer}\n"
            f"Sources:\n{source_text}"
        )
        doc_id = self.make_id("web", query + answer)
        metadata = {
            "record_type": "web_memory",
            "title": query[:120],
            "source": "tavily_web_search",
        }
        self.collection.upsert(documents=[doc], metadatas=[metadata], ids=[doc_id])
        return doc_id
