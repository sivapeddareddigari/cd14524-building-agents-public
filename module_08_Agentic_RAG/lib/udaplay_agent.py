#State-machine

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict, Union

from lib.state_machine import EntryPoint, Run, StateMachine, Step, Termination
from lib.udaplay_reporting import format_text_report
from lib.udaplay_tools import evaluate_retrieval, game_web_search, retrieve_game
from lib.udaplay_vector_store import UdaPlayVectorStore


class UdaPlayState(TypedDict, total=False):
    user_query: str
    session_id: str
    retrieved_results: List[Dict[str, Any]]
    retrieval_evaluation: Dict[str, Any]
    web_results: Dict[str, Any]
    final_report: Dict[str, Any]
    final_text: str
    tools_used: List[str]
    sources: List[Dict[str, Any]]


class UdaPlayAgent:
    def __init__(self, vector_store: UdaPlayVectorStore):
        self.vector_store = vector_store
        self.session_history: Dict[str, List[Run]] = {}
        self.workflow = self._create_workflow()

    def _retrieve_step(self, state: UdaPlayState) -> UdaPlayState:
        result = retrieve_game(query=state["user_query"], top_k=5)
        return {
            "retrieved_results": result.get("results", []),
            "tools_used": state.get("tools_used", []) + ["retrieve_game"],
        }

    def _evaluate_step(self, state: UdaPlayState) -> UdaPlayState:
        evaluation = evaluate_retrieval(
            query=state["user_query"],
            retrieved_results=state.get("retrieved_results", []),
        )
        return {
            "retrieval_evaluation": evaluation,
            "tools_used": state.get("tools_used", []) + ["evaluate_retrieval"],
        }

    def _web_search_step(self, state: UdaPlayState) -> UdaPlayState:
        web = game_web_search(query=state["user_query"])
        sources = []
        for item in web.get("results", [])[:5]:
            sources.append({
                "type": "web",
                "title": item.get("title", "Untitled"),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
            })
        return {
            "web_results": web,
            "sources": sources,
            "tools_used": state.get("tools_used", []) + ["game_web_search"],
        }

    def _persist_web_memory_step(self, state: UdaPlayState) -> UdaPlayState:
        web = state.get("web_results", {})
        if web.get("answer") and not web.get("error"):
            self.vector_store.add_web_memory(
                query=state["user_query"],
                answer=web.get("answer", ""),
                sources=web.get("results", []),
            )
            return {"tools_used": state.get("tools_used", []) + ["persist_web_memory"]}
        return {}

    @staticmethod
    def _infer_answer_from_local(query: str, top_result: Dict[str, Any]) -> str:
        meta = top_result.get("metadata", {}) or {}
        title = meta.get("title", "the game")
        q = query.lower()

        if any(term in q for term in ["developer", "developed", "made", "created"]):
            return f"{title} was developed by {meta.get('developer', 'an unknown developer')}."
        if any(term in q for term in ["publisher", "published"]):
            return f"{title} was published by {meta.get('publisher', 'an unknown publisher')}."
        if any(term in q for term in ["when", "release", "released", "launch", "launched", "date"]):
            return f"{title} was released on {meta.get('release_date', 'an unknown release date')}."
        if any(term in q for term in ["platform", "console", "available on", "launched on"]):
            return f"{title} was launched/available on: {meta.get('platforms', 'unknown platforms')}."
        if "genre" in q:
            return f"{title} is categorized as {meta.get('genre', 'an unknown genre')}."

        return top_result.get("document", "I found a relevant local record, but could not extract a specific answer.")

    def _final_answer_step(self, state: UdaPlayState) -> UdaPlayState:
        evaluation = state.get("retrieval_evaluation", {})
        retrieved = state.get("retrieved_results", [])
        web = state.get("web_results", {})
        tools_used = state.get("tools_used", [])

        fallback_used = bool(web)

        if fallback_used:
            answer = web.get("answer") or "I searched the web, but no clear answer was returned."
            confidence = "medium" if answer and not web.get("error") else "low"
            source_used = "web_search"
            sources = state.get("sources", [])
        else:
            top = retrieved[0] if retrieved else {}
            answer = self._infer_answer_from_local(state["user_query"], top) if top else "I could not find an answer."
            confidence = evaluation.get("confidence", "unknown")
            source_used = "local_vector_db"
            meta = top.get("metadata", {}) if top else {}
            sources = [{
                "type": "local",
                "title": meta.get("title", "Local game dataset"),
                "source": meta.get("source", "local_game_dataset"),
                "similarity": top.get("similarity"),
            }] if top else []

        report = {
            "question": state["user_query"],
            "answer": answer,
            "confidence": confidence,
            "source_used": source_used,
            "fallback_used": fallback_used,
            "tools_used": tools_used,
            "retrieval_evaluation": evaluation,
            "sources": sources,
        }
        return {
            "final_report": report,
            "final_text": format_text_report(report),
            "sources": sources,
        }

    def _create_workflow(self) -> StateMachine[UdaPlayState]:
        machine = StateMachine[UdaPlayState](UdaPlayState)

        entry = EntryPoint[UdaPlayState]()
        retrieve = Step[UdaPlayState]("retrieve_game", self._retrieve_step)
        evaluate = Step[UdaPlayState]("evaluate_retrieval", self._evaluate_step)
        web = Step[UdaPlayState]("game_web_search", self._web_search_step)
        persist = Step[UdaPlayState]("persist_web_memory", self._persist_web_memory_step)
        final = Step[UdaPlayState]("final_answer", self._final_answer_step)
        termination = Termination[UdaPlayState]()

        machine.add_steps([entry, retrieve, evaluate, web, persist, final, termination])
        machine.connect(entry, retrieve)
        machine.connect(retrieve, evaluate)

        def route_after_evaluation(state: UdaPlayState) -> Union[Step[UdaPlayState], str]:
            if state.get("retrieval_evaluation", {}).get("is_sufficient"):
                return final
            return web

        machine.connect(evaluate, [final, web], route_after_evaluation)
        machine.connect(web, persist)
        machine.connect(persist, final)
        machine.connect(final, termination)
        return machine

    def invoke(self, query: str, session_id: str = "default") -> Run:
        initial_state: UdaPlayState = {
            "user_query": query,
            "session_id": session_id,
            "tools_used": [],
            "sources": [],
        }
        run = self.workflow.run(initial_state)
        self.session_history.setdefault(session_id, []).append(run)
        return run

    def get_session_runs(self, session_id: str = "default") -> List[Run]:
        return self.session_history.get(session_id, [])
