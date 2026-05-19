from typing import TypedDict, List, Optional, Union
import json

from lib.state_machine import StateMachine, Step, EntryPoint, Termination, Run
from lib.llm import LLM
from lib.messages import (
    AIMessage,
    UserMessage,
    SystemMessage,
    ToolMessage,
    BaseMessage,
)
from lib.tooling import Tool, ToolCall
from lib.memory import ShortTermMemory


class AgentState(TypedDict):
    user_query: str
    instructions: str
    messages: List[BaseMessage]
    current_tool_calls: Optional[List[ToolCall]]
    session_id: str
    total_tokens: int


class Agent:
    def __init__(
        self,
        model_name: str,
        instructions: str,
        tools: Optional[List[Tool]] = None,
        temperature: float = 0.7,
    ):
        """
        Initialize an Agent.

        Args:
            model_name: Name/identifier of the LLM model to use.
            instructions: System instructions for the agent.
            tools: Optional list of tools available to the agent.
            temperature: Temperature parameter for LLM.
        """
        self.instructions = instructions
        self.tools = tools if tools else []
        self.model_name = model_name
        self.temperature = temperature

        self.memory = ShortTermMemory()
        self.workflow = self._create_state_machine()

    def _prepare_messages_step(self, state: AgentState) -> AgentState:
        """
        Prepare messages for LLM consumption.

        If this is a new session, start with the system instructions.
        If this is an existing session, reuse previous messages and append
        the new user message. This enables conversational memory.
        """
        messages = list(state.get("messages", []))

        if not messages:
            messages = [SystemMessage(content=state["instructions"])]

        messages.append(UserMessage(content=state["user_query"]))

        return {
            "user_query": state["user_query"],
            "instructions": state["instructions"],
            "messages": messages,
            "current_tool_calls": state.get("current_tool_calls"),
            "session_id": state["session_id"],
            "total_tokens": state.get("total_tokens", 0),
        }

    def _llm_step(self, state: AgentState) -> AgentState:
        """
        Send the current message history to the LLM.

        The LLM may either:
        1. Return a direct answer.
        2. Return one or more tool calls.
        """
        llm = LLM(
            model=self.model_name,
            temperature=self.temperature,
            tools=self.tools,
        )

        response = llm.invoke(state["messages"])

        tool_calls = response.tool_calls if getattr(response, "tool_calls", None) else None

        current_total = state.get("total_tokens", 0)

        token_usage = getattr(response, "token_usage", None)
        if token_usage:
            current_total += getattr(token_usage, "total_tokens", 0)

        ai_message = AIMessage(
            content=response.content,
            tool_calls=tool_calls,
        )

        return {
            "user_query": state["user_query"],
            "instructions": state["instructions"],
            "messages": state["messages"] + [ai_message],
            "current_tool_calls": tool_calls,
            "session_id": state["session_id"],
            "total_tokens": current_total,
        }

    def _tool_step(self, state: AgentState) -> AgentState:
        """
        Execute pending tool calls.

        For each tool call requested by the LLM:
        - Extract the function name.
        - Parse the function arguments.
        - Find the matching local Python tool.
        - Execute it.
        - Convert the result into a ToolMessage.
        """
        tool_calls = state.get("current_tool_calls") or []
        tool_messages: List[ToolMessage] = []

        for call in tool_calls:
            function_name = call.function.name
            raw_arguments = call.function.arguments or "{}"
            tool_call_id = call.id

            try:
                function_args = json.loads(raw_arguments)
            except json.JSONDecodeError:
                function_args = {}

            matching_tool = next(
                (t for t in self.tools if t.name == function_name),
                None,
            )

            if matching_tool:
                try:
                    result = matching_tool(**function_args)
                    content = json.dumps(result, default=str)
                except Exception as exc:
                    content = json.dumps(
                        {
                            "error": str(exc),
                            "tool": function_name,
                        },
                        default=str,
                    )

                tool_message = ToolMessage(
                    content=content,
                    tool_call_id=tool_call_id,
                    name=function_name,
                )

                tool_messages.append(tool_message)

        return {
            "user_query": state["user_query"],
            "instructions": state["instructions"],
            "messages": state["messages"] + tool_messages,
            "current_tool_calls": None,
            "session_id": state["session_id"],
            "total_tokens": state.get("total_tokens", 0),
        }

    def _create_state_machine(self) -> StateMachine[AgentState]:
        """
        Create the internal state machine for the agent.

        Workflow:
            Entry
              -> message_prep
              -> llm_processor
              -> if tool calls exist: tool_executor
              -> llm_processor
              -> termination when no tool calls remain
        """
        machine = StateMachine[AgentState](AgentState)

        entry = EntryPoint[AgentState]()
        message_prep = Step[AgentState]("message_prep", self._prepare_messages_step)
        llm_processor = Step[AgentState]("llm_processor", self._llm_step)
        tool_executor = Step[AgentState]("tool_executor", self._tool_step)
        termination = Termination[AgentState]()

        machine.add_steps(
            [
                entry,
                message_prep,
                llm_processor,
                tool_executor,
                termination,
            ]
        )

        machine.connect(entry, message_prep)
        machine.connect(message_prep, llm_processor)

        def check_tool_calls(state: AgentState) -> Union[Step[AgentState], str]:
            if state.get("current_tool_calls"):
                return tool_executor
            return termination

        machine.connect(
            llm_processor,
            [tool_executor, termination],
            check_tool_calls,
        )

        machine.connect(tool_executor, llm_processor)

        return machine

    def invoke(self, query: str, session_id: Optional[str] = None) -> Run:
        """
        Run the agent on a user query.

        Args:
            query: The user's query.
            session_id: Optional session identifier. Uses "default" if None.

        Returns:
            Run object containing snapshots and final state.
        """
        session_id = session_id or "default"

        self.memory.create_session(session_id)

        previous_messages: List[BaseMessage] = []

        last_run: Optional[Run] = self.memory.get_last_object(session_id)

        if last_run:
            last_state = last_run.get_final_state()
            if last_state:
                previous_messages = list(last_state["messages"])

        initial_state: AgentState = {
            "user_query": query,
            "instructions": self.instructions,
            "messages": previous_messages,
            "current_tool_calls": None,
            "session_id": session_id,
            "total_tokens": 0,
        }

        run_object = self.workflow.run(initial_state)

        self.memory.add(run_object, session_id)

        return run_object

    def get_session_runs(self, session_id: Optional[str] = None) -> List[Run]:
        """
        Get all Run objects for a session.
        """
        return self.memory.get_all_objects(session_id)

    def reset_session(self, session_id: Optional[str] = None):
        """
        Reset memory for a specific session.
        """
        self.memory.reset(session_id)