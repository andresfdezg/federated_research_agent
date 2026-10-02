"""Plan-and-execute agent with a human approval gate before any query reaches the data nodes.

    discover -> plan -> approve (HITL interrupt) -> execute -> reflect -> synthesize
                 ^         | revise                             | needs more
                 +---------+------------------------------------+
"""

import json
import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from fedresearch.agent import prompts
from fedresearch.agent.llm import chat_model
from fedresearch.agent.plan import AnalysisPlan, AnalysisStep, describe_step, render_result, validate_plan
from fedresearch.federation.client import FederationClient
from fedresearch.federation.combine import combine_count, combine_regression, combine_summary, site_statuses

MAX_PLANNING_ROUNDS = 3
MAX_REPAIR_ATTEMPTS = 2
COMBINE = {"count": combine_count, "summary": combine_summary, "regression": combine_regression}


class AgentState(TypedDict, total=False):
    question: str
    nodes: dict[str, dict]
    plan: dict | None
    issues: list[str]  # what the next plan must address: reviewer feedback or failed results
    rounds: int
    results: Annotated[list[dict], operator.add]
    decisions: Annotated[list[dict], operator.add]
    status: str
    report: str


class Reflection(BaseModel):
    sufficient: bool
    missing: str = Field("", description="If not sufficient: which analysis is needed next")


def build_graph(client: FederationClient, *, checkpointer=None, auto_approve: bool = False):
    llm = chat_model()
    planner = llm.with_structured_output(AnalysisPlan, method="json_schema")
    reflector = llm.with_structured_output(Reflection, method="json_schema")

    async def discover(state: AgentState) -> dict:
        return {"nodes": await client.metadata(), "rounds": 0, "issues": []}

    async def plan(state: AgentState) -> Command[Literal["approve", "__end__"]]:
        messages = [SystemMessage(prompts.planner_system(state["nodes"])), HumanMessage(state["question"])]
        previous = state.get("plan")
        if previous:
            # Show the planner its own last plan as an assistant turn, then what went wrong with it.
            latest = [r["text"] for r in state.get("results", []) if r["round"] == state["rounds"]]
            messages += [
                AIMessage(json.dumps(previous)),
                HumanMessage(prompts.REPLAN_INSTRUCTION.format(
                    results="\n\n".join(latest) or "(not executed)",
                    issues="\n".join(f"- {i}" for i in state.get("issues", [])),
                )),
            ]

        for _ in range(MAX_REPAIR_ATTEMPTS + 1):
            candidate: AnalysisPlan = await planner.ainvoke(messages)
            errors = validate_plan(candidate)
            if previous and candidate.model_dump(mode="json")["steps"] == previous["steps"]:
                errors.append("this plan is identical to the previous one, which did not work; change it")
            if not errors:
                return Command(
                    goto="approve",
                    update={"plan": candidate.model_dump(mode="json"), "rounds": state["rounds"] + 1, "issues": []},
                )
            messages += [
                AIMessage(candidate.model_dump_json()),
                HumanMessage(prompts.INVALID_PLAN_INSTRUCTION.format(errors="\n".join(f"- {e}" for e in errors))),
            ]
        return Command(
            goto=END,
            update={"status": "failed", "report": "Could not produce a valid analysis plan:\n" + "\n".join(errors)},
        )

    def approve(state: AgentState) -> Command[Literal["execute", "plan", "__end__"]]:
        steps = [AnalysisStep.model_validate(s) for s in state["plan"]["steps"]]
        if auto_approve:
            decision = {"decision": "approve", "by": "auto"}
        else:
            decision = interrupt({
                "type": "plan_approval",
                "question": state["question"],
                "rationale": state["plan"]["rationale"],
                "steps": [{"description": describe_step(s), "purpose": s.purpose} for s in steps],
                "plan": state["plan"],
                "nodes": list(state["nodes"]),
            })
        record = {"round": state["rounds"], **decision}
        match decision.get("decision"):
            case "approve":
                return Command(goto="execute", update={"decisions": [record]})
            case "revise":
                return Command(
                    goto="plan", update={"decisions": [record], "issues": [f"Reviewer: {decision.get('feedback', '')}"]}
                )
        return Command(
            goto=END,
            update={
                "decisions": [record],
                "status": "rejected",
                "report": "The reviewer rejected the analysis plan. No queries were sent to the data nodes.",
            },
        )

    async def execute(state: AgentState) -> dict:
        results = []
        for step in (AnalysisStep.model_validate(s) for s in state["plan"]["steps"]):
            responses = await client.run(step.to_request())
            sites = site_statuses(responses)
            combined = COMBINE[step.analysis](responses)
            results.append({
                "round": state["rounds"],
                "step": step.model_dump(mode="json"),
                "sites": sites,
                "combined": combined,
                "text": render_result(step, sites, combined),
            })
        return {"results": results}

    async def reflect(state: AgentState) -> Command[Literal["plan", "synthesize"]]:
        latest = [r for r in state["results"] if r["round"] == state["rounds"]]
        failed = [r for r in latest if all(not s.startswith("ok") for s in r["sites"].values())]
        if state["rounds"] >= MAX_PLANNING_ROUNDS:
            return Command(goto="synthesize")
        if failed:
            issues = [f"No node returned a result for: {describe_step(AnalysisStep.model_validate(r['step']))} "
                      f"— {'; '.join(r['sites'].values())}" for r in failed]
            return Command(goto="plan", update={"issues": issues})
        verdict: Reflection = await reflector.ainvoke([
            SystemMessage(prompts.REFLECT_SYSTEM),
            HumanMessage(f"Question: {state['question']}\n\nResults:\n\n" + "\n\n".join(r["text"] for r in state["results"])),
        ])
        if verdict.sufficient:
            return Command(goto="synthesize")
        return Command(goto="plan", update={"issues": [f"Results review: {verdict.missing}"]})

    async def synthesize(state: AgentState) -> dict:
        msg = await llm.ainvoke([
            SystemMessage(prompts.SYNTHESIS_SYSTEM),
            HumanMessage(f"Question: {state['question']}\n\nResults:\n\n" + "\n\n".join(r["text"] for r in state["results"])),
        ])
        return {"status": "completed", "report": msg.content}

    graph = StateGraph(AgentState)
    for name, fn in [("discover", discover), ("plan", plan), ("approve", approve), ("execute", execute),
                     ("reflect", reflect), ("synthesize", synthesize)]:
        graph.add_node(name, fn)
    graph.add_edge(START, "discover")
    graph.add_edge("discover", "plan")
    graph.add_edge("execute", "reflect")
    graph.add_edge("synthesize", END)
    return graph.compile(checkpointer=checkpointer)
