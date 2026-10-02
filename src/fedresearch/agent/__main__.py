"""Ask the federated research agent a question from the terminal (nodes must be running).

    python -m fedresearch.agent "Is the TCF7L2 risk allele associated with type 2 diabetes?"
    python -m fedresearch.agent --auto-approve "..."
"""

import argparse
import asyncio
import sys
import uuid

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from fedresearch.agent.graph import build_graph
from fedresearch.federation.client import FederationClient, endpoints_from_env_or_config

PROGRESS = {
    "discover": lambda u: f"discovered {len(u['nodes'])} nodes",
    "plan": lambda u: "analysis plan drafted" if u.get("plan") else "planning failed",
    "execute": lambda u: f"ran {len(u['results'])} federated analyses",
}


def ask_reviewer(request: dict) -> dict:
    print("\n" + "=" * 72)
    print("APPROVAL REQUIRED — these analyses will run on:", ", ".join(request["nodes"]))
    print(f"Rationale: {request['rationale']}\n")
    for i, step in enumerate(request["steps"], 1):
        print(f"  {i}. {step['description']}\n     purpose: {step['purpose']}")
    print("=" * 72)
    while True:
        answer = input("[a]pprove, [r]evise with feedback, or [x] reject? ").strip().lower()
        if answer in ("a", "approve"):
            return {"decision": "approve", "by": "cli"}
        if answer in ("r", "revise"):
            return {"decision": "revise", "feedback": input("Feedback for the planner: "), "by": "cli"}
        if answer in ("x", "reject"):
            return {"decision": "reject", "by": "cli"}


async def run(question: str, auto_approve: bool) -> None:
    client = FederationClient(endpoints_from_env_or_config(), requester="cli-agent")
    graph = build_graph(client, checkpointer=InMemorySaver(), auto_approve=auto_approve)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    payload: dict | Command = {"question": question}
    try:
        while payload is not None:
            pending = None
            async for chunk in graph.astream(payload, config, stream_mode="updates"):
                for node, update in chunk.items():
                    if node == "__interrupt__":
                        pending = update[0].value
                    elif node in PROGRESS and update:
                        print(f"· {PROGRESS[node](update)}")
                    elif node == "reflect":
                        print("· reviewed results")
            payload = Command(resume=ask_reviewer(pending)) if pending else None
        state = (await graph.aget_state(config)).values
        print("\n" + "=" * 72)
        for r in state.get("results", []):
            print(r["text"] + "\n")
        print("=" * 72 + "\n" + state.get("report", ""))
    finally:
        await client.aclose()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Ask the federated research agent a question.")
    parser.add_argument("question")
    parser.add_argument("--auto-approve", action="store_true", help="skip the human approval gate (evaluation only)")
    args = parser.parse_args()
    asyncio.run(run(args.question, args.auto_approve))


if __name__ == "__main__":
    main()
