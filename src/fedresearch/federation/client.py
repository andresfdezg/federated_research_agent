"""Async client that fans a request out to every node and collects their (already disclosure-controlled) responses."""

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path

import httpx

from fedresearch.node.config import DEFAULT_CONFIG, load_node_configs
from fedresearch.node.spec import CountRequest, RegressionRequest, SummaryRequest

ENDPOINTS = {CountRequest: "/v1/count", SummaryRequest: "/v1/summary", RegressionRequest: "/v1/regression"}


@dataclass(frozen=True)
class NodeEndpoint:
    name: str
    url: str


def endpoints_from_env_or_config(path: Path = DEFAULT_CONFIG) -> list[NodeEndpoint]:
    """`FRA_NODE_URLS="a=http://host:8101,b=..."` overrides the local defaults from config/nodes.yaml."""
    if urls := os.getenv("FRA_NODE_URLS"):
        return [NodeEndpoint(*item.split("=", 1)) for item in urls.split(",")]
    return [NodeEndpoint(c.name, f"http://127.0.0.1:{c.port}") for c in load_node_configs(path).values()]


class FederationClient:
    def __init__(self, endpoints: list[NodeEndpoint], requester: str = "agent", timeout: float = 60.0):
        self.endpoints = endpoints
        self._http = httpx.AsyncClient(timeout=timeout, headers={"X-Requester": requester})

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _call(self, node: NodeEndpoint, method: str, path: str, body: dict | None = None) -> dict:
        try:
            resp = await self._http.request(method, node.url + path, json=body)
        except httpx.HTTPError as exc:
            return {"node": node.name, "status": "unreachable", "reason": f"{type(exc).__name__}: {exc}"}
        if resp.status_code == 422:
            detail = resp.json().get("detail", [])
            reason = "; ".join(f"{'.'.join(map(str, d.get('loc', [])))}: {d.get('msg')}" for d in detail)
            return {"node": node.name, "status": "rejected", "reason": reason}
        if resp.status_code != 200:
            return {"node": node.name, "status": "error", "reason": f"HTTP {resp.status_code}"}
        return resp.json()

    async def metadata(self) -> dict[str, dict]:
        results = await asyncio.gather(*(self._call(n, "GET", "/v1/metadata") for n in self.endpoints))
        return {n.name: r for n, r in zip(self.endpoints, results)}

    async def run(self, request: CountRequest | SummaryRequest | RegressionRequest) -> dict[str, dict]:
        path, body = ENDPOINTS[type(request)], request.model_dump(mode="json")
        results = await asyncio.gather(*(self._call(n, "POST", path, body) for n in self.endpoints))
        return {n.name: r for n, r in zip(self.endpoints, results)}
