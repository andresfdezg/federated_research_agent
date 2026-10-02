"""HTTP API of a federated data node. Every request, including rejected ones, is written to the audit log."""

from typing import Annotated

from fastapi import FastAPI, Header, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from fedresearch.node.engine import NodeEngine
from fedresearch.node.privacy import AuditLog
from fedresearch.node.spec import CountRequest, RegressionRequest, SummaryRequest

Requester = Annotated[str, Header(description="Identity of the calling researcher/agent session")]


def create_app(engine: NodeEngine, audit: AuditLog) -> FastAPI:
    app = FastAPI(title=f"Federated node: {engine.name}", description=engine.description)

    def release(endpoint: str, req, requester: str, response: dict) -> dict:
        audit.record(
            node=engine.name,
            requester=requester,
            endpoint=endpoint,
            request=req.model_dump(),
            status=response["status"],
            released=response.get("result"),
            reason=response.get("reason"),
        )
        return response

    @app.exception_handler(RequestValidationError)
    async def rejected(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = jsonable_encoder(exc.errors())
        audit.record(
            node=engine.name,
            requester=request.headers.get("x-requester", "anonymous"),
            endpoint=request.url.path,
            status="rejected",
            errors=errors,
        )
        return JSONResponse(status_code=422, content={"node": engine.name, "status": "rejected", "detail": errors})

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "node": engine.name}

    @app.get("/v1/metadata")
    def metadata() -> dict:
        return engine.metadata()

    @app.post("/v1/count")
    def count(req: CountRequest, x_requester: Requester = "anonymous") -> dict:
        return release("count", req, x_requester, engine.count(req))

    @app.post("/v1/summary")
    def summary(req: SummaryRequest, x_requester: Requester = "anonymous") -> dict:
        return release("summary", req, x_requester, engine.summary(req))

    @app.post("/v1/regression")
    def regression(req: RegressionRequest, x_requester: Requester = "anonymous") -> dict:
        return release("regression", req, x_requester, engine.regression(req))

    return app
