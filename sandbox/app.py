"""FastAPI HTTP layer for the code execution sandbox.

Wires together guardrails.validate_code and executor.execute_code behind two
endpoints:

  GET  /healthz   — liveness/readiness probe
  POST /execute   — validate and run Python code, return captured output
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from sandbox.executor import execute_code
from sandbox.guardrails import validate_code

app = FastAPI(title="Code Sandbox", description="Isolated Python code execution sandbox")


class ExecuteRequest(BaseModel):
    code: str
    timeout: float = Field(default=10.0, gt=0, le=30.0)


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.post("/execute")
async def execute(req: ExecuteRequest) -> JSONResponse:
    if not req.code.strip():
        return JSONResponse(status_code=400, content={"error": "No code provided"})

    violations = validate_code(req.code)
    if violations:
        return JSONResponse(
            status_code=400,
            content={"error": "Code validation failed", "violations": violations},
        )

    result = await execute_code(req.code, req.timeout)
    return JSONResponse(
        status_code=200,
        content={
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.exit_code,
            "timed_out": result.timed_out,
        },
    )
