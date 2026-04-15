"""FastAPI HTTP layer for the code execution sandbox.

Wires together the multi-stage pipeline (guardrails → execute → post-review)
behind three endpoints:

  GET  /healthz   — liveness/readiness probe
  GET  /profile   — active profile introspection
  POST /execute   — validate and run Python code, return captured output
"""

import logging

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from sandbox.landlock import apply_sandbox_landlock
from sandbox.pipeline import run_pipeline
from sandbox.profiles import Profile, get_active_profile

logger = logging.getLogger(__name__)

app = FastAPI(title="Code Sandbox", description="Isolated Python code execution sandbox")

# Apply Landlock filesystem restrictions at import time (before first
# request).  Rules are inherited by subprocess children.  Degrades
# gracefully on non-Linux or older kernels.
_landlock_status = apply_sandbox_landlock()
if _landlock_status.applied:
    logger.info("Landlock active (ABI v%d)", _landlock_status.abi_version)
elif _landlock_status.reason:
    logger.info("Landlock not applied: %s", _landlock_status.reason)

# Load profile once at startup.
_profile: Profile = get_active_profile()
logger.info(
    "Sandbox profile: %s (%d allowed imports, %d blocklist entries)",
    _profile.name,
    len(_profile.allowed_imports),
    len(_profile.blocklist),
)


class ExecuteRequest(BaseModel):
    code: str
    timeout: float = Field(default=10.0, gt=0, le=30.0)


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.get("/profile")
async def profile() -> dict:
    """Return the active profile for introspection."""
    return {
        "name": _profile.name,
        "description": _profile.description,
        "allowed_imports": sorted(_profile.allowed_imports),
        "blocklist_entries": len(_profile.blocklist),
        "scan_stages": {
            "pre": _profile.scan_stages.pre,
            "post": _profile.scan_stages.post,
        },
    }


@app.post("/execute")
async def execute(req: ExecuteRequest) -> JSONResponse:
    if not req.code.strip():
        return JSONResponse(status_code=400, content={"error": "No code provided"})

    pipeline_result = await run_pipeline(req.code, _profile, req.timeout)

    if pipeline_result.rejected:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Code validation failed",
                "violations": pipeline_result.violations,
            },
        )

    result = pipeline_result.result
    return JSONResponse(
        status_code=200,
        content={
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.exit_code,
            "timed_out": result.timed_out,
        },
    )
