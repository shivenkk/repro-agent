"""
FastAPI server for ReproAgent.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.orchestrator import run_pipeline
from backend.models.schemas import ProgressUpdate

app = FastAPI(title="ReproAgent", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store results in memory
runs: dict = {}


class AnalyzeRequest(BaseModel):
    paper_url: str


class AnalyzeResponse(BaseModel):
    run_id: str
    status: str
    message: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    """Start a reproducibility analysis."""
    state = await run_pipeline(req.paper_url)
    runs[state.run_id] = state

    return AnalyzeResponse(
        run_id=state.run_id,
        status=state.status.value,
        message=state.progress_messages[-1] if state.progress_messages else "",
    )


@app.get("/results/{run_id}")
async def get_results(run_id: str):
    """Get results for a completed run."""
    if run_id not in runs:
        return {"error": "Run not found"}

    state = runs[run_id]
    report = state.report

    if not report:
        return {"error": "No report generated", "status": state.status.value}

    return {
        "run_id": state.run_id,
        "status": state.status.value,
        "paper": {
            "title": report.paper.title,
            "authors": report.paper.authors,
            "url": report.paper.url,
            "arxiv_id": report.paper.arxiv_id,
        },
        "score": report.overall_score,
        "verdict": report.verdict.value,
        "extraction_confidence": report.extraction_confidence,
        "hyperparameters": {
            name: {
                "value": field.value,
                "confidence": field.confidence,
                "source": field.source,
            }
            for name, field in report.methodology.hyperparameters.items()
        },
        "claimed_results": report.methodology.claimed_results,
        "comparisons": [
            {
                "metric": c.metric_name,
                "claimed": c.claimed,
                "achieved": c.achieved,
                "within_threshold": c.within_threshold,
            }
            for c in report.comparisons
        ],
        "missing_details": report.methodology.missing_details,
        "failure_reasons": [fr.value for fr in report.failure_reasons],
        "analysis": report.analysis,
        "recommendations": report.recommendations,
        "generated_code": report.final_code,
        "progress": state.progress_messages,
    }


@app.websocket("/ws/{run_id}")
async def websocket_endpoint(websocket: WebSocket, run_id: str):
    """WebSocket for live progress updates."""
    await websocket.accept()

    try:
        data = await websocket.receive_json()
        paper_url = data.get("paper_url", "")

        if not paper_url:
            await websocket.send_json({"error": "No paper_url provided"})
            return

        async def progress_callback(update: ProgressUpdate):
            try:
                await websocket.send_json({
                    "type": "progress",
                    "agent": update.agent,
                    "status": update.status.value,
                    "message": update.message,
                })
            except Exception:
                pass

        state = await run_pipeline(
            paper_url,
            progress_callback=progress_callback,
        )
        runs[state.run_id] = state

        await websocket.send_json({
            "type": "complete",
            "run_id": state.run_id,
            "status": state.status.value,
        })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
