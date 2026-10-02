import os
from dotenv import load_dotenv

# Load .env from the backend directory (supports local dev without manual env var exports)
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
import logging
from typing import Optional, List
from backend.graph import graph

logger = logging.getLogger(__name__)

app = FastAPI(title="Weather-Advisory Support Bot API")

# Configure CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)

class ChatResponse(BaseModel):
    session_id: str
    response: str
    location: Optional[str] = None
    matched_sops: Optional[List[str]] = None
    policy_evidence: Optional[str] = None

@app.get("/health")
def health_check():
    """Basic health-check endpoint."""
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        config = {"configurable": {"thread_id": request.session_id}}
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=request.message)]},
            config=config
        )
        
        last_message = result["messages"][-1].content if result.get("messages") else "No response generated."
        sops = [sop["sop"]["id"] for sop in result.get("matched_sops", [])] if result.get("matched_sops") else []
        
        loc = result.get("location", "Unknown")
        coords = result.get("coordinates")
        lat = coords.get("lat") if coords else 0
        lon = coords.get("lon") if coords else 0
        import datetime
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        policy_str = ", ".join(sops) if sops else "SOP-011 Exercise Conditions: All Clear (v1.0)"
        evidence_str = f"Policy: {policy_str} | Weather: Open-Meteo, {loc} ({lat}, {lon}) | fetched {timestamp} | window today" if result.get("requires_weather") else None
        
        return ChatResponse(
            session_id=request.session_id,
            response=last_message,
            location=result.get("location"),
            matched_sops=sops,
            policy_evidence=evidence_str
        )
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
