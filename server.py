import uvicorn

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent_backend import app as langgraph_app


app = FastAPI(
    title="AI Blog Generator API",
    version="1.0.0",
)


# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows local Live Server + any deployed frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Request Schema
# ---------------------------------------------------------

class BlogRequest(BaseModel):
    topic: str


# ---------------------------------------------------------
# Health Check
# ---------------------------------------------------------

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "message": "AI Blog Generator backend is running",
    }


# ---------------------------------------------------------
# Generate Blog
# ---------------------------------------------------------

@app.post("/api/generate")
async def generate_blog(request: BlogRequest):

    if not request.topic.strip():
        raise HTTPException(
            status_code=400,
            detail="Topic cannot be empty.",
        )

    try:
        result = langgraph_app.invoke(
            {
                "topic": request.topic.strip(),
                "sections": [],
            }
        )

        return {
            "status": "success",
            "topic": request.topic,
            "mode": result.get(
                "mode",
                "closed_book",
            ),
            "queries": result.get(
                "queries",
                [],
            ),
            "content": result.get(
                "final",
                "",
            ),
        }

    except Exception as e:

        print("Backend error:", repr(e))

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ---------------------------------------------------------
# Run Server
# ---------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )