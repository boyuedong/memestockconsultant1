"""
FastAPI entry point.

Run with:
    cd /Users/boyuedong/Desktop/new3:11
    uvicorn backend.main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes.chat import router as chat_router, profile_router

app = FastAPI(title="Stock Recommendation Chatbot API", version="1.0.0")

# Allow the React dev server (port 5173) and any other local origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(profile_router)


@app.get("/")
async def root():
    return {"status": "ok", "message": "Stock chatbot API is running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}
