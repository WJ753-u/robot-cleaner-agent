from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from storage.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(
    title="扫地机器人智能客服 Agent API",
    description="面向扫地机器人垂直场景的 RAG + Agent 服务接口",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1", tags=["agent"])


@app.get("/")
def root():
    return {
        "service": "robot-agent-api",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
