from fastapi import APIRouter, FastAPI
from app.api.endpoints import agent, documents, embeddings, index, llm, rerank, search, conversation, agent

api_router = APIRouter()
app = FastAPI()

# Inclure tous les routers des différents endpoints
api_router.include_router(documents.router, prefix="/documents", tags=["documents"])
api_router.include_router(embeddings.router, prefix="/embeddings", tags=["embeddings"])
api_router.include_router(index.router, prefix="/index", tags=["index"])
api_router.include_router(llm.router, prefix="/llm", tags=["llm"])
api_router.include_router(rerank.router, prefix="/rerank", tags=["rerank"])
api_router.include_router(search.router, prefix="/search", tags=["search"])

# Route pour RAG
api_router.include_router(search.rag_router, prefix="/rag", tags=["rag"])

#api_router.include_router(agent.router, prefix="/langchain", tags=["langchain"])

api_router.include_router(agent.agent_router, prefix="/agent", tags=["agent"])

api_router.include_router(conversation.conversation_router, prefix="/rag/question", tags=["conversation"])