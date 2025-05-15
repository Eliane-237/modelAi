import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
import time
import os

from app.core.config import get_settings, Settings
from app.api.api import api_router

from fastapi.middleware.cors import CORSMiddleware




# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()

# Configuration des variables d'environnement
os.environ["MILVUS_HOST"] = settings.MILVUS_HOST
os.environ["MILVUS_PORT"] = str(settings.MILVUS_PORT)

# Initialisation de l'application FastAPI
app = FastAPI(
    title=settings.APP_NAME,
    description="API pour le système de recherche et question-réponse juridique camerounais.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configuration CORS pour permettre les requêtes depuis le frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Autorise toutes les origines en développement
    allow_credentials=True,
    allow_methods=["*"],  # Autorise toutes les méthodes
    allow_headers=["*"],  # Autorise tous les headers
    expose_headers=["Content-Disposition", "X-Process-Time"],  # Headers exposés au frontend
)


routes = [
    f"{route.path} [{', '.join(route.methods)}]"
    for route in app.routes
]
logger.info(f"Routes disponibles: {routes}")

# Middleware pour le logging des requêtes
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Enregistrer le début de la requête
    logger.info(f"Requête entrante: {request.method} {request.url.path}")
    
    # Traiter la requête
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        
        # Enregistrer la fin de la requête
        logger.info(f"Requête traitée: {request.method} {request.url.path} - {response.status_code} en {process_time:.4f}s")
        
        return response
    except Exception as e:
        # Enregistrer l'erreur
        logger.error(f"Erreur lors du traitement de la requête: {request.method} {request.url.path} - {e}")
        
        # Retourner une réponse d'erreur
        return JSONResponse(
            status_code=500,
            content={"detail": f"Une erreur interne est survenue: {str(e)}"}
        )

# Inclure les routes de l'API sous un préfixe
API_PREFIX = "/api"
app.include_router(api_router, prefix=API_PREFIX)

# Route d'accueil simple
@app.get("/")
async def root():
    return {
        "name": settings.APP_NAME,
        "description": "Système de recherche et question-réponse juridique camerounais avec LangChain",
        "documentation": "/docs",
        "api_version": "1.0.0",
        "status": "online"
    }

@app.get("/test/langchain")
async def test_langchain():
    from app.services.langchain_init import init_langchain
    try:
        # Initialiser et tester LangChain
        orchestrator = init_langchain(
            data_path=settings.DATA_PATH,
            metadata_path=settings.METADATA_PATH,
            embedding_service_url=settings.EMBEDDING_SERVICE_URL,
            milvus_host=settings.MILVUS_HOST,
            milvus_port=settings.MILVUS_PORT,
            milvus_collection=settings.MILVUS_COLLECTION,
            llm_service_url=settings.LLM_SERVICE_URL,
            llm_model=settings.LLM_MODEL,
            embedding_dim=settings.EMBEDDING_DIM
        )
        
        # Tester avec une requête simple
        test_query = "Quelles sont les obligations fiscales d'une entreprise au Cameroun?"
        result = orchestrator.generate_response(test_query)
        
        return {
            "status": "success", 
            "message": "LangChain test successful",
            "query": test_query,
            "response": result.get("response", ""),
            "source_count": len(result.get("source_documents", []))
        }
    except Exception as e:
        import traceback
        logger.error(f"Erreur lors du test LangChain: {e}")
        logger.error(traceback.format_exc())
        return {
            "status": "error", 
            "message": str(e),
            "traceback": traceback.format_exc()
        }
    
# Route de vérification de santé
@app.get("/health")
async def health_check():
    # Vérifier l'état des services essentiels
    health_status = {
        "status": "ok",
        "timestamp": time.time(),
        "services": {
            "api": "online"
            # Vous pourriez ajouter d'autres vérifications ici:
            # "milvus": "...",
            # "llm": "...",
            # etc.
        }
    }
    return health_status

# Événement de démarrage de l'application
@app.on_event("startup")
async def startup_event():
    """Événement exécuté au démarrage de l'application."""
    try:
        logger.info("🚀 Démarrage du système RAG avec LangChain...")
        
        # Pré-initialiser LangChain (facultatif, améliore le premier temps de réponse)
        from app.services.langchain_init import init_langchain
        init_langchain(
            
            data_path=settings.DATA_PATH,
            metadata_path=settings.METADATA_PATH,
            embedding_service_url=settings.EMBEDDING_SERVICE_URL,
            milvus_host=settings.MILVUS_HOST,
            milvus_port=settings.MILVUS_PORT,
            milvus_collection=settings.MILVUS_COLLECTION,
            llm_service_url=settings.LLM_SERVICE_URL,
            llm_model=settings.LLM_MODEL,
            embedding_dim=settings.EMBEDDING_DIM
        )
        
        logger.info("✅ Système RAG démarré avec succès")
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'initialisation de LangChain: {e}")
        # Continuer malgré l'erreur - initialisation à la demande possible

# Événement d'arrêt de l'application
@app.on_event("shutdown")
async def shutdown_event():
    """Événement exécuté à l'arrêt de l'application."""
    try:
        logger.info("🛑 Arrêt du système RAG...")
        
        # Nettoyage des ressources LangChain
        from app.services.langchain_init import reset_orchestrator
        reset_orchestrator()
        
        logger.info("✅ Système RAG arrêté proprement")
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'arrêt: {e}")

# Point d'entrée pour uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)