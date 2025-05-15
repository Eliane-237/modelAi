"""
Dépendances FastAPI pour l'API du système RAG juridique camerounais.
Ce module fournit des fonctions pour initialiser et obtenir les services en tant que dépendances.
"""

from fastapi import Depends

from app.core.config import get_settings, Settings
from app.services.embedding_service import EmbeddingService
from app.services.milvus_service import MilvusService
from app.services.llm_service import LlmService
from app.services.rerank_service import RerankService
from app.services.search_service import SearchService
from app.services.pdf_service import PDFService
from app.services.rag_system import RAGSystem

# Ajouter l'import du service LangChain
from app.services.langchain_service import LangChainService, get_langchain_service
from app.services.langchain_init import init_langchain, get_orchestrator

# Cache des services
_embedding_service = None
_milvus_service = None
_llm_service = None
_rerank_service = None
_search_service = None
_pdf_service = None
_rag_system = None
_langchain_service = None

def get_embedding_service(settings: Settings = Depends(get_settings)) -> EmbeddingService:
    """Récupère le service d'embedding."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService(server_url=settings.EMBEDDING_SERVICE_URL)
    return _embedding_service

def get_milvus_service(settings: Settings = Depends(get_settings)) -> MilvusService:
    """Récupère le service Milvus."""
    global _milvus_service
    if _milvus_service is None:
        _milvus_service = MilvusService(
            collection_name=settings.MILVUS_COLLECTION, 
            dim=settings.EMBEDDING_DIM,
            host=settings.MILVUS_HOST,
            port=settings.MILVUS_PORT
        )
    return _milvus_service

def get_llm_service(settings: Settings = Depends(get_settings)) -> LlmService:
    """Récupère le service LLM."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LlmService(
            model_name=settings.LLM_MODEL,
            server_url=settings.LLM_SERVICE_URL
        )
    return _llm_service

def get_rerank_service(
    llm_service: LlmService = Depends(get_llm_service)
) -> RerankService:
    """Récupère le service de reranking."""
    global _rerank_service
    if _rerank_service is None:
        _rerank_service = RerankService(llm_service=llm_service)
    return _rerank_service

def get_search_service(
    milvus_service: MilvusService = Depends(get_milvus_service),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    settings: Settings = Depends(get_settings)
) -> SearchService:
    """Récupère le service de recherche."""
    global _search_service
    if _search_service is None:
        _search_service = SearchService(
            milvus_service=milvus_service,
            embedding_service=embedding_service,
            top_k=settings.DEFAULT_TOP_K
        )
    return _search_service

def get_pdf_service(
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    milvus_service: MilvusService = Depends(get_milvus_service),
    settings: Settings = Depends(get_settings)
) -> PDFService:
    """Récupère le service PDF."""
    global _pdf_service
    if _pdf_service is None:
        _pdf_service = PDFService(
            embedding_service=embedding_service,
            milvus_service=milvus_service,
            data_path=settings.DATA_PATH,
            metadata_path=settings.METADATA_PATH
        )
    return _pdf_service

def get_rag_system(
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    milvus_service: MilvusService = Depends(get_milvus_service),
    llm_service: LlmService = Depends(get_llm_service),
    rerank_service: RerankService = Depends(get_rerank_service),
    settings: Settings = Depends(get_settings)
) -> RAGSystem:
    """Récupère le système RAG."""
    global _rag_system
    if _rag_system is None:
        _rag_system = RAGSystem(
            collection_name=settings.MILVUS_COLLECTION,
            embedding_dim=settings.EMBEDDING_DIM,
            top_k=settings.DEFAULT_TOP_K,
            llm_model=settings.LLM_MODEL,
            max_context_length=4000,
            save_dir=f"{settings.METADATA_PATH}/rag_results"
        )
    return _rag_system

def get_langchain_service_from_deps(
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    milvus_service: MilvusService = Depends(get_milvus_service),
    llm_service: LlmService = Depends(get_llm_service),
    rerank_service: RerankService = Depends(get_rerank_service),
    search_service: SearchService = Depends(get_search_service),
    settings: Settings = Depends(get_settings)
) -> LangChainService:
    """Récupère le service LangChain."""
    global _langchain_service
    if _langchain_service is None:
        # Essayer d'abord de récupérer l'orchestrateur global
        _langchain_service = get_orchestrator()
        
        # S'il n'existe pas, en créer un nouveau
        if _langchain_service is None:
            _langchain_service = get_langchain_service(
                embedding_service=embedding_service,
                milvus_service=milvus_service,
                llm_service=llm_service,
                rerank_service=rerank_service,
                search_service=search_service,
                data_path=settings.DATA_PATH,
                metadata_path=settings.METADATA_PATH
            )
    
    return _langchain_service

def get_initialized_orchestrator():
    """
    Récupère une instance initialisée de l'orchestrateur LangChain.
    Si aucune instance n'existe, déclenche une erreur.
    """
    orchestrator = get_orchestrator()
    if not orchestrator:
        raise ValueError("Orchestrateur LangChain non initialisé.")
    return orchestrator