"""
Module d'initialisation de LangChain pour le système RAG juridique camerounais.
Configure les services nécessaires (embeddings, Milvus, LLM) et retourne un orchestrateur.
Ajoute des vérifications robustes pour LlmService et MilvusService.
"""

import os
import logging
import time
from typing import Optional, Dict, Any, Callable
import requests
from contextlib import contextmanager

# Imports pour les services existants
from app.services.embedding_service import EmbeddingService
from app.services.milvus_service import MilvusService
from app.services.llm_service import LlmService
from app.services.rerank_service import RerankService
from app.services.search_service import SearchService

# Import pour le service LangChain
from app.services.langchain_service import LangChainService, get_langchain_service

# Configuration du logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Contexte pour gérer l'orchestrateur comme un singleton
_orchestrator_instance = None

@contextmanager
def orchestrator_context():
    """Gère l'orchestrateur comme un singleton avec un contexte."""
    global _orchestrator_instance
    try:
        yield _orchestrator_instance
    finally:
        pass  # Pas de réinitialisation automatique pour FastAPI

def init_langchain(
    data_path: str,
    metadata_path: str,
    embedding_service_url: str = "http://10.100.212.118:8000",
    milvus_host: str = "10.100.212.118",
    milvus_port: int = 19530,
    milvus_collection: str = "documents_collection",
    llm_service_url: str = "http://10.100.212.118:8001/generate",
    llm_model: str = "llama3.2:latest",
    embedding_dim: int = 1024,
    streaming_callback: Optional[Callable] = None,
    embedding_service: Optional[EmbeddingService] = None,
    milvus_service: Optional[MilvusService] = None,
    llm_service: Optional[LlmService] = None,
    rerank_service: Optional[RerankService] = None,
    search_service: Optional[SearchService] = None
) -> LangChainService:
    """
    Initialise LangChain et retourne l'orchestrateur configuré.

    Args:
        data_path: Chemin vers le répertoire de données.
        metadata_path: Chemin vers le répertoire de métadonnées.
        embedding_service_url: URL du service d'embedding.
        milvus_host: Adresse du serveur Milvus.
        milvus_port: Port du serveur Milvus.
        milvus_collection: Nom de la collection Milvus.
        llm_service_url: URL du service LLM.
        llm_model: Nom du modèle LLM.
        embedding_dim: Dimension des embeddings.
        streaming_callback: Fonction de callback pour le streaming.
        embedding_service: Service d'embedding pré-instancié (optionnel).
        milvus_service: Service Milvus pré-instancié (optionnel).
        llm_service: Service LLM pré-instancié (optionnel).
        rerank_service: Service de reranking pré-instancié (optionnel).
        search_service: Service de recherche pré-instancié (optionnel).

    Returns:
        LangChainService: Orchestrateur configuré.

    Raises:
        ValueError: Si l'initialisation d'un service échoue.
    """
    global _orchestrator_instance

    # Si l'orchestrateur existe déjà, le retourner
    if _orchestrator_instance is not None:
        logger.info("Orchestrateur déjà initialisé, retour de l'instance existante")
        return _orchestrator_instance

    try:
        logger.info("🚀 Initialisation de LangChain...")

        # Initialiser ou utiliser les services fournis
        logger.info("Initialisation de EmbeddingService...")
        _embedding_service = embedding_service or EmbeddingService(server_url=embedding_service_url)
        # Vérifier la connectivité au service d'embedding
        try:
            test_embedding = _embedding_service.generate_embeddings(["Test"])[0]
            logger.info("Connexion à EmbeddingService réussie")
        except Exception as e:
            logger.error(f"Échec de la connexion à EmbeddingService: {e}")
            raise ValueError(f"Échec de la connexion à EmbeddingService: {e}")

        logger.info("Initialisation de MilvusService...")
        _milvus_service = milvus_service or MilvusService(
            collection_name=milvus_collection,
            dim=embedding_dim,
            host=milvus_host,
            port=milvus_port
        )
        # Vérifier la connectivité à Milvus en testant l'état de la collection
        try:
            _milvus_service.search([0.0] * embedding_dim, top_k=1)
            logger.info("Connexion à MilvusService réussie: collection accessible")
        except Exception as e:
            logger.error(f"Échec de la connexion à MilvusService: {e}")
            raise ValueError(f"Échec de la connexion à MilvusService: {e}")

        logger.info("Initialisation de LlmService...")
        _llm_service = llm_service or LlmService(
            model_name=llm_model,
            server_url=llm_service_url
        )
        # Vérifier la connectivité à l'API LLM
        try:
            test_response = _llm_service.generate_response("Test de connexion", max_length=10)
            logger.info(f"Connexion à LlmService réussie: {test_response}")
            # Vérification explicite que _llm_service est une instance valide
            if not isinstance(_llm_service, LlmService):
                raise ValueError(f"LlmService n'est pas une instance valide: {type(_llm_service)}")
        except Exception as e:
            logger.error(f"Échec de la connexion à LlmService: {e}")
            raise ValueError(f"Échec de la connexion à LlmService: {e}")

        logger.info("Initialisation de SearchService...")
        _search_service = search_service or SearchService(
            milvus_service=_milvus_service,
            embedding_service=_embedding_service
        )

        logger.info("Initialisation de RerankService...")
        _rerank_service = rerank_service
        if _rerank_service is None and _llm_service is not None:
            try:
                _rerank_service = RerankService(llm_service=_llm_service)
                logger.info("✅ Service de reranking initialisé")
            except Exception as e:
                logger.warning(f"⚠️ Impossible d'initialiser le service de reranking: {e}")
                _rerank_service = None

        # Créer le répertoire pour l'historique des conversations
        chat_history_dir = os.path.join(metadata_path, "chat_history")
        os.makedirs(chat_history_dir, exist_ok=True)

        # Initialiser le service LangChain
        logger.info("Initialisation de LangChainService...")
        orchestrator = get_langchain_service(
            embedding_service=_embedding_service,
            milvus_service=_milvus_service,
            llm_service=_llm_service,
            rerank_service=_rerank_service,
            search_service=_search_service,
            data_path=data_path,
            metadata_path=metadata_path,
            save_dir=chat_history_dir
        )

        # Stocker l'orchestrateur
        _orchestrator_instance = orchestrator

        logger.info("✅ LangChain initialisé avec succès")
        return orchestrator

    except Exception as e:
        logger.error(f"❌ Erreur lors de l'initialisation de LangChain: {e}")
        raise

def get_orchestrator() -> Optional[LangChainService]:
    """
    Récupère l'orchestrateur LangChain.

    Returns:
        LangChainService: Orchestrateur ou None s'il n'est pas initialisé.
    """
    global _orchestrator_instance
    if _orchestrator_instance is None:
        logger.warning("Orchestrateur non initialisé. Appelez init_langchain() d'abord.")
    return _orchestrator_instance

def reset_orchestrator() -> None:
    """
    Réinitialise l'orchestrateur LangChain.
    """
    global _orchestrator_instance
    if _orchestrator_instance:
        _orchestrator_instance.reset_conversation()
        logger.info("Orchestrateur réinitialisé")
    _orchestrator_instance = None

def test_services(
    embedding_service: EmbeddingService,
    milvus_service: MilvusService,
    llm_service: LlmService,
    search_service: SearchService,
    rerank_service: Optional[RerankService] = None
) -> Dict[str, bool]:
    """
    Teste l'initialisation et la connectivité des services.

    Args:
        embedding_service: Service d'embedding.
        milvus_service: Service Milvus.
        llm_service: Service LLM.
        search_service: Service de recherche.
        rerank_service: Service de reranking (optionnel).

    Returns:
        Dict[str, bool]: Résultats des tests pour chaque service.
    """
    results = {
        "embedding_service": False,
        "milvus_service": False,
        "llm_service": False,
        "search_service": False,
        "rerank_service": False if rerank_service else True
    }

    # Tester EmbeddingService
    try:
        test_embedding = embedding_service.generate_embeddings(["Test"])[0]
        results["embedding_service"] = True
        logger.info("Test EmbeddingService réussi")
    except Exception as e:
        logger.error(f"Test EmbeddingService échoué: {e}")

    # Tester MilvusService
    try:
        milvus_service.search([0.0] * 1024, top_k=1)
        results["milvus_service"] = True
        logger.info("Test MilvusService réussi")
    except Exception as e:
        logger.error(f"Test MilvusService échoué: {e}")

    # Tester LlmService
    try:
        test_response = llm_service.generate_response("Test de connexion", max_length=10)
        results["llm_service"] = True
        logger.info(f"Test LlmService réussi: {test_response}")
    except Exception as e:
        logger.error(f"Test LlmService échoué: {e}")

    # Tester SearchService
    try:
        test_results = search_service.search("test", top_k=1)
        results["search_service"] = True
        logger.info("Test SearchService réussi")
    except Exception as e:
        logger.error(f"Test SearchService échoué: {e}")

    # Tester RerankService (si fourni)
    if rerank_service:
        try:
            test_results = rerank_service.rerank("test", [{"text": "Test document"}])
            results["rerank_service"] = True
            logger.info("Test RerankService réussi")
        except Exception as e:
            logger.error(f"Test RerankService échoué: {e}")

    return results

# Point d'entrée pour les tests
if __name__ == "__main__":
    # Chemins par défaut pour les tests
    data_path = "/home/mea/Documents/modelAi/data"
    metadata_path = "/home/mea/Documents/modelAi/metadata"

    # Initialiser LangChain
    try:
        orchestrator = init_langchain(
            data_path=data_path,
            metadata_path=metadata_path
        )

        # Tester une requête simple
        result = orchestrator.generate_response(
            "Quelles sont les obligations fiscales d'une entreprise au Cameroun?"
        )

        print(f"Réponse: {result['response']}")
        print(f"Sources: {len(result.get('source_documents', []))} documents")
        print(f"Domaines: {result.get('domains', [])}")
        print(f"Suggestions: {result.get('suggestions', [])}")

        # Tester les services
        test_results = test_services(
            embedding_service=orchestrator.embedding_service,
            milvus_service=orchestrator.milvus_service,
            llm_service=orchestrator.llm_service,
            search_service=orchestrator.search_service,
            rerank_service=orchestrator.rerank_service
        )
        print("Résultats des tests des services:", test_results)

    except Exception as e:
        print(f"Erreur lors du test: {e}")