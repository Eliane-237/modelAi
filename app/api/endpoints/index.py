import logging
from fastapi import APIRouter, HTTPException, Depends, Query, FastAPI
from typing import List, Dict, Any, Optional
import traceback

from app.core.config import get_settings, Settings
from app.core.dependencies import get_milvus_service
from app.services.milvus_service import MilvusService
from pydantic import BaseModel

# Configuration du logger
logger = logging.getLogger(__name__)

router = APIRouter()
app = FastAPI()

# Modèles pour les requêtes d'indexation
class IndexRequest(BaseModel):
    embeddings: List[List[float]]
    texts: Optional[List[str]] = None
    metadata: Optional[List[Dict[str, Any]]] = None

class IndexResponse(BaseModel):
    inserted_count: int
    success: bool
    message: str
    ids: Optional[List[str]] = None

@router.post("/", response_model=IndexResponse)
async def index_embeddings(
    request: IndexRequest,
    milvus_service: MilvusService = Depends(get_milvus_service)
):
    """
    Indexe des embeddings dans Milvus avec textes et métadonnées optionnels.
    """
    try:
        embeddings = request.embeddings
        
        if not embeddings:
            raise HTTPException(status_code=400, detail="La liste d'embeddings ne peut pas être vide")
        
        # Vérifier les dimensions
        embedding_dim = len(embeddings[0]) if embeddings else 0
        expected_dim = milvus_service.dim
        
        if embedding_dim != expected_dim:
            raise HTTPException(
                status_code=400, 
                detail=f"Dimension incorrecte: reçu {embedding_dim}, attendu {expected_dim}"
            )
        
        # Préparer les textes et métadonnées
        texts = request.texts or [""] * len(embeddings)
        metadata = request.metadata or [{}] * len(embeddings)
        
        # S'assurer que toutes les listes ont la même longueur
        if len(texts) != len(embeddings) or len(metadata) != len(embeddings):
            raise HTTPException(
                status_code=400, 
                detail="Les listes d'embeddings, textes et métadonnées doivent avoir la même longueur"
            )
        
        # Insérer dans Milvus
        logger.info(f"Indexation de {len(embeddings)} embeddings dans Milvus")
        result = milvus_service.insert_documents_with_metadata(embeddings, texts, metadata)
        
        logger.info(f"Indexation réussie: {result} documents indexés")
        return IndexResponse(
            inserted_count=result,
            success=True,
            message=f"{result} embeddings indexés avec succès"
        )
    
    except ValueError as e:
        logger.error(f"Erreur de validation: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Erreur lors de l'indexation des embeddings: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/document/{document_id}")
async def delete_document_embeddings(
    document_id: str,
    milvus_service: MilvusService = Depends(get_milvus_service)
):
    """
    Supprime tous les embeddings associés à un document spécifique.
    """
    try:
        # Supprimer par document_id
        logger.info(f"Suppression des embeddings pour le document: {document_id}")
        deleted_count = milvus_service.delete_by_document_id(document_id)
        
        return {
            "document_id": document_id,
            "deleted_count": deleted_count,
            "success": True,
            "message": f"{deleted_count} embeddings supprimés pour le document {document_id}"
        }
    except Exception as e:
        logger.error(f"Erreur lors de la suppression des embeddings: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_collection_stats(
    milvus_service: MilvusService = Depends(get_milvus_service)
):
    """
    Récupère les statistiques de la collection Milvus.
    """
    try:
        # Vérifier d'abord si la connexion est active
        if not getattr(milvus_service, 'collection', None):
            # Tentative de reconnexion explicite
            logger.warning("Collection Milvus non disponible, tentative de reconnexion...")
            try:
                # Cette partie dépend de l'implémentation de votre MilvusService
                # Idéalement, il devrait y avoir une méthode reconnect() ou similaire
                if hasattr(milvus_service, '_connect_to_milvus'):
                    milvus_service._connect_to_milvus()
                if hasattr(milvus_service, '_ensure_collection'):
                    milvus_service._ensure_collection()
            except Exception as reconnect_error:
                logger.error(f"Échec de la reconnexion: {reconnect_error}")
                return {
                    "collection_name": milvus_service.collection_name,
                    "status": "disconnected",
                    "error": "Impossible de se connecter à Milvus",
                    "host": getattr(milvus_service, 'host', 'unknown'),
                    "port": getattr(milvus_service, 'port', 'unknown')
                }
        
        # Obtenir les statistiques
        stats = milvus_service.get_collection_stats()
        
        return {
            "collection_name": milvus_service.collection_name,
            "status": "connected",
            "stats": stats,
            "entity_count": stats.get("row_count", 0),
            "embedding_dim": milvus_service.dim,
            "host": getattr(milvus_service, 'host', 'unknown'),
            "port": getattr(milvus_service, 'port', 'unknown')
        }
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des statistiques: {e}")
        logger.error(traceback.format_exc())
        # Retourner plus d'informations diagnostiques plutôt que de lever une exception
        return {
            "collection_name": getattr(milvus_service, 'collection_name', 'unknown'),
            "status": "error",
            "error": str(e),
            "host": getattr(milvus_service, 'host', 'unknown'),
            "port": getattr(milvus_service, 'port', 'unknown')
        }

@router.post("/flush")
async def flush_collection(
    milvus_service: MilvusService = Depends(get_milvus_service)
):
    """
    Force l'écriture de toutes les données en mémoire sur le disque.
    """
    try:
        milvus_service.collection.flush()
        return {"success": True, "message": "Collection vidée avec succès"}
    except Exception as e:
        logger.error(f"Erreur lors du flush de la collection: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/connection-check")
async def check_milvus_connection(
    settings: Settings = Depends(get_settings)
):
    """
    Endpoint de diagnostic pour vérifier la connexion à Milvus.
    """
    try:
        # Récupérer les paramètres directement depuis les settings
        host = settings.MILVUS_HOST
        port = settings.MILVUS_PORT
        collection = settings.MILVUS_COLLECTION
        
        # Tenter une connexion explicite pour diagnostiquer
        from pymilvus import connections
        connections.connect("default", host=host, port=str(port), timeout=10.0)
        
        # Vérifier si la connexion a réussi
        server_info = "Inconnu"
        try:
            from pymilvus import utility
            server_info = utility.get_server_version()
        except:
            server_info = "Connecté, mais impossible de récupérer la version"
            
        return {
            "status": "connected",
            "message": "Connexion à Milvus réussie",
            "server_version": server_info,
            "config": {
                "host": host,
                "port": port,
                "collection": collection
            }
        }
    except Exception as e:
        logger.error(f"Erreur lors du test de connexion Milvus: {e}")
        logger.error(traceback.format_exc())
        return {
            "status": "error",
            "message": f"Erreur de connexion: {str(e)}",
            "config": {
                "host": settings.MILVUS_HOST,
                "port": settings.MILVUS_PORT,
                "collection": settings.MILVUS_COLLECTION
            },
            "error_type": type(e).__name__
        }