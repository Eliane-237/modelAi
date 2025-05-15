import logging
from fastapi import APIRouter, HTTPException, Depends, FastAPI
from typing import List

from app.core.config import get_settings, Settings
from app.core.dependencies import get_embedding_service
from app.services.embedding_service import EmbeddingService
from app.models.schemas import TextsRequest, EmbeddingsResponse

# Configuration du logger
logger = logging.getLogger(__name__)

router = APIRouter()
app = FastAPI()
@router.post("/", response_model=EmbeddingsResponse)
async def generate_embeddings(
    request: TextsRequest,
    embedding_service: EmbeddingService = Depends(get_embedding_service)
):
    """
    Génère des embeddings pour les textes fournis.
    """
    try:
        if not request.texts:
            raise HTTPException(status_code=400, detail="La liste de textes ne peut pas être vide")
        
        logger.info(f"Génération d'embeddings pour {len(request.texts)} textes")
        embeddings = embedding_service.generate_embeddings(request.texts)
        
        if not embeddings:
            raise HTTPException(status_code=500, detail="Échec de la génération des embeddings")
        
        logger.info(f"Embeddings générés avec succès: {len(embeddings)} vecteurs")
        return EmbeddingsResponse(embeddings=embeddings)
    
    except ValueError as e:
        logger.error(f"Erreur de validation: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Erreur lors de la génération des embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch", response_model=EmbeddingsResponse)
async def generate_embeddings_batch(
    texts: List[str],
    embedding_service: EmbeddingService = Depends(get_embedding_service)
):
    """
    Alternative simplifiée pour générer des embeddings.
    Accepte directement une liste de textes.
    """
    try:
        if not texts:
            raise HTTPException(status_code=400, detail="La liste de textes ne peut pas être vide")
        
        logger.info(f"Génération d'embeddings batch pour {len(texts)} textes")
        embeddings = embedding_service.generate_embeddings(texts)
        
        if not embeddings:
            raise HTTPException(status_code=500, detail="Échec de la génération des embeddings")
        
        logger.info(f"Embeddings batch générés avec succès: {len(embeddings)} vecteurs")
        return EmbeddingsResponse(embeddings=embeddings)
    
    except ValueError as e:
        logger.error(f"Erreur de validation dans la génération batch: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Erreur lors de la génération batch des embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/info")
async def get_embedding_info(
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    settings: Settings = Depends(get_settings)
):
    """
    Récupère des informations sur le service d'embedding.
    """
    try:
        # Cette fonction peut être améliorée si votre EmbeddingService dispose
        # de méthodes pour obtenir des informations sur le modèle
        info = {
            "embedding_dim": settings.EMBEDDING_DIM,
            "server_url": settings.EMBEDDING_SERVICE_URL,
            "model": "BGE-M3",  # À remplacer par une valeur dynamique si disponible
            "status": "online"
        }
        
        return info
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des infos d'embedding: {e}")
        raise HTTPException(status_code=500, detail=str(e))