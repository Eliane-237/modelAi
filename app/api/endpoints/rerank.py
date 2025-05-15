import logging
from fastapi import APIRouter, HTTPException, Depends, Query, FastAPI
from typing import List, Dict, Any

from app.core.dependencies import get_rerank_service
from app.services.rerank_service import RerankService
from pydantic import BaseModel, Field

# Configuration du logger
logger = logging.getLogger(__name__)

router = APIRouter()
app = FastAPI()

# Modèles pour la requête de reranking
class RerankRequest(BaseModel):
    query: str = Field(..., description="Requête utilisateur")
    results: List[Dict[str, Any]] = Field(..., description="Résultats à réordonner")
    use_llm: bool = Field(False, description="Utiliser le LLM pour le reranking")

class RerankResponse(BaseModel):
    results: List[Dict[str, Any]] = Field(..., description="Résultats réordonnés")
    original_count: int = Field(..., description="Nombre de résultats originaux")
    query: str = Field(..., description="Requête originale")
    use_llm: bool = Field(..., description="LLM utilisé pour le reranking")

@router.post("/", response_model=RerankResponse)
async def rerank_results(
    request: RerankRequest,
    rerank_service: RerankService = Depends(get_rerank_service)
):
    """
    Réordonne les résultats de recherche en fonction de leur pertinence.
    """
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="La requête ne peut pas être vide")
        
        if not request.results:
            raise HTTPException(status_code=400, detail="La liste de résultats ne peut pas être vide")
        
        # Appliquer le reranking
        logger.info(f"Reranking de {len(request.results)} résultats pour la requête: '{request.query}'")
        
        reranked_results = rerank_service.rerank(
            query=request.query,
            results=request.results,
            use_llm=request.use_llm
        )
        
        # Construire la réponse
        logger.info(f"Reranking terminé avec succès: {len(reranked_results)} résultats")
        
        return RerankResponse(
            results=reranked_results,
            original_count=len(request.results),
            query=request.query,
            use_llm=request.use_llm
        )
    
    except Exception as e:
        logger.error(f"Erreur lors du reranking: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/legal-score")
async def legal_relevance_score(
    query: str = Query(..., description="Requête utilisateur"),
    text: str = Query(..., description="Texte à évaluer"),
    rerank_service: RerankService = Depends(get_rerank_service)
):
    """
    Calcule le score de pertinence juridique d'un texte par rapport à une requête.
    Utile pour le débogage et l'analyse.
    """
    try:
        if not hasattr(rerank_service, '_calculate_legal_relevance_score'):
            raise HTTPException(
                status_code=400,
                detail="Le service de reranking ne dispose pas de la méthode de score juridique"
            )
        
        # Calculer le score
        score = rerank_service._calculate_legal_relevance_score(query, text)
        
        # Calculer aussi les autres scores pour comparaison
        scores = {}
        
        if hasattr(rerank_service, '_calculate_bm25_score'):
            scores["bm25"] = rerank_service._calculate_bm25_score(query, text)
        
        # Score article si applicable
        if hasattr(rerank_service, '_calculate_article_match_score'):
            scores["article_match"] = rerank_service._calculate_article_match_score(query, text)
        
        # TF-IDF si applicable (attention, nécessite une liste de documents)
        if hasattr(rerank_service, '_calculate_tfidf_score'):
            try:
                tfidf_scores = rerank_service._calculate_tfidf_score(query, [text])
                scores["tfidf"] = tfidf_scores[0] if tfidf_scores else 0.0
            except:
                scores["tfidf"] = "N/A (requiert une liste de documents)"
        
        # LLM score si activé
        llm_score = None
        if hasattr(rerank_service, '_get_llm_score') and rerank_service.llm_available:
            try:
                llm_score = rerank_service._get_llm_score(query, text)
                scores["llm"] = llm_score
            except:
                scores["llm"] = "N/A (erreur de génération)"
        
        return {
            "query": query,
            "text_length": len(text),
            "text_preview": text[:100] + "..." if len(text) > 100 else text,
            "legal_relevance_score": score,
            "other_scores": scores
        }
    
    except Exception as e:
        logger.error(f"Erreur lors du calcul du score de pertinence juridique: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/cache-info")
async def get_cache_info(
    rerank_service: RerankService = Depends(get_rerank_service)
):
    """
    Récupère des informations sur le cache du service de reranking.
    """
    try:
        cache_size = len(getattr(rerank_service, 'score_cache', {}))
        max_cache_size = getattr(rerank_service, 'cache_size', 0)
        
        return {
            "cache_entries": cache_size,
            "max_cache_size": max_cache_size,
            "cache_usage_percent": (cache_size / max_cache_size * 100) if max_cache_size > 0 else 0,
            "llm_available": getattr(rerank_service, 'llm_available', False)
        }
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des informations de cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))