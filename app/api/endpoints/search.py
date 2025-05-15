import logging
import time
from fastapi import APIRouter, HTTPException, Depends, Query, Path, FastAPI
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import json

from app.core.config import get_settings, Settings
from app.core.dependencies import get_search_service, get_rerank_service, get_rag_system
from app.services.search_service import SearchService
from app.services.rerank_service import RerankService
from app.services.rag_system import RAGSystem
from app.models.schemas import SearchRequest, SearchResponse, RAGRequest, RAGResponse, ConversationRequest, ConversationResponse

# Configuration du logger
logger = logging.getLogger(__name__)

router = APIRouter()
rag_router = APIRouter()  # Router séparé pour les endpoints RAG
app = FastAPI()

conversation_router = APIRouter()

@router.post("/", response_model=SearchResponse)
async def search_documents(
    request: SearchRequest,
    search_service: SearchService = Depends(get_search_service),
    rerank_service: RerankService = Depends(get_rerank_service),
    settings: Settings = Depends(get_settings)
):
    """
    Recherche des documents pertinents en fonction de la requête utilisateur.
    Optionnellement, réordonne les résultats avec le service de reranking.
    """
    try:
        start_time = time.time()
        logger.info(f"🔎 Recherche pour: '{request.query}'")
        
        # Vérifier les paramètres
        top_k = min(max(1, request.top_k), settings.MAX_TOP_K)
        
        # Effectuer la recherche
        if request.filter:
            # Construire l'expression de filtrage Milvus
            filter_parts = []
            for key, value in request.filter.items():
                if isinstance(value, str):
                    filter_parts.append(f"{key} == '{value}'")
                else:
                    filter_parts.append(f"{key} == {value}")
            
            filter_expr = " && ".join(filter_parts) if filter_parts else None
            
            # Recherche avec filtre
            search_results = search_service.search(
                query=request.query,
                top_k=top_k,
                filter_expr=filter_expr
            )
        else:
            # Recherche avec expansion de requête si pas de filtre spécifique
            search_results = search_service.search_with_expansion(
                query=request.query
            )[:top_k]
        
        # Appliquer le reranking si demandé
        if request.use_rerank and search_results:
            search_results = rerank_service.rerank(
                query=request.query,
                results=search_results,
                use_llm=request.use_llm_rerank
            )
        
        # Formatage des résultats
        formatted_results = []
        for result in search_results:
            metadata = result.get("metadata", {})
            
            # Vérifier que les champs obligatoires sont présents
            result_item = {
                "text": result.get("text", ""),
                "score": result.get("rerank_score", result.get("score", 0.0)),
                "metadata": {
                    "document_id": metadata.get("document_id", ""),
                    "chunk_id": metadata.get("chunk_id", ""),
                    "filename": metadata.get("filename", "Document inconnu"),
                    "page_number": metadata.get("page_number", 0),
                    "extraction_method": metadata.get("extraction_method", "unknown"),
                    "section_type": metadata.get("section_type", None),
                    "section_number": metadata.get("section_number", None),
                    "section_title": metadata.get("section_title", None),
                }
            }
            
            formatted_results.append(result_item)
        
        # Construire la réponse
        search_time = time.time() - start_time
        logger.info(f"✅ Recherche terminée en {search_time:.2f}s - {len(formatted_results)} résultats")
        
        response = SearchResponse(
            results=formatted_results,
            query=request.query,
            total_results=len(formatted_results),
            search_time=search_time,
            metadata={
                "use_rerank": request.use_rerank,
                "use_llm_rerank": request.use_llm_rerank if request.use_rerank else False,
                "filter_applied": bool(request.filter),
                "top_k": top_k
            }
        )
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la recherche: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la recherche: {str(e)}")

@router.get("/simple", response_model=SearchResponse)
async def simple_search(
    query: str = Query(..., description="Requête utilisateur"),
    top_k: int = Query(5, description="Nombre de résultats à retourner"),
    use_rerank: bool = Query(True, description="Utiliser le reranking"),
    search_service: SearchService = Depends(get_search_service),
    rerank_service: RerankService = Depends(get_rerank_service),
    settings: Settings = Depends(get_settings)
):
    """
    Version simplifiée de l'API de recherche avec paramètres GET.
    Utilisable facilement depuis un navigateur ou un script simple.
    """
    try:
        # Réutiliser l'endpoint principal avec des valeurs par défaut
        request = SearchRequest(
            query=query,
            top_k=top_k,
            use_rerank=use_rerank,
            use_llm_rerank=False
        )
        
        return await search_documents(request, search_service, rerank_service, settings)
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la recherche simple: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Erreur lors de la recherche: {str(e)}"}
        )

@router.get("/facets")
async def get_facets(
    search_service: SearchService = Depends(get_search_service)
):
    """
    Récupère les facettes disponibles pour le filtrage (auteurs, documents, etc.).
    """
    try:
        milvus_service = search_service.milvus_service
        
        # Récupérer les statistiques de la collection
        stats = milvus_service.get_collection_stats()
        
        # Liste des documents uniques (limité aux premiers pour des raisons de performance)
        filenames = []
        doc_ids = []
        
        try:
            # Tenter de récupérer des valeurs uniques (si la fonction est disponible dans votre MilvusService)
            filenames = milvus_service.get_unique_values("filename", limit=100)
            doc_ids = milvus_service.get_unique_values("document_id", limit=100)
        except:
            # Fallback si la méthode n'est pas implémentée
            logger.warning("Méthode get_unique_values non disponible, utilisation des stats de collection.")
            pass
        
        return {
            "stats": stats,
            "facets": {
                "filenames": filenames,
                "document_ids": doc_ids
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la récupération des facettes: {e}")
        return JSONResponse(
            status_code=500, 
            content={"error": f"Erreur lors de la récupération des facettes: {str(e)}"}
        )

@router.get("/document/{document_id}")
async def get_document_info(
    document_id: str,
    search_service: SearchService = Depends(get_search_service)
):
    """
    Récupère les informations sur un document spécifique.
    """
    try:
        milvus_service = search_service.milvus_service
        
        # Construire l'expression de filtrage
        filter_expr = f"document_id == '{document_id}'"
        
        # Récupérer quelques chunks du document
        sample_chunks = []
        try:
            # Tenter de récupérer avec la méthode get_documents_by_filter si elle existe
            sample_chunks = milvus_service.get_documents_by_filter(
                filter_expr=filter_expr,
                limit=5
            )
        except AttributeError:
            # Méthode alternative si get_documents_by_filter n'existe pas
            logger.warning("Méthode get_documents_by_filter non disponible, utilisant une recherche alternative.")
            # Implémenter ici une méthode alternative pour récupérer les chunks du document
            # En utilisant d'autres méthodes disponibles dans MilvusService
            # Par exemple:  
            results = milvus_service.collection.query(
                expr=filter_expr,  
                output_fields=["text", "metadata"],
                limit=5
            )
            
            sample_chunks = [
                {"text": result.entity.get("text"), "metadata": json.loads(result.entity.get("metadata"))}  
                for result in results
            ]
        
        if not sample_chunks:
            raise HTTPException(status_code=404, detail=f"Document non trouvé: {document_id}")
        
        # Extraire les métadonnées du document à partir du premier chunk
        doc_metadata = {
            "document_id": document_id,
            "filename": sample_chunks[0].get("metadata", {}).get("filename", ""),
            "title": sample_chunks[0].get("metadata", {}).get("title", ""),
            "chunk_count": 0,  # À remplir si la méthode count_documents_by_filter existe
            "samples": [chunk.get("text", "")[:200] + "..." for chunk in sample_chunks]
        }
        
        # Tenter de compter les chunks si la méthode existe
        try:
            doc_metadata["chunk_count"] = milvus_service.count_documents_by_filter(filter_expr)
        except AttributeError:
            logger.warning("Méthode count_documents_by_filter non disponible.")
            doc_metadata["chunk_count"] = len(sample_chunks)
        
        return doc_metadata
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur lors de la récupération des informations sur le document: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la récupération des informations sur le document: {str(e)}"
        )

@router.get("/expand")
async def expand_query(
    query: str = Query(..., description="Requête à étendre"),
    search_service: SearchService = Depends(get_search_service)
):
    """
    Étend une requête avec des termes connexes pour améliorer la recherche.
    Utile pour le débogage et l'analyse.
    """
    try:
        if not hasattr(search_service, 'expand_query'):
            raise HTTPException(
                status_code=400,
                detail="Le service de recherche ne supporte pas l'expansion de requête"
            )
        
        # Étendre la requête
        expanded_queries = search_service.expand_query(query)
        
        return {
            "original_query": query,
            "expanded_queries": expanded_queries,
            "expansion_count": len(expanded_queries)
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'expansion de requête: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoints RAG - pour les requêtes de question-réponse avancées
@rag_router.post("/question", response_model=RAGResponse)
async def ask_question(
    request: RAGRequest,
    rag_system: RAGSystem = Depends(get_rag_system)
):
    """
    Répond à une question en utilisant le système RAG complet.
    Recherche les documents pertinents, applique le reranking, et génère une réponse.
    """
    try:
        # Utiliser le système RAG pour générer une réponse
        response = rag_system.generate_answer(
            query=request.query
        )
        
        # Convertir la réponse au format attendu
        rag_response = RAGResponse(
            query=response.get("query", request.query),
            answer=response.get("answer", "Aucune réponse générée."),
            source_documents=response.get("source_documents", []),
            stats=response.get("stats", {}),
            success=response.get("success", False),
            error=response.get("error", None)
        )
        
        return rag_response
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la génération de la réponse: {e}")
        
        # Retourner une réponse d'erreur formatée
        return RAGResponse(
            query=request.query,
            answer=f"Erreur lors de la génération de la réponse: {str(e)}",
            source_documents=[],
            stats={"error": str(e)},
            success=False,
            error=str(e)
        )

@rag_router.get("/question", response_model=RAGResponse)
async def ask_question_simple(
    query: str = Query(..., description="Question de l'utilisateur"),
    use_expansion: bool = Query(True, description="Utiliser l'expansion de requête"),
    use_reranking: bool = Query(True, description="Appliquer le reranking aux résultats"),
    rag_system: RAGSystem = Depends(get_rag_system)
):
    """
    Version simplifiée de l'endpoint de question-réponse, utilisable via GET.
    """
    try:
        # Créer une requête RAG
        request = RAGRequest(
            query=query,
            use_expansion=use_expansion,
            use_reranking=use_reranking
        )
        
        # Utiliser l'endpoint principal
        return await ask_question(request, rag_system)
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la génération de la réponse (GET): {e}")
        
        # Retourner une réponse d'erreur formatée
        return RAGResponse(
            query=query,
            answer=f"Erreur lors de la génération de la réponse: {str(e)}",
            source_documents=[],
            stats={"error": str(e)},
            success=False,
            error=str(e)
        )

@rag_router.get("/history", response_model=List[Dict[str, Any]])
async def get_rag_history(
    limit: int = Query(10, description="Nombre maximum d'éléments d'historique à retourner"),
    rag_system: RAGSystem = Depends(get_rag_system),
    settings: Settings = Depends(get_settings)
):
    """
    Récupère l'historique des dernières questions-réponses.
    """
    try:
        # Vérifier si la méthode existe
        if not hasattr(rag_system, '_list_saved_answers'):
            # Implémentation alternative: lire directement depuis le répertoire de sauvegarde
            import os
            import json
            
            save_dir = getattr(rag_system, 'save_dir', f"{settings.METADATA_PATH}/rag_results")
            
            if not os.path.exists(save_dir):
                return []
            
            # Lister les fichiers JSON
            files = []
            for filename in os.listdir(save_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(save_dir, filename)
                    try:
                        # Lire le timestamp depuis le nom du fichier (format: timestamp_query.json)
                        timestamp = int(filename.split('_')[0]) if '_' in filename else 0
                        
                        files.append((file_path, timestamp))
                    except:
                        continue
            
            # Trier par timestamp décroissant et limiter
            files.sort(key=lambda x: x[1], reverse=True)
            files = files[:limit]
            
            # Charger les fichiers
            history = []
            for file_path, _ in files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Ne garder que les champs importants
                        history.append({
                            "query": data.get("query", ""),
                            "answer": data.get("answer", ""),
                            "timestamp": data.get("stats", {}).get("timestamp", 0),
                            "success": data.get("success", False)
                        })
                except:
                    continue
            
            return history
        else:
            # Utiliser la méthode intégrée si disponible
            return rag_system._list_saved_answers(limit)
            
    except Exception as e:
        logger.error(f"❌ Erreur lors de la récupération de l'historique RAG: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    

@conversation_router.post("/conversation", response_model=ConversationResponse)
async def conversation_endpoint(
    request: ConversationRequest,
    rag_system: RAGSystem = Depends(get_rag_system)
):
    """
    Endpoint de conversation RAG avec gestion de l'historique.
    
    Permet une interaction conversationnelle avec le système RAG,
    en prenant en compte l'historique de la conversation.
    """
    try:
        logger.info(f"Requête entrante: Message de conversation - {request.message}")
        
        # Convertir l'historique si nécessaire
        conversation_history = request.conversation_history or []
        
        # Générer la réponse avec l'historique de conversation
        response = rag_system.handle_conversation(
            message=request.message,
            conversation_history=conversation_history
        )
        
        # Formater la réponse selon le modèle ConversationResponse
        conversation_response = ConversationResponse(
            query=request.message,
            answer=response.get('answer', 'Aucune réponse générée.'),
            explanation=response.get('explanation', ''),
            source_documents=response.get('source_documents', []),
            success=response.get('success', False),
            stats=response.get('stats', {})
        )
        
        logger.info(f"Requête traitée: Conversation - Succès: {conversation_response.success}")
        
        return conversation_response
        
    except Exception as e:
        logger.error(f"Erreur lors de la conversation: {e}")
        
        # Gérer l'erreur de manière informative
        return ConversationResponse(
            query=request.message,
            answer=f"Erreur lors de la génération de la réponse: {str(e)}",
            success=False,
            stats={"error": str(e)}
        )