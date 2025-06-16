"""
Routes FastAPI pour l'interaction avec l'agent RAG juridique camerounais.
Ce module fournit des endpoints spécifiques pour utiliser les capacités avancées
d'agent de LexCam.
"""

import logging
import time
import asyncio
import json
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel, Field

from app.core.config import get_settings, Settings
from app.core.dependencies import get_embedding_service, get_milvus_service, get_llm_service, get_rerank_service
from app.core.dependencies import get_langchain_service_from_deps
from app.services.langchain_service import LangChainService

# Configuration du logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Création du router FastAPI
agent_router = APIRouter()

# Modèles Pydantic pour les requêtes et réponses
class AgentRequest(BaseModel):
    query: str = Field(..., description="Requête ou question de l'utilisateur")
    session_id: Optional[int] = Field(None, description="ID de session pour maintenir la conversation")
    streaming: bool = Field(False, description="Activer le streaming de la réponse")
    use_planning: bool = Field(True, description="Activer la planification pour les questions complexes")
    user_id: Optional[str] = Field(None, description="Identifiant de l'utilisateur (facultatif)")

class SourceDocument(BaseModel):
    """Modèle pour les documents sources sérialisables"""
    text: str = Field(..., description="Contenu du document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Métadonnées du document")
    source: Optional[str] = Field(None, description="Source du document")
    page_number: Optional[str] = Field(None, description="Numéro de page")

class AgentResponse(BaseModel):
    query: str = Field(..., description="Requête originale")
    response: str = Field(..., description="Réponse générée")
    session_id: int = Field(..., description="ID de session")
    source_documents: List[SourceDocument] = Field(default_factory=list, description="Documents sources")
    response_time: float = Field(..., description="Temps de réponse")
    domains: List[str] = Field(default_factory=list, description="Domaines juridiques")
    intent: Optional[str] = Field(None, description="Intention de la requête")
    plan: Optional[Dict[str, Any]] = Field(None, description="Plan de réponse")
    error: Optional[str] = Field(None, description="Message d'erreur")

class SessionInfo(BaseModel):
    session_id: int = Field(..., description="ID de la session")
    first_query: str = Field(..., description="Première requête de la session")
    start_time: float = Field(default_factory=time.time, description="Timestamp de début de session")
    last_time: float = Field(default_factory=time.time, description="Timestamp de dernière interaction")
    interactions: int = Field(default=0, description="Nombre d'interactions dans la session")

class SessionsResponse(BaseModel):
    sessions: List[SessionInfo] = Field([], description="Liste des sessions disponibles")
    count: int = Field(..., description="Nombre total de sessions")

class FeedbackModel(BaseModel):
    session_id: int = Field(..., description="ID de la session")
    message_id: str = Field(..., description="ID du message évalué")
    rating: int = Field(..., description="Évaluation (1-5)")
    comment: Optional[str] = Field(None, description="Commentaire facultatif")

@agent_router.post("/chat", response_model=AgentResponse)
async def agent_chat(
    request: AgentRequest,
    langchain_service: LangChainService = Depends(get_langchain_service_from_deps)
):
    try:
        # Vérifier la requête
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="La requête ne peut pas être vide")

        # Charger la session si ID fourni
        if request.session_id:
            langchain_service.load_conversation_history(request.session_id)

        # Générer la réponse avec l'approche agent
        start_time = time.time()
        result = langchain_service.generate_response(request.query, streaming=False)
        response_time = time.time() - start_time

        # Construire la réponse
        return AgentResponse(
            query=result.get('query', request.query),
            response=result.get('response', "Désolé, je n'ai pas pu générer de réponse."),
            session_id=langchain_service.session_id,
            source_documents=result.get('source_documents', []),  # Devrait maintenant être sérialisable
            response_time=response_time,
            domains=result.get('domains', []),
            intent=result.get('intent', None),
            plan=result.get('plan', None),
            error=result.get('error')
        )
    
    except Exception as e:
        logger.error(f"Erreur lors de l'interaction avec l'agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Ajoutez cette fonction GET à votre fichier app/api/endpoints/agent.py
# Placez-la après vos autres endpoints

import json

@agent_router.get("/chat/stream")
async def agent_chat_stream_get(
    query: str = Query(..., description="Requête utilisateur"),
    session_id: Optional[str] = Query(None, description="ID de session (peut être vide)"),
    streaming: bool = Query(True, description="Activer le streaming"),
    langchain_service: LangChainService = Depends(get_langchain_service_from_deps)
):
    """
    Version GET du streaming pour faciliter l'utilisation avec EventSource.
    Utilise Server-Sent Events pour le streaming de la réponse.
    """
    try:
        # Vérifier la requête
        if not query.strip():
            raise HTTPException(status_code=400, detail="La requête ne peut pas être vide")
        
        # Convertir session_id si fourni et non vide
        parsed_session_id = None
        if session_id and session_id.strip():
            try:
                parsed_session_id = int(session_id)
            except ValueError:
                # Si ce n'est pas un entier, on ignore
                pass
        
        # Charger la session si ID fourni
        if parsed_session_id:
            langchain_service.load_conversation_history(parsed_session_id)
        
        # Fonction générateur pour les événements SSE
        async def stream_generator():
            try:
                # Envoyer l'événement de début
                start_time = time.time()
                session_id_to_use = langchain_service.session_id or int(time.time())
                
                start_data = {
                    "session_id": session_id_to_use,
                    "query": query,
                    "timestamp": time.time()
                }
                
                yield f"event: start\n"
                yield f"data: {json.dumps(start_data)}\n\n"
                
                # Générer la réponse avec streaming
                result = langchain_service.generate_response(query, streaming=True)
                
                # Vérifier si le streaming est supporté
                if result.get("streaming", False) and "response_generator" in result:
                    # Streaming réel depuis le LLM
                    response_generator = result["response_generator"]
                    
                    for token in response_generator:
                        yield f"event: token\n"
                        yield f"data: {token}\n\n"
                        # Petit délai pour éviter de surcharger le client
                        await asyncio.sleep(0.01)
                else:
                    # Fallback : simulation du streaming
                    response_text = result.get("response", "Je n'ai pas pu générer de réponse.")
                    words = response_text.split()
                    
                    for i, word in enumerate(words):
                        token = word + (" " if i < len(words) - 1 else "")
                        yield f"event: token\n"
                        yield f"data: {token}\n\n"
                        await asyncio.sleep(0.05)  # Délai pour simuler la génération
                
                # Envoyer l'événement de fin avec les métadonnées
                end_time = time.time()
                end_data = {
                    "session_id": session_id_to_use,
                    "source_documents": result.get('source_documents', []),
                    "domains": result.get('domains', []),
                    "intent": result.get('intent', ''),
                    "response_time": end_time - start_time
                }
                
                yield f"event: end\n"
                yield f"data: {json.dumps(end_data)}\n\n"
                
            except Exception as e:
                logger.error(f"Erreur lors du streaming: {e}")
                error_data = {"error": str(e)}
                yield f"event: error\n"
                yield f"data: {json.dumps(error_data)}\n\n"
        
        # Retourner la réponse en streaming avec les bons headers
        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control"
            }
        )
        
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation du streaming: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@agent_router.get("/sessions", response_model=SessionsResponse)
async def list_agent_sessions(
    limit: int = Query(10, description="Nombre maximum de sessions à retourner"),
    langchain_service: LangChainService = Depends(get_langchain_service_from_deps)
):
    """
    Liste les sessions de conversation disponibles.
    """
    try:
        # Récupérer la liste des sessions
        sessions = langchain_service.list_available_sessions()
        
        # Limiter le nombre de sessions retournées
        sessions = sessions[:limit]
        
        return SessionsResponse(
            sessions=sessions,
            count=len(sessions)
        )
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@agent_router.post("/reset")
async def reset_agent_session(
    session_id: Optional[int] = None,
    langchain_service: LangChainService = Depends(get_langchain_service_from_deps)
):
    """
    Réinitialise une session de conversation.
    Si session_id est fourni, charge cette session puis la réinitialise.
    Sinon, réinitialise la session actuelle.
    """
    try:
        # Si un ID de session est fourni, charger d'abord cette session
        if session_id:
            langchain_service.load_conversation_history(session_id)
        
        # Réinitialiser la conversation
        langchain_service.reset_conversation()
        
        return {
            "success": True,
            "message": "Session réinitialisée avec succès",
            "new_session_id": langchain_service.session_id
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de la réinitialisation de la session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@agent_router.get("/sessions/{session_id}")
async def get_agent_session_info(
    session_id: int,
    langchain_service: LangChainService = Depends(get_langchain_service_from_deps)
):
    """
    Récupère les informations d'une session spécifique.
    """
    try:
        # Charger temporairement la session pour obtenir ses informations
        success = langchain_service.load_conversation_history(session_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Session {session_id} non trouvée")
        
        # Récupérer l'historique des requêtes
        query_history = langchain_service.get_query_history()
        
        # Récupérer les messages de la mémoire
        messages = langchain_service.memory.messages
        
        return {
            "session_id": session_id,
            "query_history": query_history,
            "messages": messages,
            "message_count": len(messages),
            "domains": list(langchain_service.memory.domains)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des informations de session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@agent_router.post("/feedback")
async def submit_agent_feedback(
    feedback: FeedbackModel,
    settings: Settings = Depends(get_settings)
):
    """
    Enregistre le feedback utilisateur pour améliorer le système.
    """
    try:
        # Créer le répertoire de feedback s'il n'existe pas
        import os
        feedback_dir = f"{settings.METADATA_PATH}/feedback"
        os.makedirs(feedback_dir, exist_ok=True)
        
        # Compléter les données de feedback
        feedback_data = feedback.dict()
        feedback_data["timestamp"] = time.time()
        
        # Enregistrer le feedback dans un fichier JSON
        import json
        
        # Créer un nom de fichier unique
        filename = f"{feedback_dir}/feedback_{feedback_data['timestamp']}_{feedback.message_id}.json"
        
        # Enregistrer le feedback
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(feedback_data, f, ensure_ascii=False, indent=2)
        
        return {"status": "success", "message": "Feedback enregistré avec succès"}
        
    except Exception as e:
        logger.error(f"Erreur lors de l'enregistrement du feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@agent_router.post("/analyze")
async def analyze_query(
    query: str,
    langchain_service: LangChainService = Depends(get_langchain_service_from_deps)
):
    """
    Analyse une requête sans générer de réponse.
    Utile pour comprendre comment l'agent interprète la question.
    """
    try:
        # Utiliser la méthode d'identification d'intention
        intent_analysis = langchain_service._identify_query_intent(query)
        
        # Planifier pour les requêtes complexes
        plan = None
        if len(query) > 20:
            plan = langchain_service._plan_complex_query(query, intent_analysis)
        
        return {
            "query": query,
            "domains": intent_analysis.get("domains", []),
            "intent": intent_analysis.get("intent", ""),
            "relevant_texts": intent_analysis.get("relevant_texts", []),
            "analysis": intent_analysis.get("analysis", ""),
            "plan": plan
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse de la requête: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@agent_router.get("/tools")
async def list_agent_tools(
    langchain_service: LangChainService = Depends(get_langchain_service_from_deps)
):
    """
    Liste les outils disponibles pour l'agent.
    """
    try:
        tools = langchain_service.tools
        
        formatted_tools = []
        for tool in tools:
            formatted_tools.append({
                "name": tool.name,
                "description": tool.description
            })
        
        return {
            "tools": formatted_tools,
            "count": len(formatted_tools)
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des outils: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@agent_router.post("/tool/{tool_name}")
async def execute_agent_tool(
    tool_name: str,
    query: str,
    langchain_service: LangChainService = Depends(get_langchain_service_from_deps)
):
    """
    Exécute un outil spécifique de l'agent.
    """
    try:
        # Rechercher l'outil par son nom
        tool = None
        for t in langchain_service.tools:
            if t.name == tool_name:
                tool = t
                break
        
        if not tool:
            raise HTTPException(status_code=404, detail=f"Outil '{tool_name}' non trouvé")
        
        # Exécuter l'outil
        result = tool.func(query)
        
        return {
            "tool": tool_name,
            "query": query,
            "result": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution de l'outil {tool_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    

    