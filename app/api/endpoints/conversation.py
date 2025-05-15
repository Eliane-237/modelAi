from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Any, Optional
import logging

from app.core.dependencies import get_rag_system
from app.services.rag_system import RAGSystem
from app.models.schemas import ConversationRequest, ConversationResponse

# Configuration du logger
logger = logging.getLogger(__name__)

# Créer un routeur pour les endpoints de conversation
conversation_router = APIRouter()

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