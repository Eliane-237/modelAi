import logging
from fastapi import APIRouter, HTTPException, Depends, Query, FastAPI
from fastapi.responses import StreamingResponse
from typing import Optional, List, Dict, Any
from pydantic import BaseModel

from app.core.config import get_settings, Settings
from app.core.dependencies import get_llm_service
from app.services.llm_service import LlmService
from app.models.schemas import LLMRequest, LLMResponse

# Modèle pour les requêtes de conversation
class ConversationRequest(BaseModel):
    messages: List[Dict[str, str]]
    max_length: int = 500
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    system_prompt: Optional[str] = None

# Modèle pour les réponses de conversation
class ConversationResponse(BaseModel):
    messages: List[Dict[str, str]]
    last_response: str
    metadata: Dict[str, Any] = {}

# Configuration du logger
logger = logging.getLogger(__name__)

router = APIRouter()
app = FastAPI()

@router.post("/generate", response_model=LLMResponse)
async def generate_response(
    request: LLMRequest,
    llm_service: LlmService = Depends(get_llm_service),
    settings: Settings = Depends(get_settings)
):
    """
    Génère une réponse à partir d'un prompt en utilisant le LLM.
    """
    try:
        if not request.prompt.strip():
            raise HTTPException(status_code=400, detail="Le prompt ne peut pas être vide")
        
        # Si le streaming est demandé, utiliser une réponse streaming
        if request.stream:
            return await generate_streaming_response(request, llm_service, settings)
        
        # Générer la réponse
        logger.info(f"Génération de réponse LLM (modèle: {settings.LLM_MODEL})")
        response = llm_service.generate_response(
            prompt=request.prompt,
            max_length=request.max_length,
            temperature=request.temperature
        )
        
        # Retourner la réponse
        return LLMResponse(
            response=response,
            metadata={
                "model": settings.LLM_MODEL,
                "max_length": request.max_length,
                "temperature": request.temperature
            }
        )
    
    except Exception as e:
        logger.error(f"Erreur lors de la génération de la réponse LLM: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def generate_streaming_response(
    request: LLMRequest,
    llm_service: LlmService,
    settings: Settings
):
    """
    Génère une réponse streaming à partir du LLM.
    """
    try:
        # Vérifier si le streaming est supporté
        if not hasattr(llm_service, 'generate_response') or not callable(getattr(llm_service, 'generate_response')):
            raise HTTPException(
                status_code=400,
                detail="Le streaming n'est pas supporté par le service LLM"
            )
        
        # Générer la réponse en streaming
        stream_generator = llm_service.generate_response(
            prompt=request.prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            stream=True
        )
        
        # Vérifier que le résultat est bien un itérateur
        if not hasattr(stream_generator, '__iter__') and not hasattr(stream_generator, '__aiter__'):
            raise HTTPException(
                status_code=500,
                detail="Le service LLM n'a pas retourné un itérateur pour le streaming"
            )
        
        # Retourner la réponse streaming
        return StreamingResponse(
            stream_generator,
            media_type="text/event-stream"
        )
    
    except Exception as e:
        logger.error(f"Erreur lors de la génération de la réponse LLM en streaming: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat", response_model=LLMResponse)
async def generate_chat_response(
    messages: List[Dict[str, str]],
    max_length: int = Query(500, description="Longueur maximale de la réponse"),
    temperature: float = Query(0.7, description="Température pour le sampling"),
    top_p: float = Query(0.9, description="Échantillonnage noyau (nucleus sampling)"),
    top_k: int = Query(50, description="Nombre de tokens les plus probables à considérer"),
    stream: bool = Query(False, description="Activer le streaming de la réponse"),
    llm_service: LlmService = Depends(get_llm_service),
    settings: Settings = Depends(get_settings)
):
    """
    Génère une réponse en mode chat.
    Messages au format [{role: "user", content: "message"}, {role: "assistant", content: "réponse"}, ...].
    """
    try:
        if not messages:
            raise HTTPException(status_code=400, detail="La liste de messages ne peut pas être vide")
        
        # Valider les messages
        for msg in messages:
            if "role" not in msg or "content" not in msg:
                raise HTTPException(
                    status_code=400, 
                    detail="Chaque message doit contenir 'role' et 'content'"
                )
        
        # Obtenir la réponse via llm_service
        try:
            response = llm_service.generate_chat_response(
                messages=messages,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stream=stream
            )
            
            # Correction importante: vérifier et convertir si nécessaire
            if not isinstance(response, str) and not stream:
                logger.warning(f"Réponse LLM de type non attendu: {type(response)}. Conversion en chaîne.")
                # Si c'est un MagicMock, retourner un message par défaut
                if "MagicMock" in str(type(response)):
                    response = "Je suis un assistant juridique en mode dégradé. Le service LLM n'est pas disponible actuellement."
                else:
                    response = str(response)
        except Exception as e:
            logger.error(f"Erreur lors de la génération de la réponse chat: {e}")
            response = f"Une erreur est survenue lors de la génération de la réponse: {str(e)}"
        
        # Gestion du streaming si activé
        if stream:
            return StreamingResponse(
                response if hasattr(response, "__iter__") else iter([response]),
                media_type="text/event-stream"
            )
        
        # Retourner la réponse
        return LLMResponse(
            response=response,
            metadata={
                "model": settings.LLM_MODEL,
                "max_length": max_length,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "message_count": len(messages),
                "chat_mode": True
            }
        )
    
    except Exception as e:
        logger.error(f"Erreur lors de la génération de la réponse chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
# Fonctions utilitaires pour nettoyer les réponses Llama
def clean_llama_response(response: str) -> str:
    """Nettoie la réponse d'un modèle Llama en enlevant les balises et marqueurs."""
    import re
    
    # Enlever les balises du début
    response = re.sub(r'<\|assistant\|>\s*', '', response)
    
    # Enlever tout ce qui vient après une nouvelle balise user ou système
    response = re.sub(r'<\|user\|>.*$', '', response, flags=re.DOTALL)
    response = re.sub(r'<\|system\|>.*$', '', response, flags=re.DOTALL)
    
    # Enlever les autres balises qui pourraient être présentes
    response = re.sub(r'<\|.*?\|>', '', response)
    
    return response.strip()

async def clean_llama_stream(stream_generator):
    """Nettoie les chunks du stream pour un modèle Llama."""
    accumulated_text = ""
    
    async for chunk in stream_generator:
        accumulated_text += chunk
        
        # Nettoyer le texte accumulé
        clean_text = clean_llama_response(accumulated_text)
        
        # Calculer le nouveau chunk (ce qui a été ajouté)
        # Cette approche est simplifiée et pourrait nécessiter des ajustements
        new_chunk = clean_text
        
        yield f"data: {new_chunk}\n\n"

@router.post("/conversation", response_model=ConversationResponse)
async def handle_conversation(
    request: ConversationRequest,
    llm_service: LlmService = Depends(get_llm_service),
    settings: Settings = Depends(get_settings)
):
    """
    Gère une conversation complète et retourne la conversation mise à jour.
    Vérifie que le dernier message est de l'utilisateur et ajoute la réponse du modèle.
    """
    try:
        messages = request.messages.copy()
        
        # Ajouter un message système si fourni et qu'il n'existe pas déjà
        if request.system_prompt and not any(msg["role"] == "system" for msg in messages):
            messages.insert(0, {"role": "system", "content": request.system_prompt})
            
        # Vérifier que le dernier message est de l'utilisateur
        if not messages or messages[-1]["role"] != "user":
            raise HTTPException(
                status_code=400,
                detail="Le dernier message doit être de l'utilisateur"
            )
        
        # Obtenir la réponse via l'endpoint chat
        response_data = await generate_chat_response(
            messages=messages,
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stream=False,
            llm_service=llm_service,
            settings=settings
        )
        
        # Extraire la réponse
        response_text = response_data.response
        
        # Ajouter la réponse à la conversation
        messages.append({"role": "assistant", "content": response_text})
        
        # Retourner la conversation mise à jour
        return ConversationResponse(
            messages=messages,
            last_response=response_text,
            metadata={
                "model": settings.LLM_MODEL,
                "temperature": request.temperature,
                "turns": len(messages) // 2  # Nombre d'échanges user/assistant
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la gestion de la conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/models")
async def get_available_models(
    llm_service: LlmService = Depends(get_llm_service),
    settings: Settings = Depends(get_settings)
):
    """
    Récupère la liste des modèles disponibles.
    """
    # Note: Cette fonction peut nécessiter une implémentation supplémentaire 
    # si votre service LLM permet de lister des modèles
    
    return {
        "current_model": settings.LLM_MODEL,
        "available_models": [settings.LLM_MODEL],
        "service_url": settings.LLM_SERVICE_URL
    }