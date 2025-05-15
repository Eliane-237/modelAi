import requests
import logging
import time
import json
from typing import Optional, Dict, Any, List, Generator, Union
from typing import Iterator
from concurrent.futures import ThreadPoolExecutor, TimeoutError as ConcurrentTimeoutError

# Add these imports for streaming
from fastapi.responses import StreamingResponse
from starlette.types import Send, Receive, Scope
from starlette.responses import StreamingResponse as StarletteStreamingResponse


# Configuration du logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

class LlmService:
    def __init__(
        self, 
        model_name: str = "llama3.2:latest", 
        server_url: str = "http://10.100.212.118:8001/generate" , # URL du serveur LLM distant
        timeout: int = 60
    ):
        """
        Initialise le service LLM avec un modèle et une URL de serveur spécifiques.
        
        Args:
            model_name: Nom du modèle à utiliser
            server_url: URL du serveur LLM distant
        """
        self.model_name = model_name
        self.server_url = server_url
        self.timeout = timeout
        
        # Vérifier la connexion initiale
        self._check_connection()

    def _check_connection(self):
        """
        Vérifie la connexion au serveur LLM distant.
        """
        try:
            # Envoi d'une requête de test minimale
            test_payload = {
                "model": self.model_name,
                "prompt": "Test de connexion",
                "max_length": 10,
                "stream": False
            }
            response = requests.post(self.server_url, json=test_payload, timeout=200)
            response.raise_for_status()
            logger.info(f"✅ Connexion réussie au serveur LLM sur {self.server_url}")
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Impossible de se connecter au serveur LLM : {e}")
            raise RuntimeError(f"Erreur de connexion au serveur LLM : {e}")

    def generate_response(
        self,
        prompt: str,
        max_length: int = 500,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        stream: bool = False,  # Défini sur False par défaut
        dynamic_timeout: bool = True,
        max_wait_time: int = 300,  # Temps maximum d'attente (5 minutes)
        **kwargs
    ) -> Union[str, Iterator[str]]:
        """
        Génération de réponse avec gestion dynamique du timeout.
        
        Args:
            dynamic_timeout: Active la gestion dynamique du timeout
            max_wait_time: Temps maximum d'attente avant abandon
            stream: Si True, retourne un générateur qui produit la réponse par morceaux
        """
        def make_request():
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "max_length": max_length,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "stream": stream,
                **kwargs
            }
            
            try:
                # Calcul dynamique du timeout
                start_time = time.time()
                
                response = requests.post(
                    self.server_url, 
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    stream=stream,
                    timeout=None  # Pas de timeout prédéfini
                )
                
                response.raise_for_status()
                return response
                    
            except Exception as e:
                logger.error(f"Erreur de requête : {e}")
                raise

        # Gestion du streaming
        if stream:
            try:
                response = make_request()
                
                def stream_generator():
                    for line in response.iter_lines():
                        if line:
                            line_text = line.decode('utf-8')
                            # Gérer la réponse du format serveur
                            if line_text.startswith('data: '):
                                line_text = line_text[6:]
                            
                            try:
                                # Essayer de parser comme JSON si c'est le format du serveur
                                json_data = json.loads(line_text)
                                if "token" in json_data:
                                    yield json_data["token"]
                                elif "generated_text" in json_data:
                                    yield json_data["generated_text"]
                                else:
                                    yield line_text
                            except json.JSONDecodeError:
                                # Si ce n'est pas du JSON, renvoyer le texte brut
                                yield line_text
                
                return stream_generator()
                
            except Exception as e:
                logger.error(f"Erreur de streaming : {e}")
                def error_generator():
                    yield f"Erreur: {str(e)}"
                return error_generator()
        
        # Mode non-streaming avec gestion de timeout
        elif dynamic_timeout:
            with ThreadPoolExecutor(max_workers=1) as executor:
                try:
                    future = executor.submit(make_request)
                    response = future.result(timeout=max_wait_time)
                    return response.json().get("generated_text", "")
                    
                except ConcurrentTimeoutError:
                    logger.warning(f"Temps d'attente maximum de {max_wait_time}s dépassé")
                    return self._fallback_response(prompt)
                    
                except Exception as e:
                    logger.error(f"Erreur lors de la génération : {e}")
                    return self._fallback_response(prompt)
        
        # Méthode traditionnelle sans gestion dynamique
        else:
            try:
                response = make_request()
                return response.json().get("generated_text", "")
                
            except Exception as e:
                logger.error(f"Erreur de requête : {e}")
                return self._fallback_response(prompt)
            

    def generate_streaming_response(
        self, 
        prompt: str, 
        max_length: int = 500, 
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50
    ) -> StreamingResponse:
        """
        Génère une réponse en streaming compatible avec FastAPI.
        
        Args:
            prompt: Texte de prompt
            max_length: Longueur maximale de la réponse
            temperature: Contrôle de la créativité
            top_p: Échantillonnage noyau
            top_k: Nombre de tokens les plus probables
        
        Returns:
            Réponse streaming de FastAPI
        """
        def generate():
            try:
                # Utiliser generate_response avec streaming
                text_generator = self.generate_response(
                    prompt=prompt,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    stream=True
                )
                
                for chunk in text_generator:
                    yield f"data: {chunk}\n\n"
            except Exception as e:
                logger.error(f"Erreur de streaming : {e}")
                yield f"data: {str(e)}\n\n"
        
        return StreamingResponse(generate(), media_type="text/event-stream")


                

    def _fallback_response(self, original_prompt: str) -> str:
        """
        Mécanisme de réponse de secours en cas d'échec.
        """
        fallback_responses = [
            "Je suis désolé, mais je ne peux pas générer de réponse pour le moment. Veuillez réessayer ultérieurement.",
            "Le service de génération de réponse est temporairement indisponible.",
            f"Contexte initial : {original_prompt[:100]}... (traitement interrompu)"
        ]
        
        # Sélectionner une réponse de secours
        import random
        return random.choice(fallback_responses)

    def _log_request_details(
        self, 
        prompt: str, 
        start_time: float, 
        success: bool, 
        error: Optional[Exception] = None
    ):
        """
        Journalisation détaillée des requêtes.
        """
        duration = time.time() - start_time
        log_data = {
            "model": self.model_name,
            "prompt_length": len(prompt),
            "success": success,
            "duration": duration
        }
        
        if not success and error:
            log_data["error"] = str(error)
        
        logger.info(f"LLM Request Log: {log_data}")


    def generate_chat_response(
        self, 
        messages: List[Dict[str, str]], 
        max_length: int = 500, 
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        stream: bool = False
    ) -> Union[str, Iterator[str]]:
        """
        Génère une réponse dans un format de conversation.
        
        Args:
            messages: Liste de messages au format [{"role": "user/assistant", "content": "message"}]
            max_length: Longueur maximale de la réponse
            temperature: Contrôle de la créativité
            top_p: Échantillonnage noyau
            top_k: Nombre de tokens les plus probables
            stream: Activer/désactiver le streaming
        
        Returns:
            Réponse générée (str ou itérateur)
        """
        # Convertir le format de chat en prompt unique pour la v1
        # Dans une version future, on pourrait améliorer le support du format multi-tours
        full_prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        
        return self.generate_response(
            prompt=full_prompt,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stream=stream
        )

    def generate_with_context(
        self, 
        context: str, 
        question: str, 
        max_length: int = 500, 
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        stream: bool = False
    ) -> Union[str, Iterator[str]]:
        """
        Génère une réponse basée sur un contexte et une question.
        
        Args:
            context: Contexte fourni pour la génération
            question: Question spécifique à poser sur le contexte
            max_length: Longueur maximale de la réponse
            temperature: Contrôle de la créativité
            top_p: Échantillonnage noyau
            top_k: Nombre de tokens les plus probables
            stream: Activer/désactiver le streaming
        
        Returns:
            Réponse générée (str ou itérateur)
        """
        # Formater le prompt avec contexte et question
        full_prompt = f"Contexte:\n{context}\n\nQuestion: {question}\n\nRéponse:"
        
        return self.generate_response(
            prompt=full_prompt,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stream=stream
        )

# Test rapide
def test_llm_service():
    try:
        # Initialiser le service LLM
        llm_service = LlmService(
            model_name="llama3.2:latest",
            server_url="http://10.100.212.118:8001/generate"  # URL de votre serveur LLM
        )
        
        # Test avec une requête simple (non-streaming)
        prompt = "Explique les principes fondamentaux de la fiscalité camerounaise en 3 points."
        response = llm_service.generate_response(
            prompt, 
            max_length=300, 
            temperature=0.5,
            stream=False
        )
        
        print("=== Réponse du LLM (Non-Streaming) ===")
        print(response)
        
        # Test avec streaming
        print("\n=== Réponse du LLM (Streaming) ===")
        streaming_response = llm_service.generate_response(
            prompt, 
            max_length=300, 
            temperature=0.5,
            stream=True
        )
        
        # Collecter les chunks de la réponse streaming
        full_streaming_response = ""
        for chunk in streaming_response:
            print(chunk, end='', flush=True)
            full_streaming_response += chunk
        print()  # Nouvelle ligne après le streaming
        
        # Test avec contexte
        context = """
        La fiscalité camerounaise est un système complexe qui vise à générer des revenus pour l'État 
        tout en soutenant le développement économique. Le Code Général des Impôts définit les principales 
        obligations fiscales des entreprises et des individus.
        """
        question = "Quels sont les principaux types d'impôts pour les entreprises au Cameroun ?"
        
        context_response = llm_service.generate_with_context(
            context, 
            question, 
            max_length=300,
            stream=False
        )
        
        print("\n=== Réponse basée sur Contexte ===")
        print(context_response)
        
    except Exception as e:
        print(f"Erreur lors du test : {e}")

if __name__ == "__main__":
    test_llm_service()