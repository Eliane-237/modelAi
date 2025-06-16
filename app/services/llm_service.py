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
        stream: bool = False,
        **kwargs
    ) -> Union[str, Iterator[str]]:
        """
        Génère une réponse avec support streaming véritable.
        
        Args:
            prompt: Texte d'entrée
            stream: Si True, retourne un Iterator[str] pour le streaming
            
        Returns:
            str si stream=False, Iterator[str] si stream=True
        """
        
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
        
        logger.info(f"Génération de la réponse avec le LLM (streaming: {stream})")
        
        try:
            if stream:
                return self._generate_streaming(payload)
            else:
                return self._generate_standard(payload)
        except Exception as e:
            logger.error(f"Erreur LLM: {e}")
            if stream:
                return self._fallback_streaming(prompt)
            else:
                return self._fallback_response(prompt)
    
    def _generate_streaming(self, payload: Dict[str, Any]) -> Iterator[str]:
        """
        Génère une réponse en streaming - Version corrigée.
        """
        try:
            response = requests.post(
                self.server_url,
                json=payload,
                stream=True,
                timeout=self.timeout,
                headers={'Accept': 'text/event-stream'}
            )
            response.raise_for_status()
            
            logger.info("🔄 Streaming démarré")
            
            # Traiter la réponse streaming
            for line in response.iter_lines(decode_unicode=True):
                if line.strip():
                    # Gestion des différents formats de réponse streaming
                    if line.startswith('data: '):
                        data = line[6:]  # Enlever 'data: '
                        if data.strip() and data != '[DONE]':
                            try:
                                # Essayer de parser comme JSON
                                json_data = json.loads(data)
                                if 'token' in json_data:
                                    yield json_data['token']
                                elif 'text' in json_data:
                                    yield json_data['text']
                                elif 'generated_text' in json_data:
                                    yield json_data['generated_text']
                                else:
                                    yield data
                            except json.JSONDecodeError:
                                # Si ce n'est pas du JSON, c'est probablement du texte direct
                                yield data
                    elif line.startswith('{'):
                        # Ligne JSON directe
                        try:
                            json_data = json.loads(line)
                            if 'token' in json_data:
                                yield json_data['token']
                            elif 'text' in json_data:
                                yield json_data['text']
                        except json.JSONDecodeError:
                            pass
                    else:
                        # Texte brut
                        yield line
            
            logger.info("✅ Streaming terminé")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur de requête streaming: {e}")
            # Fallback vers simulation de streaming
            yield from self._simulate_streaming(payload.get('prompt', ''))
        except Exception as e:
            logger.error(f"Erreur streaming: {e}")
            yield from self._simulate_streaming(payload.get('prompt', ''))
    
    def _generate_standard(self, payload: Dict[str, Any]) -> str:
        """Génère une réponse standard (non-streaming)."""
        try:
            response = requests.post(
                self.server_url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("generated_text", "")
            
        except Exception as e:
            logger.error(f"Erreur génération standard: {e}")
            return self._fallback_response(payload.get('prompt', ''))
    
    def _generate_standard(self, payload: Dict[str, Any]) -> str:
        """Génère une réponse standard (non-streaming)."""
        response = requests.post(self.server_url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        return result.get("generated_text", "")          

    def _fallback_response(self, original_prompt: str) -> str:
        """
        Mécanisme de réponse de secours en cas d'échec.
        """
        fallback_responses = [
            "Je suis désolé, mais je ne peux pas générer de réponse pour le moment. Veuillez réessayer ultérieurement.",
            "Le service de génération de réponse est temporairement indisponible. Veuillez patienter quelques instants.",
            "Une erreur technique est survenue. L'équipe technique a été notifiée."
        ]
        
        # Retourner une réponse de secours appropriée
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


    # Test du streaming
def test_streaming():
    """Test rapide du streaming"""
    llm_service = LlmService()
    
    prompt = "Expliquez brièvement le droit de la famille au Cameroun."
    
    print("=== Test Streaming ===")
    print("Prompt:", prompt)
    print("Réponse streaming:")
    
    # Test streaming
    response_stream = llm_service.generate_response(
        prompt, 
        max_length=200, 
        temperature=0.7, 
        stream=True
    )
    
    full_response = ""
    for token in response_stream:
        print(token, end='', flush=True)
        full_response += token
    
    print(f"\n\nRéponse complète: {len(full_response)} caractères")

if __name__ == "__main__":
    test_streaming()