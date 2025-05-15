import requests
import logging
import time
import hashlib
from typing import List, Optional, Dict, Any
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from fastapi import FastAPI


# Configuration du logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

app = FastAPI()

class EmbeddingService:
    def __init__(
            
            self, 
            server_url="http://10.100.212.118:8000", 
            max_retries=3, 
            timeout=60
    ):
        """
        Initialise la connexion avec l'API d'embeddings sur le serveur distant.
        
        Args:
            server_url: URL du serveur d'embedding
            max_retries: Nombre maximal de tentatives de connexion
            timeout: Délai d'attente en secondes avant échec de la connexion
        """
        self.server_url = server_url
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = self._create_retry_session()

       
        # Tester la connexion à l'initialisation
        self._test_connection()
    
    def _create_retry_session(self):
        """
        Crée une session HTTP avec gestion automatique des nouvelles tentatives.
        """
        session = requests.Session()
        retry = Retry(
            total=self.max_retries,
            read=self.max_retries,
            connect=self.max_retries,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["POST", "GET"]
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session
    
    def _test_connection(self):
        """
        Teste la connexion au serveur d'embedding.
        """
        try:
            # Envoyer une requête de test minimale
            test_text = ["Test de connexion"]
            logger.info(f"🔄 Test de connexion au serveur d'embedding à {self.server_url}")
            
            response = self.session.post(
                f"{self.server_url}/generate_embeddings/",
                json={"texts": test_text},
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                logger.info(f"✅ Connexion au serveur d'embedding réussie")
            else:
                logger.warning(f"⚠️ Connexion au serveur d'embedding établie mais avec code d'état {response.status_code}")
                
        except requests.exceptions.ConnectTimeout:
            logger.warning(f"⚠️ Délai de connexion expiré lors du test de connexion à {self.server_url}")
        except requests.exceptions.ConnectionError:
            logger.warning(f"⚠️ Impossible d'établir une connexion avec {self.server_url}")
        except Exception as e:
            logger.warning(f"⚠️ Erreur lors du test de connexion: {e}")

       
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Envoie les textes à l'API distante pour obtenir les embeddings.
        
        Args:
            texts: Liste de textes à convertir en embeddings
            
        Returns:
            Liste de vecteurs d'embedding
            
        Raises:
            ValueError: Si la liste de textes est vide
            RuntimeError: Si la connexion ou la réponse API échoue
        """
        if not texts:
            raise ValueError("⚠️ La liste de textes ne peut pas être vide.")

        start_time = time.time()
        
        try:
            logger.info(f"📡 Envoi de {len(texts)} textes pour embeddings à {self.server_url}")
            
            response = self.session.post(
                f"{self.server_url}/generate_embeddings/",
                json={"texts": texts},
                timeout=self.timeout
            )
            
            # Vérifier la réponse HTTP
            if response.status_code != 200:
                logger.error(f"🚨 Erreur API {response.status_code}: {response.text}")
                raise RuntimeError(f"L'API a retourné une erreur {response.status_code}: {response.text}")
            
            # Extraire les embeddings de la réponse
            result = response.json()
            embeddings = result.get("embeddings", [])
            
            if not embeddings:
                logger.error("🚨 Réponse vide de l'API d'embeddings.")
                raise RuntimeError("🚨 Réponse vide de l'API.")
            
            # Vérification de la cohérence des dimensions des embeddings
            if any(len(embed) != len(embeddings[0]) for embed in embeddings):
                logger.error("🚨 Incohérence des dimensions des embeddings reçus.")
                raise ValueError("Tous les embeddings doivent avoir la même dimension.")
            
            # Calculer le temps de traitement
            processing_time = time.time() - start_time
            
            logger.info(f"✅ {len(embeddings)} embeddings reçus avec succès en {processing_time:.2f}s.")
            logger.info(f"📊 Dimension des embeddings: {len(embeddings[0])}")
            
            return embeddings
            
        except requests.exceptions.ConnectTimeout:
            logger.error(f"🕒 Délai de connexion expiré pour {self.server_url}")
            raise RuntimeError(f"Impossible de se connecter au serveur d'embeddings (timeout): {self.server_url}")
        
        except requests.exceptions.ConnectionError:
            logger.error(f"🔌 Erreur de connexion pour {self.server_url}")
            raise RuntimeError(f"Le serveur d'embeddings est inaccessible: {self.server_url}")
        
        except requests.exceptions.RequestException as e:
            logger.error(f"🚨 Erreur de connexion à l'API d'embeddings : {e}")
            raise RuntimeError(f"Impossible de contacter l'API: {e}")
        
        except Exception as e:
            logger.error(f"🚨 Erreur inattendue: {e}")
            raise RuntimeError(f"Erreur lors de la génération des embeddings: {e}")

    def generate_embeddings_batch(self, texts: List[str], batch_size=32) -> List[List[float]]:
        """Génère des embeddings avec une approche par lots pour de meilleures performances."""
        if not texts:
            return []
        
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            try:
                # Utiliser des requêtes partielles pour éviter les timeout
                batch_embeddings = self._generate_embeddings_request(batch)
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Erreur lors de la génération d'embeddings pour le lot {i//batch_size}: {e}")
                # Si l'erreur est due à la taille, réessayer avec des lots plus petits
                if batch_size > 4 and "size" in str(e).lower():
                    smaller_batch_size = batch_size // 2
                    sub_embeddings = self.generate_embeddings_batch(batch, smaller_batch_size)
                    all_embeddings.extend(sub_embeddings)
                else:
                    # Fallback: générer un par un en cas d'erreur
                    for text in batch:
                        try:
                            emb = self._generate_embeddings_request([text])
                            all_embeddings.extend(emb)
                        except:
                            # Ajouter un embedding zéro en cas d'échec total
                            all_embeddings.append([0.0] * self.embedding_dim)
        
        return all_embeddings

    
        
       
# Test rapide
if __name__ == "__main__":
    embedding_service = EmbeddingService()

    example_chunks = [
        "L'intelligence artificielle transforme le monde.",
        "Les modèles de langage comme GPT sont puissants."
    ]

    try:
        embeddings = embedding_service.generate_embeddings(example_chunks)
        print(f"✅ Nombre d'embeddings générés: {len(embeddings)}")
        print(f"🔍 Premier embedding (dimensions: {len(embeddings[0])}):\n{embeddings[0][:5]}... (tronqué)")
    except ValueError as e:
        print(f"❌ Erreur de validation: {e}")
    except RuntimeError as e:
        print(f"❌ Erreur de connexion: {e}")
    except Exception as e:
        print(f"❌ Erreur inattendue: {e}")