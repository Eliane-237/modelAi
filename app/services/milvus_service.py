import logging
import json
from typing import List, Dict, Any, Optional
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections, utility

# Configuration du logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class MilvusService:
    def __init__(
        self, 
        collection_name: str, 
        dim: int = 1024,
        host: str = "10.100.212.133", 
        port: int = 19530,
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Initialise le service Milvus avec des options de connexion avancées.
        
        Args:
            collection_name: Nom de la collection Milvus
            dim: Dimension des vecteurs d'embedding
            host: Adresse du serveur Milvus
            port: Port du serveur Milvus
            timeout: Délai d'attente pour la connexion
            max_retries: Nombre maximum de tentatives de connexion
        """
        self.collection_name = collection_name
        self.dim = dim
        self.host = host
        self.port = port
        self.timeout = timeout
        self.max_retries = max_retries
        
        self.collection = None
        
        # Méthode de connexion améliorée
        self._connect_to_milvus()
        
        # Vérifier/créer la collection
        self._ensure_collection()

    def _connect_to_milvus(self):
        """
        Établit la connexion avec le serveur Milvus distant avec des tentatives multiples.
        """
        import time
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Tentative de connexion à Milvus ({attempt + 1}/{self.max_retries})...")
                
                connections.connect(
                    alias="default", 
                    host=self.host, 
                    port=str(self.port),
                    timeout=self.timeout
                )
                
                # Vérifier la connexion
                utility.list_collections()
                
                logger.info(f"✅ Connecté au serveur Milvus sur {self.host}:{self.port}")
                return
            
            except Exception as e:
                logger.warning(f"⚠️ Échec de connexion (Tentative {attempt + 1}/{self.max_retries}): {e}")
                
                # Backoff exponentiel
                time.sleep(2 ** attempt)
        
        # Si toutes les tentatives échouent
        error_msg = f"❌ Impossible de se connecter au serveur Milvus après {self.max_retries} tentatives"
        logger.error(error_msg)
        raise ConnectionError(error_msg)

    def _ensure_collection(self):
        """
        S'assure que la collection existe avec le bon schéma.
        Crée la collection si nécessaire.
        """
        try:
            # Vérifier si la collection existe
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
                logger.info(f"Collection existante '{self.collection_name}' chargée")
                
                # Vérifier si le schéma est compatible
                if not self._is_schema_compatible(self.collection.schema):
                    logger.warning(f"Schéma incompatible, recréation de la collection '{self.collection_name}'")
                    utility.drop_collection(self.collection_name)
                    self._define_and_create_collection()
            else:
                logger.info(f"Collection '{self.collection_name}' non trouvée, création en cours...")
                self._define_and_create_collection()
        
        except Exception as e:
            logger.error(f"Erreur lors de la vérification/création de la collection: {e}")
            raise

    def _define_and_create_collection(self):
        """
        Définit et crée la collection avec le schéma approprié.
        """
        try:
            # Définir les champs
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="page_number", dtype=DataType.INT64),
                FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535)
            ]
            
            # Créer le schéma
            schema = CollectionSchema(fields)
            
            # Créer la collection
            self.collection = Collection(name=self.collection_name, schema=schema)
            logger.info(f"Collection '{self.collection_name}' créée avec succès")
            
            # Créer l'index
            index_params = {
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {
                    "M": 16,
                    "efConstruction": 200
                }
            }
            
            self.collection.create_index(field_name="embedding", index_params=index_params)
            logger.info("Index créé sur le champ 'embedding'")
            
        except Exception as e:
            logger.error(f"Erreur lors de la création de la collection: {e}")
            raise

    def _is_schema_compatible(self, existing_schema):

        """
        Vérifie si le schéma existant est compatible avec nos besoins.
        """
        try:
            # Vérifier la dimension du vecteur d'embedding
            for field in existing_schema.fields:
                if field.name == "embedding" and field.dtype == DataType.FLOAT_VECTOR:
                    field_params = getattr(field, "params", {})
                    if isinstance(field_params, dict) and field_params.get("dim") == self.dim:
                        return True
            return False
        except Exception as e:
            logger.error(f"Erreur lors de la vérification du schéma: {e}")
            return False

    def insert_documents(self, embeddings, texts=None, metadata_list=None):
        """
        Insère des embeddings dans la collection Milvus.
        
        Args:
            embeddings: Liste des vecteurs d'embedding
            texts: Liste des textes correspondants (optionnel)
            metadata_list: Liste des métadonnées (optionnel)
        """
        if not embeddings:
            logger.warning("Aucun embedding à insérer")
            return
            
        try:
            # Vérifier la dimension des embeddings
            if len(embeddings[0]) != self.dim:
                logger.warning(f"Dimension des embeddings ({len(embeddings[0])}) différente de celle attendue ({self.dim})")
                logger.info("Adaptation de la dimension des embeddings...")
                embeddings = self._adjust_embedding_dimension(embeddings)
            
            # Préparer les données à insérer
            data = [embeddings]
            
            # Ajouter les textes si fournis
            if texts:
                if len(texts) != len(embeddings):
                    raise ValueError("Le nombre de textes doit correspondre au nombre d'embeddings")
                data.append(texts)
            else:
                data.append([""] * len(embeddings))
                
            # Ajouter les métadonnées si fournies
            if metadata_list:
                if len(metadata_list) != len(embeddings):
                    raise ValueError("Le nombre de métadonnées doit correspondre au nombre d'embeddings")
                
                # Extraire les champs de métadonnées
                document_ids = []
                chunk_ids = []
                page_numbers = []
                metadata_json = []
                
                for meta in metadata_list:
                    document_ids.append(meta.get("document_id", ""))
                    chunk_ids.append(meta.get("chunk_id", ""))
                    page_numbers.append(meta.get("page_number", 0))
                    
                    # Convertir les métadonnées complètes en JSON
                    metadata_json.append(json.dumps(meta))
                
                data.extend([document_ids, chunk_ids, page_numbers, metadata_json])
            else:
                # Valeurs par défaut
                data.extend([[""] * len(embeddings), [""] * len(embeddings), [0] * len(embeddings), ["{}"] * len(embeddings)])
            
            # Insérer les données
            insert_result = self.collection.insert(data)
            logger.info(f"Insertion réussie: {insert_result.insert_count} enregistrements")
            
            # Flush pour s'assurer que les données sont persistantes
            self.collection.flush()
            logger.info("Données persistées avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'insertion des documents: {e}")
            raise

    def _adjust_embedding_dimension(self, embeddings):
        """
        Ajuste la dimension des embeddings pour correspondre à celle attendue par Milvus.
        
        Args:
            embeddings: Liste des vecteurs d'embedding
            
        Returns:
            Liste des vecteurs d'embedding ajustés
        """
        adjusted_embeddings = []
        for embedding in embeddings:
            current_dim = len(embedding)
            
            if current_dim < self.dim:
                # Padding pour augmenter la dimension
                adjusted = embedding + [0.0] * (self.dim - current_dim)
            elif current_dim > self.dim:
                # Troncature pour réduire la dimension
                adjusted = embedding[:self.dim]
            else:
                adjusted = embedding
                
            adjusted_embeddings.append(adjusted)
            
        return adjusted_embeddings
    

    def search(self, query_embedding, top_k=5):

        """
        Recherche les embeddings les plus similaires dans Milvus.
        Compatible avec les collections existantes qui peuvent avoir des schémas différents.
        
        Args:
            query_embedding: Vecteur d'embedding de requête
            top_k: Nombre de résultats à retourner
        
        Returns:
            Liste de dictionnaires contenant le texte et le score de similarité
        """
        if not self.collection:
            raise Exception("Collection not initialized.")
        
        # Vérification de la dimension
        if len(query_embedding) != self.dim:
            raise ValueError(f"Query embedding dimension should be {self.dim}")
        
        try:
            # Chargement de la collection pour s'assurer que l'index est disponible
            self.collection.load()
            
            # Récupérer les champs disponibles dans la collection
            schema = self.collection.schema
            available_fields = [field.name for field in schema.fields]
            logger.info(f"Champs disponibles dans la collection: {available_fields}")
            
            # Filtrer les champs de sortie en fonction des champs disponibles
            base_output_fields = ["text"]
            additional_fields = ["document_id", "page_number", "filename", "chunk_id"]
            
            # Ne conserver que les champs qui existent réellement dans la collection
            output_fields = [field for field in base_output_fields if field in available_fields]
            output_fields += [field for field in additional_fields if field in available_fields]
            
            logger.info(f"Utilisation des champs de sortie: {output_fields}")
            
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }
            
            # Effectuer la recherche avec les champs disponibles
            results = self.collection.search(
                data=[query_embedding], 
                anns_field="embedding", 
                param=search_params, 
                limit=top_k,
                output_fields=output_fields if output_fields else None
            )
            
            # Formater les résultats pour les rendre plus faciles à utiliser
            formatted_results = []
            
            for hit in results[0]:
                try:
                    result = {"score": hit.distance}
                    
                    # Récupérer les attributs disponibles de manière sécurisée
                    entity = hit.entity
                    for field in output_fields:
                        try:
                            # Utiliser getattr avec une valeur par défaut
                            value = getattr(entity, field, None)
                            result[field] = value
                        except Exception as field_error:
                            logger.warning(f"Impossible d'accéder au champ {field}: {field_error}")
                    
                    # Assurer qu'il y a au moins un champ 'text'
                    if "text" not in result or not result["text"]:
                        result["text"] = "Texte non disponible"
                    
                    formatted_results.append(result)
                    
                except Exception as hit_error:
                    logger.error(f"Erreur lors de l'extraction d'un résultat: {hit_error}")
                    # Ajouter un résultat minimal en cas d'erreur
                    formatted_results.append({
                        "text": "Erreur de récupération du texte",
                        "score": hit.distance
                    })
            
            logger.info(f"Found {len(formatted_results)} results for the query.")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Erreur lors de la recherche: {e}")
            raise
        finally:
            # Libérer la mémoire
            try:
                self.collection.release()
            except Exception as release_error:
                logger.warning(f"Impossible de libérer la collection: {release_error}")

    def insert_documents_with_metadata(self, embeddings, texts, metadata_list):
            """
            Adapte l'insertion avec métadonnées vers la méthode existante.
            """
            logger.info(f"Insertion dans Milvus de {len(embeddings)} embeddings avec textes et métadonnées")
            return self.insert_documents(embeddings, texts, metadata_list)

    def get_collection_stats(self):
        """Retourne des statistiques sur la collection."""
        try:
            if not self.collection:
                return {"error": "Collection non initialisée"}
                
            stats = {
                "name": self.collection_name,
                "dimension": self.dim,
                "entity_count": self.collection.num_entities
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des statistiques: {e}")
            return {"error": str(e)}
        
    def get_documents_by_filter(self, filter_expr: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Récupère les documents correspondant à une expression de filtre.
        
        Args:
            filter_expr: Expression de filtre Milvus
            limit: Nombre maximum de documents à retourner
            
        Returns:
            Liste de documents (dictionnaires)
        """
        try:
            # Charger la collection
            self.collection.load()
            
            # Exécuter la requête avec filtrage
            results = self.collection.query(
                expr=filter_expr,
                output_fields=["text", "metadata"],
                limit=limit
            )
            
            results = [result.entity for result in results]
            
            # Convertir les résultats en dictionnaires
            documents = []
            for result in results:
                document = {
                    "text": result["text"],
                    "metadata": json.loads(result["metadata"])  
                }
                documents.append(document)
            
            return documents

        except Exception as e:
            logger.error(f"Erreur lors de la récupération des documents par filtre: {e}")
            return []
        finally:
            # Libérer la collection
            self.collection.release()


# Point d'entrée pour les tests
if __name__ == "__main__":
    try:
        # Test de connexion
        service = MilvusService(
            collection_name="test_collection", 
            dim=1024,
            host="10.100.212.133",
            port=19530
        )
        
        # Vérifier les statistiques de la collection
        stats = service.get_collection_stats()
        print(f"Statistiques de la collection: {json.dumps(stats, indent=2)}")
        
    except Exception as e:
        print(f"Erreur lors du test de connexion Milvus : {e}")