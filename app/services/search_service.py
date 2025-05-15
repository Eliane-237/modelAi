import logging
import json
from typing import List, Tuple, Dict, Any
from pymilvus import Collection
from app.services.embedding_service import EmbeddingService
from app.services.milvus_service import MilvusService
import time

# Configuration du logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class SearchService:
    def __init__(self, milvus_service: MilvusService, embedding_service: EmbeddingService, top_k: int = 5):
        self.milvus_service = milvus_service
        self.embedding_service = embedding_service
        self.top_k = top_k

    def search(self, query: str, top_k: int = None, filter_expr: str = None) -> List[Dict[str, Any]]:
        """
        Recherche des documents pertinents pour la requête.
        
        Args:
            query: La requête de l'utilisateur
            top_k: Nombre maximal de résultats à retourner (utilise la valeur par défaut de la classe si None)
            filter_expr: Expression de filtrage Milvus optionnelle
            
        Returns:
            Liste de dictionnaires contenant les documents trouvés avec leurs métadonnées
        """
        try:
            start_time = time.time()
            logger.info(f"🔎 Recherche des documents pour la requête : {query}")
            
            # Utiliser top_k de la requête ou la valeur par défaut
            actual_top_k = top_k if top_k is not None else self.top_k
            
            # Génération de l'embedding
            embedding_start = time.time()
            query_embeddings = self.embedding_service.generate_embeddings([query])
            if not query_embeddings or len(query_embeddings) == 0:
                logger.error("❌ Aucun embedding généré pour la requête.")
                return []
                
            query_embedding = query_embeddings[0]
            embedding_time = time.time() - embedding_start
            logger.info(f"⏱️ Temps de génération de l'embedding : {embedding_time:.3f}s")
            
            # Recherche dans Milvus
            search_start = time.time()
            try:
                # Charger la collection
                self.milvus_service.collection.load()
                
                # Déterminer les champs disponibles dans le schéma
                schema = self.milvus_service.collection.schema
                field_names = [field.name for field in schema.fields]
                
                # On cherche les champs qui pourraient contenir les données JSON
                possible_json_fields = [field for field in field_names if field in 
                                    ["text", "metadata_json", "json_data", "data", "content"]]
                
                # Configuration des paramètres de recherche
                search_params = {
                    "metric_type": "COSINE",
                    "params": {"nprobe": 10}
                }
                
                # Ajouter l'expression de filtrage si fournie
                search_kwargs = {
                    "data": [query_embedding],
                    "anns_field": "embedding",
                    "param": search_params,
                    "limit": actual_top_k,  # Utiliser la valeur déterminée plus haut
                    "output_fields": field_names if field_names else None
                }
                
                # Ajouter le filtre si présent
                if filter_expr:
                    search_kwargs["expr"] = filter_expr
                    logger.info(f"Recherche avec filtre: {filter_expr}")
                
                # Effectuer la recherche avec tous les paramètres
                search_results = self.milvus_service.collection.search(**search_kwargs)
                
                # Traitement des résultats
                results = []
                for hit in search_results[0]:
                    try:
                        # Structure de base du résultat
                        result = {
                            "score": hit.distance,
                            "id": getattr(hit, "id", None)
                        }
                        
                        # Tenter d'extraire le texte et les métadonnées
                        text_found = False
                        metadata_found = False
                        
                        # Essayer d'accéder à l'entity
                        entity = hit.entity
                        
                        # Parcourir tous les champs disponibles
                        for field_name in dir(entity):
                            # Ignorer les attributs internes et les méthodes
                            if field_name.startswith('_') or callable(getattr(entity, field_name)):
                                continue
                                
                            field_value = getattr(entity, field_name)
                            
                            # Traiter le champ metadata spécifiquement
                            if field_name == "metadata":
                                if isinstance(field_value, str):
                                    try:
                                        result["metadata"] = json.loads(field_value)
                                    except json.JSONDecodeError:
                                        result["metadata"] = {"document_id": "", "chunk_id": "", "filename": "Document inconnu", "page_number": 0}
                                elif isinstance(field_value, dict):
                                    result["metadata"] = field_value
                                else:
                                    result["metadata"] = {"document_id": "", "chunk_id": "", "filename": "Document inconnu", "page_number": 0}
                                metadata_found = True
                                continue
                            
                            # Essayer de traiter comme JSON si c'est une chaîne
                            if isinstance(field_value, str):
                                try:
                                    json_obj = json.loads(field_value)
                                    
                                    # Si c'est un JSON qui contient "text" et "metadata"
                                    if "text" in json_obj and "metadata" in json_obj:
                                        result["text"] = json_obj["text"]
                                        result["metadata"] = json_obj["metadata"]
                                        text_found = True
                                        metadata_found = True
                                    # Si c'est juste le texte
                                    elif isinstance(json_obj, str):
                                        result["text"] = json_obj
                                        text_found = True
                                    # Si c'est un autre type de JSON
                                    else:
                                        result[field_name] = json_obj
                                        
                                        # Chercher le texte et les métadonnées dans l'objet JSON
                                        if "text" in json_obj and not text_found:
                                            result["text"] = json_obj["text"]
                                            text_found = True
                                        if "metadata" in json_obj and not metadata_found:
                                            result["metadata"] = json_obj["metadata"]
                                            metadata_found = True
                                except:
                                    # Si ce n'est pas du JSON, c'est peut-être directement le texte
                                    if field_name == "text" and not text_found:
                                        result["text"] = field_value
                                        text_found = True
                                    else:
                                        result[field_name] = field_value
                            else:
                                # Pour les valeurs non-string
                                result[field_name] = field_value
                        
                        # Si aucun texte n'a été trouvé, rechercher dans un autre format
                        if not text_found:
                            # Essayer de voir si le texte est dans un champ direct
                            for text_field in ["text", "content", "data"]:
                                if hasattr(entity, text_field):
                                    result["text"] = getattr(entity, text_field)
                                    text_found = True
                                    break
                        
                        # Si toujours pas de texte, utiliser un placeholder
                        if not text_found:
                            result["text"] = f"Document #{result.get('id', 'inconnu')}"
                        
                        # S'assurer que metadata existe toujours
                        if not metadata_found:
                            result["metadata"] = {"document_id": "", "chunk_id": "", "filename": "Document inconnu", "page_number": 0}
                        
                        results.append(result)
                        
                    except Exception as e:
                        logger.error(f"Erreur lors de l'extraction d'un résultat: {str(e)}")
                        results.append({
                            "text": "Erreur d'extraction",
                            "score": hit.distance,
                            "error": str(e),
                            "metadata": {"document_id": "", "chunk_id": "", "filename": "Document inconnu", "page_number": 0}
                        })
                
                search_time = time.time() - search_start
                logger.info(f"⏱️ Temps de recherche Milvus : {search_time:.10f}s")
                logger.info(f"✅ {len(results)} documents récupérés pour la requête '{query}'")
                
                return results
                
            except Exception as e:
                logger.error(f"Erreur lors de la recherche: {e}")
                return []
                
        except Exception as e:
            logger.error(f"❌ Erreur lors de la recherche : {e}")
            return []
        finally:
            try:
                # Libérer la collection
                self.milvus_service.collection.release()
            except:
                pass
            
            total_time = time.time() - start_time
            logger.info(f"⏱️ Temps total de recherche: {total_time:.3f}s")

    def expand_query(self, query: str) -> List[str]:
        """
        Génère des variantes de la requête pour améliorer la recherche juridique.
        
        Args:
            query: Requête originale de l'utilisateur
            
        Returns:
            Liste des requêtes étendues, incluant l'originale
        """
        expanded_queries = [query]  # On garde toujours la requête originale
        
        # Termes juridiques camerounais spécifiques et leurs synonymes
        legal_synonyms = {
            "impôt": ["taxation", "prélèvement fiscal", "imposition", "taxe", "IRPP", "IR"],
            "déclaration": ["déclaration fiscale", "formulaire fiscal", "déclaration d'impôt", "déclaration de revenus"],
            "revenus": ["gains", "recettes", "bénéfices", "rémunérations", "salaires"],
            "loi": ["législation", "texte juridique", "réglementation", "circulaire", "disposition légale"],
            "taxe": ["impôt", "redevance", "contribution", "prélèvement", "droit fiscal"],
            "entreprise": ["société", "établissement", "personne morale", "entité commerciale"],
            "personne": ["individu", "contribuable", "personne physique", "assujetti"],
            "paiement": ["règlement", "acquittement", "versement", "paiement d'impôt"],
            "exemption": ["exonération", "dégrèvement", "allègement fiscal", "dispense"],
            "obligation": ["devoir fiscal", "exigence légale", "obligation fiscale", "obligation déclarative"]
        }
        
        # Codes spécifiques camerounais et leur signification
        legal_codes = {
            "IRPP": ["Impôt sur le Revenu des Personnes Physiques"],
            "IS": ["Impôt sur les Sociétés"],
            "TVA": ["Taxe sur la Valeur Ajoutée"],
            "DGI": ["Direction Générale des Impôts"],
            "CGI": ["Code Général des Impôts"],
            "TPF": ["Taxe Professionnelle Foncière"],
            "TPR": ["Taxe Professionnelle de Redevance"],
            "TDL": ["Taxe de Développement Local"]
        }
        
        try:
            import re
            
            # 1. Expansion basée sur les synonymes juridiques
            query_words = re.findall(r'\w+', query.lower())
            for word in query_words:
                for key, synonyms in legal_synonyms.items():
                    if word == key or word in [s.lower() for s in synonyms]:
                        # Ajouter des variantes avec synonymes
                        for synonym in synonyms:
                            new_query = re.sub(r'\b' + re.escape(word) + r'\b', synonym, query, flags=re.IGNORECASE)
                            if new_query != query and new_query not in expanded_queries:
                                expanded_queries.append(new_query)
            
            # 2. Expansion basée sur les codes juridiques camerounais
            for code, meanings in legal_codes.items():
                if code in query.upper():
                    # Remplacer le code par sa signification
                    for meaning in meanings:
                        new_query = re.sub(r'\b' + re.escape(code) + r'\b', meaning, query, flags=re.IGNORECASE)
                        if new_query not in expanded_queries:
                            expanded_queries.append(new_query)
                # Chercher si une signification est dans la requête
                for meaning in meanings:
                    if meaning.lower() in query.lower():
                        # Remplacer la signification par le code
                        new_query = re.sub(r'\b' + re.escape(meaning) + r'\b', code, query, flags=re.IGNORECASE)
                        if new_query not in expanded_queries:
                            expanded_queries.append(new_query)
            
            # 3. Expansion spécifique pour les documents juridiques camerounais
            if any(term in query.lower() for term in ["déclaration", "impôt", "revenu", "irpp"]):
                expanded_queries.append("déclaration IRPP cameroun")
                expanded_queries.append("formulaire déclaration impôt")
                expanded_queries.append("déclaration fiscale revenu")
            
            if "taxe" in query.lower() or "impôt" in query.lower():
                expanded_queries.append(query + " cameroun")
                expanded_queries.append(query + " code général des impôts")
                expanded_queries.append(query + " loi fiscale")
            
            # 4. Rechercher des sections spécifiques si la requête semble cibler un article
            article_match = re.search(r'\barticle\s+(\d+)\b', query.lower())
            if article_match:
                article_num = article_match.group(1)
                expanded_queries.append(f"article {article_num}")
                expanded_queries.append(f"Art. {article_num}")
                expanded_queries.append(f"ARTICLE {article_num}")
            
            logger.info(f"Expansion de requête: {query} -> {expanded_queries}")
            return expanded_queries
            
        except Exception as e:
            logger.error(f"Erreur lors de l'expansion de requête: {e}")
            return [query]  # Retourner la requête originale en cas d'erreur
        
    def search_with_expansion(self, query: str, top_k: int = None, filter_expr: str = None) -> List[Dict[str, Any]]:
        """
        Recherche des documents pertinents avec expansion de requête adaptée aux documents juridiques.
        
        Args:
            query: La requête de l'utilisateur
            
        Returns:
            Liste de dictionnaires contenant les documents trouvés avec leurs métadonnées
        """
        try:
                # Générer des variantes de la requête
            expanded_queries = self.expand_query(query)
            
            # Résultats combinés de toutes les recherches
            all_results = []
            seen_chunks = set()  # Éviter les doublons par chunk_id
            seen_docs = set()    # Suivre les documents uniques
            doc_count = {}       # Compter les apparitions par document_id
            
            # D'abord, chercher avec la requête originale
            original_results = self.search(query=query,
                top_k=top_k,
                filter_expr=filter_expr
            )
            if original_results:
                for result in original_results:
                    metadata = result.get("metadata", {})
                    chunk_id = metadata.get("chunk_id", "")
                    doc_id = metadata.get("document_id", "")
                    
                    if doc_id:
                        doc_count[doc_id] = doc_count.get(doc_id, 0) + 1
                        seen_docs.add(doc_id)
                    
                    if chunk_id and chunk_id not in seen_chunks:
                        seen_chunks.add(chunk_id)
                        result["matched_query"] = query  # Requête originale
                        all_results.append(result)
            
            # Ensuite, essayer les requêtes d'expansion (sauf la requête originale)
            for expanded_query in [q for q in expanded_queries if q != query]:
                try:
                    results = self.search(expanded_query)
                    
                    # Ajouter les nouveaux résultats
                    for result in results:
                        metadata = result.get("metadata", {})
                        chunk_id = metadata.get("chunk_id", "")
                        doc_id = metadata.get("document_id", "")
                        
                        if doc_id:
                            doc_count[doc_id] = doc_count.get(doc_id, 0) + 1
                            seen_docs.add(doc_id)
                        
                        if chunk_id and chunk_id not in seen_chunks:
                            seen_chunks.add(chunk_id)
                            result["matched_query"] = expanded_query
                            all_results.append(result)
                except Exception as e:
                    # Ne pas échouer si une requête d'expansion échoue
                    logger.warning(f"Échec de l'expansion '{expanded_query}': {e}")
                    continue
            
            # Si aucun résultat trouvé, retourner au moins les résultats originaux
            if not all_results and original_results:
                return original_results
                
                # Limiter au nombre de résultats demandés
            return all_results
            
        except Exception as e:
            logger.error(f"Erreur lors de la recherche avec expansion: {e}")
            # Fallback à la recherche standard
            return self.search(query)

    def _group_results_by_document(self, results: List[Dict[str, Any]], top_docs: List[str]) -> List[Dict[str, Any]]:
        """
        Réorganise les résultats pour favoriser la cohérence documentaire.
        Maintient la pertinence globale tout en groupant les chunks du même document.
        
        Args:
            results: Liste des résultats à réorganiser
            top_docs: Liste des IDs des documents les plus pertinents
            
        Returns:
            Liste réorganisée des résultats
        """
        if not results:
            return []
        
        # Organiser les résultats par document
        doc_results = {}
        for result in results:
            doc_id = result.get("metadata", {}).get("document_id", "unknown")
            if doc_id not in doc_results:
                doc_results[doc_id] = []
            doc_results[doc_id].append(result)
        
        # Trier les résultats de chaque document par score
        for doc_id in doc_results:
            doc_results[doc_id].sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # Réorganiser les résultats en favorisant les top documents
        # mais en maintenant une diversité globale
        grouped = []
        
        # D'abord, ajouter les meilleurs résultats de chaque document important
        for doc_id in top_docs:
            if doc_id in doc_results and doc_results[doc_id]:
                grouped.append(doc_results[doc_id][0])  # Meilleur résultat
                doc_results[doc_id] = doc_results[doc_id][1:]  # Retirer le meilleur
        
        # Ensuite, ajouter les autres résultats en alternant les documents
        # pour maintenir une diversité documentaire
        remaining_docs = list(doc_results.keys())
        while remaining_docs:
            for doc_id in remaining_docs[:]:
                if doc_results[doc_id]:
                    grouped.append(doc_results[doc_id][0])
                    doc_results[doc_id] = doc_results[doc_id][1:]
                    
                    if not doc_results[doc_id]:
                        remaining_docs.remove(doc_id)
        
        return grouped


    

if __name__ == "__main__":
    # Création des services
    milvus_service = MilvusService(
        collection_name="documents_collection", 
        dim=1024  # Assurez-vous que cette dimension correspond à votre modèle d'embedding
    )
    embedding_service = EmbeddingService()
    search_service = SearchService(milvus_service, embedding_service, top_k=5)

    # Interface utilisateur simple
    print("\n=== Système de recherche documentaire ===\n")
    
    while True:
        query = input("\nEntrez votre requête de recherche (tapez 'exit' pour quitter) : ")
        if query.lower() == 'exit':
            print("Au revoir!")
            break
            
        if not query.strip():
            print("Veuillez entrer une requête valide.")
            continue
            
        # Effectuer la recherche
        start_time = time.time()
        results = search_service.search(query)
        total_time = time.time() - start_time
        
        # Afficher les résultats
        print(f"\n🔍 Documents trouvés ({total_time:.3f}s):")
        
        if not results:
            print("Aucun document pertinent trouvé.")
        else:
            for i, result in enumerate(results, 1):
                # Extraire les informations
                text = result.get("text", "Texte non disponible")
                score = result.get("score", 0.0)
                metadata = result.get("metadata", {})
                
                # Extraire les métadonnées pertinentes si disponibles
                filename = metadata.get("filename", "Document inconnu")
                page = metadata.get("page_number", "?")
                
                # Afficher un extrait du texte (max 100 caractères)
                text_preview = text[:100] + "..." if len(text) > 100 else text
                
                print(f"{i}. [{score:.2f}] {filename} (p.{page}): {text_preview}")