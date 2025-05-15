import re
import logging
import time
import json
import os
from typing import List, Dict, Any, Optional, Tuple

# Importer les services existants
from app.services.embedding_service import EmbeddingService
from app.services.milvus_service import MilvusService
from app.services.search_service import SearchService
from app.services.rerank_service import RerankService
from app.services.llm_service import LlmService
from fastapi import FastAPI

# Configuration du logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

class RAGSystem:
    """
    Système RAG complet pour le domaine juridique camerounais.
    Intègre la recherche, le reranking, la réorganisation du contexte et la génération de réponses.
    """
    def __init__(
        self, 
        collection_name: str = "documents_collection", 
        embedding_dim: int = 1024,
        top_k: int = 8,
        llm_model: str = "llama3.2:latest",
        max_context_length: int = 4000,
        save_dir: str = "./rag_results"
    ):
        """
        Initialise le système RAG avec tous ses composants.
        
        Args:
            collection_name: Nom de la collection Milvus
            embedding_dim: Dimension des embeddings
            top_k: Nombre de résultats à retourner lors de la recherche initiale
            llm_model: Nom du modèle LLM à utiliser
            max_context_length: Longueur maximale du contexte envoyé au LLM
            save_dir: Répertoire pour sauvegarder les résultats
        """
        # Créer le répertoire de sauvegarde s'il n'existe pas
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.top_k = top_k
        self.max_context_length = max_context_length

        # Initialisation des services
        try:
            # Services de base
            logger.info("🚀 Initialisation des services...")
            self.embedding_service = EmbeddingService()
            self.milvus_service = MilvusService(
                collection_name=collection_name, 
                dim=embedding_dim
            )
            logger.info("✅ Services d'embedding et Milvus initialisés")

            # Service de recherche
            self.search_service = SearchService(
                milvus_service=self.milvus_service,
                embedding_service=self.embedding_service,
                top_k=top_k
            )
            logger.info("✅ Service de recherche initialisé")

            # Service LLM
            try:
                self.llm_service = LlmService(model_name=llm_model)
                self.llm_available = True
                logger.info("✅ Service LLM initialisé")
            except Exception as llm_error:
                logger.warning(f"⚠️ Service LLM non disponible: {llm_error}")
                self.llm_service = None
                self.llm_available = False

            # Service de reranking
            self.rerank_service = RerankService(
                llm_service=self.llm_service,
                cache_size=100
            )
            logger.info("✅ Service de reranking initialisé")

        except Exception as e:
            logger.error(f"❌ Erreur lors de l'initialisation des services: {e}")
            raise RuntimeError(f"Impossible d'initialiser les services : {e}")

    def detect_query_type(self, query: str) -> str:
        """
        Détecte le type de requête juridique.
        
        Args:
            query: Requête de l'utilisateur
            
        Returns:
            Type de requête identifié
        """
        query_lower = query.lower()
        
        # Détection de recherche d'articles spécifiques
        if re.search(r"article\s+\d+", query_lower) or any(pattern in query_lower for pattern in ["article", "loi", "constitution"]):
            return "article"
        
        # Détection de demande de définition
        if any(term in query_lower for term in ["qu'est-ce que", "définition", "signifie", "définir"]):
            return "definition"
        
        # Détection de demande de procédure
        if any(term in query_lower for term in ["comment", "procédure", "processus", "démarche", "étape"]):
            return "procedure"
        
        # Détection de requête fiscale
        if any(term in query_lower for term in ["impôt", "taxe", "fiscal", "tva", "irpp", "revenu"]):
            return "fiscal"
        
        # Par défaut, requête générale
        return "general"

    def format_context_for_prompt(self, reranked_results: List[Dict[str, Any]]) -> str:
        """
        Formate les résultats réorganisés en un contexte structuré pour le LLM.
        
        Args:
            reranked_results: Liste de résultats reclassés et réorganisés
            
        Returns:
            Contexte formaté pour le prompt LLM
        """
        if not reranked_results:
            return "Aucune information pertinente trouvée."
        
        # Regrouper par document pour une meilleure organisation
        document_groups = {}
        for result in reranked_results:
            metadata = result.get("metadata", {})
            doc_id = metadata.get("document_id", "inconnu")
            filename = metadata.get("filename", "Document inconnu")
            page = metadata.get("page_number", "?")
            
            doc_key = f"{filename} (ID: {doc_id})"
            if doc_key not in document_groups:
                document_groups[doc_key] = []
            
            document_groups[doc_key].append({
                "text": result.get("text", ""),
                "page": page,
                "score": result.get("rerank_score", result.get("score", 0.0))
            })
        
        # Formater le contexte
        context_parts = []
        
        for doc_name, extracts in document_groups.items():
            # Ajouter le titre du document
            context_parts.append(f"### {doc_name}")
            
            # Trier les extraits par score (meilleurs d'abord)
            sorted_extracts = sorted(extracts, key=lambda x: x["score"], reverse=True)
            
            # Ajouter chaque extrait
            for extract in sorted_extracts:
                context_parts.append(f"[Page {extract['page']}] {extract['text']}")
            
            # Ajouter un séparateur entre les documents
            context_parts.append("---")
        
        return "\n\n".join(context_parts)

    def _create_prompt(self, query: str, context: str) -> str:
        """
        Méthode originale de création de prompt.
        Conservée pour compatibilité et comme méthode de secours.
        
        Args:
            query: Question de l'utilisateur
            context: Contexte formaté avec les informations pertinentes
            
        Returns:
            Prompt complet pour le LLM
        """
        # Base du prompt
        prompt = f"""En tant qu'expert juridique camerounais, je vais répondre à la question suivante 
    en me basant uniquement sur les extraits de documents fournis:

    QUESTION: {query}

    EXTRAITS DE DOCUMENTS PERTINENTS:
    {context}

    """
        
        # Instructions génériques par défaut
        prompt += """INSTRUCTIONS:
    - Répondez de manière claire, concise et précise
    - Utilisez uniquement les informations présentes dans les extraits
    - Citez les sources pertinentes (document, page)
    - Indiquez si la réponse est partielle ou incomplète
    - Structurez votre réponse de manière logique et compréhensible
    """
        
        prompt += "\nRÉPONSE:"
        
        return prompt

    def search_and_rerank(
        self, 
        query: str, 
        use_expansion: bool = True, 
        use_reranking: bool = True
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Effectue la recherche et le reranking pour une requête.
        
        Args:
            query: Requête de l'utilisateur
            use_expansion: Utiliser l'expansion de requête
            use_reranking: Appliquer le reranking aux résultats
            
        Returns:
            Tuple de (résultats reclassés, statistiques)
        """
        start_time = time.time()
        top_k = 5

        # Recherche avec ou sans expansion
        search_results = (
            self.search_service.search_with_expansion(query, top_k=top_k) 
            if use_expansion 
            else self.search_service.search(query, top_k=top_k)
        )
        
        # Initialiser les statistiques
        stats = {
            "timestamp": start_time,
            "use_expansion": use_expansion,
            "use_reranking": use_reranking
        }
        
        try:
            # Recherche avec ou sans expansion
            search_start = time.time()
            logger.info(f"🔍 Recherche pour la requête: '{query}'")
            
            search_results = (
                self.search_service.search_with_expansion(query) 
                if use_expansion 
                else self.search_service.search(query)
            )
            
            search_time = time.time() - search_start
            stats["search_time"] = search_time
            stats["raw_results_count"] = len(search_results)
            
            logger.info(f"✅ Recherche terminée en {search_time:.3f}s, {len(search_results)} résultats trouvés")
            
            # Vérifier si des résultats ont été trouvés
            if not search_results:
                logger.warning("⚠️ Aucun résultat trouvé pour cette requête")
                stats["total_time"] = time.time() - start_time
                return [], stats
            
            # Reranking si demandé
            if use_reranking and self.rerank_service:
                rerank_start = time.time()
                logger.info(f"📊 Reranking de {len(search_results)} résultats...")
                
                try:
                    reranked_results = self.rerank_service.rerank(
                        query=query,
                        results=search_results,
                        use_llm=False
                    )
                    
                    rerank_time = time.time() - rerank_start
                    stats["rerank_time"] = rerank_time
                    stats["total_time"] = time.time() - start_time
                    
                    logger.info(f"✅ Reranking terminé en {rerank_time:.3f}s")
                    
                    return reranked_results[:self.top_k], stats
                    
                except Exception as e:
                    logger.error(f"❌ Erreur lors du reranking: {e}")
                    # Fallback aux résultats originaux
                    return search_results[:self.top_k], stats
            else:
                # Retourner les résultats de recherche originaux
                return search_results[:self.top_k], stats
                
        except Exception as e:
            logger.error(f"❌ Erreur lors de la recherche et du reranking: {e}")
            return [], stats

    def generate_answer(self, query: str) -> Dict[str, Any]:
        try:
            # Recherche et reranking
            results, search_stats = self.search_and_rerank(query)
            
            if not results:
                return {
                    "query": query,
                    "answer": "Je n'ai pas trouvé d'informations pertinentes.",
                    "source_documents": [],
                    "stats": search_stats,
                    "success": False
                }
            
            # Détecter le type de requête
            query_type = self.detect_query_type(query)
            
            # Formater le contexte
            context = self.format_context_for_prompt(results)
            
            # Sélectionner dynamiquement le meilleur prompt
            prompt = self._select_best_prompt(query, context, query_type)
            
            try:
                # Tentative de génération LLM avec timeout et retry
                llm_start = time.time()
                response = self._generate_llm_response_with_fallback(prompt, results)
                
                return {
                    "query": query,
                    "answer": response,
                    "source_documents": results[:5],
                    "stats": {
                        **search_stats,
                        "llm_generation_time": time.time() - llm_start,
                        "query_type": query_type
                    },
                    "success": True
                }
            
            except Exception as e:
                # Fallback : générer une réponse à partir des extraits
                logging.warning(f"Fallback activé : {e}")
                summary_response = self._generate_summary_from_extracts(results)
                
                return {
                    "query": query,
                    "answer": summary_response,
                    "source_documents": results[:5],
                    "stats": search_stats,
                    "success": False,
                    "error": str(e)
                }
        
        except Exception as e:
            logging.error(f"Erreur globale : {e}")
            return {
                "query": query,
                "answer": "Une erreur est survenue lors du traitement de votre requête.",
                "source_documents": [],
                "stats": {},
                "success": False,
                "error": str(e)
            }

    def _generate_source_explanation(self, results: List[Dict]) -> str:
        """
        Génère une explication sur les sources des informations.
        """
        if not results:
            return "Aucune source n'a pu être identifiée."
        
        source_info = []
        unique_docs = set()
        
        for result in results[:3]:  # Limiter aux 3 premières sources
            metadata = result.get('metadata', {})
            filename = metadata.get('filename', 'Document inconnu')
            page = metadata.get('page_number', '?')
            document_id = metadata.get('document_id', '')
            
            if document_id not in unique_docs:
                unique_docs.add(document_id)
                source_info.append(f"- {filename} (page {page})")
        
        return f"Informations extraites de : \n" + "\n".join(source_info)
        
    def _generate_llm_response_with_fallback(self, prompt: str, results: List[Dict]) -> str:
        # Tentative de génération avec plusieurs méthodes
        methods = [
            lambda: self.llm_service.generate_response(prompt, timeout=120),
            lambda: self.llm_service.generate_response(prompt, max_length=300, timeout=60),
            lambda: self._generate_summary_from_extracts(results)
        ]
        
        for method in methods:
            try:
                response = method()
                if response and len(response) >= 20:
                    return response
            except Exception as e:
                logging.warning(f"Méthode de génération échouée : {e}")
        
        return "Je n'ai pas pu générer de réponse détaillée."

    def _generate_summary_from_extracts(self, results: List[Dict]) -> str:
        # Génération d'un résumé à partir des extraits
        summary_parts = []
        for result in results[:3]:  # Limiter aux 3 premiers résultats
            text = result.get('text', '')
            if text and len(text) > 50:
                summary_parts.append(f"- {text[:200]}...")
        
        return "Résumé des extraits pertinents :\n" + "\n".join(summary_parts)
    
    def handle_conversation(
        self, 
        message: str, 
        conversation_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Gère une conversation complète avec historique et contexte.
        
        Args:
            message: Message de l'utilisateur
            conversation_history: Historique des conversations précédentes
        
        Returns:
            Dictionnaire avec la réponse, les sources, et les explications
        """
        if conversation_history is None:
            conversation_history = []
        
        # Détecter le type de requête
        query_type = self.detect_query_type(message)
        
        # Adapter le prompt en fonction du contexte de conversation
        if query_type == "clarification" and conversation_history:
            # Si c'est une demande de clarification, utiliser le contexte précédent
            last_query = conversation_history[-1].get('query', '')
            message = f"Peux-tu m'expliquer plus en détail la réponse précédente à propos de : {last_query}"
        
        # Recherche et reranking
        results, search_stats = self.search_and_rerank(message)
        
        if not results:
            return {
                "query": message,
                "answer": "Je n'ai pas trouvé d'informations pertinentes sur ce sujet.",
                "explanation": "Ma recherche dans les documents juridiques n'a pas permis de trouver des informations correspondant à votre requête.",
                "source_documents": [],
                "stats": search_stats,
                "success": False
            }
        
        # Formater les documents sources de manière sérialisable
        formatted_results = self._format_source_documents(results)
        
        # Formater le contexte
        context = self.format_context_for_prompt(formatted_results)
        
        # Créer un prompt adaptatif
        prompt = self._create_adaptive_prompt(message, context, query_type)
        
        try:
            # Génération de la réponse
            response = self._generate_llm_response_with_fallback(prompt, formatted_results)
            
            # Générer une explication sur la source des informations
            source_explanation = self._generate_source_explanation(formatted_results)
            
            return {
                "query": message,
                "answer": response,
                "explanation": source_explanation,
                "source_documents": formatted_results[:3],  # Limiter aux 3 premières sources
                "stats": search_stats,
                "success": True
            }
        
        except Exception as e:
            return {
                "query": message,
                "answer": "Je n'ai pas pu générer de réponse complète.",
                "explanation": f"Une erreur est survenue lors de la génération de la réponse : {str(e)}",
                "source_documents": [],
                "stats": search_stats,
                "success": False
            }

    def _create_adaptive_prompt(self, query: str, context: str, query_type: str) -> str:
        """
        Crée un prompt adaptatif en fonction du type de requête.
        
        Args:
            query: Requête de l'utilisateur
            context: Contexte formaté
            query_type: Type de requête détecté
        
        Returns:
            Prompt adapté au type de requête
        """
        base_prompt = f"""En tant qu'expert juridique camerounais, je vais répondre à la question suivante 
    en me basant uniquement sur les extraits de documents fournis:

    QUESTION: {query}

    EXTRAITS DE DOCUMENTS PERTINENTS:
    {context}

    """
        
        # Instructions spécifiques selon le type de requête
        if query_type == "article":
            base_prompt += """INSTRUCTIONS SPÉCIFIQUES POUR LES ARTICLES:
    - Citez précisément le texte de l'article demandé s'il est présent dans les documents
    - Expliquez clairement la signification et l'importance de cet article
    - Mentionnez la source exacte (document, page) des informations
    - Donnez un contexte détaillé et compréhensible
    - Si l'article n'est pas intégralement présent, indiquez-le clairement
    """
        elif query_type == "definition":
            base_prompt += """INSTRUCTIONS SPÉCIFIQUES POUR LES DÉFINITIONS:
    - Fournir une définition précise et complète du terme juridique
    - Expliquez le contexte et l'importance de ce terme dans le droit camerounais
    - Citez les sources pertinentes des définitions
    - Rendez la définition accessible et claire
    - Mentionnez les implications légales si pertinent
    """
        elif query_type == "procedure":
            base_prompt += """INSTRUCTIONS SPÉCIFIQUES POUR LES PROCÉDURES:
    - Décrivez les étapes de la procédure de manière chronologique et détaillée
    - Identifiez les documents requis, les délais et les autorités compétentes
    - Expliquez chaque étape en termes simples et pratiques
    - Fournissez des conseils ou des points d'attention importants
    - Citez les sources réglementaires de chaque étape
    """
        elif query_type == "fiscal":
            base_prompt += """INSTRUCTIONS SPÉCIFIQUES POUR LES QUESTIONS FISCALES:
    - Détaillez précisément les aspects fiscaux mentionnés
    - Expliquez les taux, exonérations et obligations fiscales
    - Donnez des exemples concrets et pratiques
    - Situez les dispositions dans le contexte fiscal camerounais
    - Indiquez les références légales spécifiques
    """
        else:
            # Utiliser les instructions génériques si aucun type spécifique n'est détecté
            base_prompt += """INSTRUCTIONS GÉNÉRIQUES:
    - Répondez de manière claire, concise et précise
    - Utilisez uniquement les informations des extraits fournis
    - Rendez votre réponse accessible et compréhensible
    - Donnez un aperçu contextuel si possible
    - Citez les sources de vos informations
    """
        
        base_prompt += "\nRÉPONSE DÉTAILLÉE:"
        
        return base_prompt

    def _select_best_prompt(self, query: str, context: str, query_type: str) -> str:
        """
        Sélectionne dynamiquement le meilleur prompt en fonction du contexte.
        
        Args:
            query: Requête de l'utilisateur
            context: Contexte formaté
            query_type: Type de requête détecté
        
        Returns:
            Prompt le plus approprié
        """
        try:
            # Tenter d'abord le prompt adaptatif
            adaptive_prompt = self._create_adaptive_prompt(query, context, query_type)
            
            # Vérifier si le prompt adaptatif est suffisamment long et informatif
            if len(adaptive_prompt) > 200:  # S'assurer qu'il contient des instructions substantielles
                return adaptive_prompt
            
            # Fallback au prompt générique si le prompt adaptatif est trop court
            return self._create_prompt(query, context)
        
        except Exception as e:
            # En cas d'erreur, utiliser le prompt générique
            logger.warning(f"Erreur lors de la création du prompt adaptatif: {e}")
            return self._create_prompt(query, context)


    def _generate_source_explanation(self, results: List[Dict[str, Any]]) -> str:
        """
        Génère une explication détaillée sur l'origine des informations.
        
        Args:
            results: Liste des résultats de recherche
        
        Returns:
            Chaîne expliquant l'origine des informations
        """
        if not results:
            return "Aucune source n'a pu être identifiée."
        
        # Extraire les informations de source uniques
        sources = {}
        for result in results[:3]:  # Limiter aux 3 premières sources
            metadata = result.get('metadata', {})
            filename = metadata.get('filename', 'Document inconnu')
            page = metadata.get('page_number', '?')
            document_id = metadata.get('document_id', '')
            
            # Utiliser le document_id comme clé unique
            if document_id not in sources:
                sources[document_id] = {
                    'filename': filename,
                    'pages': set(),
                    'chunks_count': 0
                }
            sources[document_id]['pages'].add(page)
            sources[document_id]['chunks_count'] += 1
        
        # Construire l'explication
        explanation_parts = []
        for source_details in sources.values():
            pages_str = ', '.join(map(str, sorted(source_details['pages'])))
            explanation_parts.append(
                f"- {source_details['filename']} (pages {pages_str}, "
                f"{source_details['chunks_count']} segment(s) pertinent(s))"
            )
        
        return "Informations extraites de :\n" + "\n".join(explanation_parts)


    def _format_source_documents(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Formate les documents sources pour correspondre au modèle attendu.
        Gère différents types de résultats, y compris les objets Milvus.
        
        Args:
            results: Liste des résultats de recherche
            
        Returns:
            Liste de documents formatés
        """
        formatted_docs = []
        
        for result in results:
            # Gérer les différents formats de résultat
            try:
                # Si c'est un objet Milvus Hit
                if hasattr(result, 'entity'):
                    # Extraire les informations de l'entité
                    entity = result.entity
                    
                    # Initialiser un dictionnaire de résultat
                    formatted_result = {
                        "score": getattr(result, "distance", 0.0),
                        "metadata": {}
                    }
                    
                    # Essayer de récupérer le texte
                    text_fields = ['text', 'content', 'data']
                    for field in text_fields:
                        try:
                            text = getattr(entity, field, None)
                            if text:
                                formatted_result["text"] = text
                                break
                        except:
                            continue
                    
                    # Récupérer les métadonnées
                    try:
                        # Essayer de charger les métadonnées JSON si possible
                        metadata_str = getattr(entity, "metadata", "{}")
                        if isinstance(metadata_str, str):
                            try:
                                metadata = json.loads(metadata_str)
                            except:
                                metadata = {}
                        elif isinstance(metadata_str, dict):
                            metadata = metadata_str
                        else:
                            metadata = {}
                        
                        # Ajouter d'autres champs de métadonnées possibles
                        for field in ['document_id', 'chunk_id', 'filename', 'page_number']:
                            try:
                                value = getattr(entity, field, None)
                                if value:
                                    metadata[field] = value
                            except:
                                pass
                        
                        formatted_result["metadata"] = metadata
                    except Exception as meta_error:
                        logger.warning(f"Erreur lors de l'extraction des métadonnées : {meta_error}")
                    
                    # Ajouter à la liste des documents formatés
                    formatted_docs.append(formatted_result)
                
                # Si c'est déjà un dictionnaire
                elif isinstance(result, dict):
                    # Vérifier et formater les documents existants
                    formatted_result = {
                        "text": result.get("text", ""),
                        "score": result.get("score", 0.0),
                        "metadata": result.get("metadata", {})
                    }
                    formatted_docs.append(formatted_result)
                
                # Gérer d'autres types de résultats potentiels
                else:
                    # Conversion de secours
                    formatted_docs.append({
                        "text": str(result),
                        "score": 0.0,
                        "metadata": {}
                    })
            
            except Exception as e:
                logger.error(f"Erreur lors du formatage d'un document source : {e}")
                # Document de secours en cas d'erreur
                formatted_docs.append({
                    "text": "Erreur de traitement du document",
                    "score": 0.0,
                    "metadata": {"error": str(e)}
                })
        
        return formatted_docs

# Point d'entrée principal
if __name__ == "__main__":
    try:
        # Initialiser le système RAG
        rag_system = RAGSystem()
        
        # Lancer la session interactive
        rag_system.interactive_session()
        
    except Exception as e:
        print(f"Erreur lors de l'initialisation du système : {e}")