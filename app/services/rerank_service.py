import logging
import re
import numpy as np
import json
from typing import List, Dict, Tuple, Any, Optional, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from threading import Lock
from app.services.llm_service import LlmService

# Configuration du logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class RerankService:
 
    def __init__(self, llm_service: Optional[LlmService] = None, cache_size: int = 1000):
        """
        Initialise le service de reranking avec des méthodes multiples.
        
        Args:
            llm_service: Service LLM pour le reranking sémantique (optionnel)
            cache_size: Taille du cache pour les scores LLM
        """
        # Initialiser le service LLM avec gestion des erreurs
        try:
            self.llm_service = llm_service if llm_service else LlmService("llama3.2", server_url="http://10.100.212.118:8001", check_connection=False)
            self.llm_available = True
            logger.info("✅ Service LLM initialisé avec succès")
        except Exception as e:
            self.llm_service = None
            self.llm_available = False
            logger.warning(f"⚠️ Service LLM non disponible: {e}. Fonctionnement en mode dégradé.")
        
        self.tfidf_vectorizer = TfidfVectorizer(
            min_df=1, max_df=0.9, stop_words='english', 
            ngram_range=(1, 2), use_idf=True
        )
        
        # Cache pour les scores LLM (pour éviter de recalculer)
        self.score_cache = {}
        self.cache_size = cache_size
        self.cache_lock = Lock()
        
        # Poids pour les différentes méthodes - optimisés pour la documentation juridique camerounaise
        self.weights = {
            "bm25": 0.30,       # Score lexical/BM25 (optimisé pour le juridique)
            "tfidf": 0.20,      # Score TF-IDF
            "original": 0.15,   # Score original de similarité vectorielle
            "legal": 0.20,      # Nouveau score de pertinence juridique
            "llm": 0.15         # Score LLM (réduit pour plus de prévisibilité)
        }
        
        # Ajustement des poids si LLM n'est pas disponible
        if not self.llm_available:
            # Redistribuer le poids du LLM aux autres méthodes
            llm_weight = self.weights["llm"]
            self.weights["llm"] = 0.0
            self.weights["bm25"] += llm_weight * 0.4
            self.weights["tfidf"] += llm_weight * 0.3
            self.weights["legal"] += llm_weight * 0.3
            logger.info(f"Poids ajustés en l'absence de LLM: {self.weights}")
        
        # Termes juridiques camerounais pour le scoring
        self.legal_terms = [
            "impôt", "déclaration", "taxe", "revenu", "contribuable", "assujetti", 
            "article", "paragraphe", "alinéa", "section", "chapitre", "titre",
            "loi", "décret", "arrêté", "circulaire", "code", "disposition",
            "irpp", "tva", "is", "itl", "fiscal", "exonération", "imposition",
            "cameroun", "camerounais", "dgi", "cgi", "redevance", "constitution"
        ]


    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalise une liste de scores dans [0,1]."""
        if not scores:
            return []
            
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
            
        return [(score - min_score) / (max_score - min_score) for score in scores]

    def _calculate_bm25_score(self, query: str, text: str) -> float:
        """
        Calcule un score de type BM25 simplifié entre la requête et le texte.
        Cette implémentation est une approximation de BM25 adaptée aux documents juridiques.
        """
        # Paramètres BM25
        k1 = 1.5  # Augmenté pour documents juridiques (valeur standard: 1.2)
        b = 0.70  # Légèrement réduit pour documents juridiques (valeur standard: 0.75)
        
        # Tokenisation améliorée pour le français et termes juridiques
        query_terms = re.findall(r'\w+', query.lower())
        text_terms = re.findall(r'\w+', text.lower())
        
        if not query_terms or not text_terms:
            return 0.0
            
        # Calculer la fréquence des termes dans le document
        term_freq = {}
        for term in text_terms:
            term_freq[term] = term_freq.get(term, 0) + 1
            
        # Calculer la longueur du document
        doc_length = len(text_terms)
        avg_doc_length = doc_length  # Pour une implémentation complète, utilisez la moyenne de votre corpus
        
        # Calculer le score BM25
        score = 0.0
        for term in query_terms:
            if term in term_freq:
                # Formule BM25 simplifiée
                tf = term_freq[term]
                term_score = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_length / avg_doc_length))
                
                # Bonus pour les termes juridiques
                if term in self.legal_terms:
                    term_score *= 1.5  # Amplifier l'importance des termes juridiques
                    
                score += term_score
                
        return score

    def _calculate_tfidf_score(self, query: str, documents: List[str]) -> List[float]:
        """
        Calcule les scores TF-IDF entre la requête et une liste de documents.
        
        Args:
            query: Requête de l'utilisateur
            documents: Liste de textes à comparer
            
        Returns:
            Liste de scores TF-IDF
        """
        if not documents:
            return []
            
        # Ajouter la requête à la liste des documents pour la vectorisation
        all_texts = documents + [query]
        
        try:
            # Configurer le vectoriseur pour le français et documents juridiques
            self.tfidf_vectorizer = TfidfVectorizer(
                min_df=1, max_df=0.9, 
                ngram_range=(1, 3),  # Augmenté pour capturer les expressions juridiques
                use_idf=True,
                stop_words=[  # Mots vides français courants
                    "le", "la", "les", "un", "une", "des", "et", "ou", "de", "du", "en",
                    "au", "aux", "avec", "ce", "ces", "dans", "sur", "pour", "par", "est"
                ]
            )
            
            # Entraîner le vectoriseur TF-IDF
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
            
            # Calculer la similarité cosinus entre la requête et chaque document
            query_vector = tfidf_matrix[-1]  # Vecteur de la requête (dernier élément)
            document_vectors = tfidf_matrix[:-1]  # Vecteurs des documents
            
            similarities = cosine_similarity(query_vector, document_vectors).flatten()
            return similarities.tolist()
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul TF-IDF: {e}")
            return [0.0] * len(documents)

    def _get_llm_score(self, query: str, text: str) -> float:
        """
        Obtient un score de pertinence via LLM, avec mise en cache.
        Optimisé pour les questions juridiques.
        
        Args:
            query: Requête utilisateur
            text: Texte à évaluer
            
        Returns:
            Score entre 0 et 1
        """
        if not self.llm_available or not self.llm_service:
            return 0.5  # Score neutre si LLM non disponible
        
        try:
            # Utiliser des kwargs pour gérer différents paramètres
            response = self.llm_service.generate_response(
                prompt=f"...",
                timeout=60,  # Ajouter un timeout
                temperature=0.5
            )
            # Reste du code inchangé
        except Exception as e:
            logger.warning(f"Erreur lors de l'obtention du score LLM: {e}")
            return 0.5
            
        # Créer une clé de cache unique
        cache_key = f"{hash(query)}_{hash(text)}"
        
        # Vérifier si le score est déjà en cache
        with self.cache_lock:
            if cache_key in self.score_cache:
                return self.score_cache[cache_key]
        
        try:
            # Créer un prompt pour évaluer la pertinence juridique
            prompt = (
                f"Vous êtes un expert juridique camerounais. Sur une échelle de 0 à 1, évaluez la pertinence "
                f"de ce texte par rapport à la requête juridique.\n\n"
                f"Requête: {query}\n\n"
                f"Texte: {text}\n\n"
                f"Donnez uniquement un nombre entre 0 et 1 (sans explication): "
            )
            
            # Générer la réponse
            response = self.llm_service.generate_response(prompt)
            
            # Extraire le score numérique de la réponse
            score_match = re.search(r"([0-9]*[.])?[0-9]+", response)
            if score_match:
                score = float(score_match.group(0))
                score = max(0.0, min(1.0, score))  # Limiter entre 0 et 1
            else:
                logger.warning(f"Impossible d'extraire un score numérique de la réponse LLM: {response}")
                score = 0.5  # Valeur par défaut
            
            # Mettre en cache le résultat
            with self.cache_lock:
                if len(self.score_cache) >= self.cache_size:
                    # Supprimer une entrée aléatoire si le cache est plein
                    self.score_cache.pop(next(iter(self.score_cache)))
                self.score_cache[cache_key] = score
            
            return score
            
        except Exception as e:
            logger.error(f"Erreur lors de l'obtention du score LLM: {e}")
            return 0.5  # Valeur neutre en cas d'erreur

    def _calculate_legal_relevance_score(self, query: str, text: str) -> float:
        """
        Calcule un score de pertinence spécifique aux documents juridiques camerounais.
        
        Args:
            query: Requête de l'utilisateur
            text: Texte à évaluer
            
        Returns:
            Score de pertinence juridique entre 0 et 1
        """
        if not text:
            return 0.0
            
        # 1. Vérifier la présence de termes juridiques pertinents
        legal_term_count = sum(1 for term in self.legal_terms if term in text.lower())
        legal_term_score = min(legal_term_count / 5, 1.0)  # Max 5 termes pour un score de 1
        
        # 2. Vérifier si le texte contient des références à des articles spécifiques
        article_pattern = r'article\s+\d+'
        article_matches = re.findall(article_pattern, text.lower())
        article_score = min(len(article_matches) / 3, 1.0)  # Max 3 références pour un score de 1
        
        # 3. Vérifier si le texte contient des informations numériques précises (montants, pourcentages)
        number_pattern = r'\d+(?:[.,]\d+)?(?:\s*%|\s*francs|\s*FCFA|\s*XAF)?'
        number_matches = re.findall(number_pattern, text)
        number_score = min(len(number_matches) / 5, 1.0)  # Max 5 valeurs numériques pour un score de 1
        
        # 4. Pertinence de la date (des documents plus récents pourraient être plus pertinents)
        date_score = 0.0
        date_patterns = [
            r'\b\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}\b',  # 01/01/2022
            r'\b\d{4}\b'  # 2022
        ]
        
        for pattern in date_patterns:
            dates = re.findall(pattern, text)
            if dates:
                try:
                    # Tenter d'extraire l'année
                    years = [int(re.search(r'\d{4}', date).group(0)) for date in dates if re.search(r'\d{4}', date)]
                    if years:
                        most_recent_year = max(years)
                        if 2000 <= most_recent_year <= 2025:  # Année raisonnable
                            # Plus l'année est récente, meilleur est le score
                            date_score = (most_recent_year - 2000) / 25  # Normaliser entre 0 et 1
                            break
                except:
                    pass
        
        # 5. Bonus pour les textes contenant des termes spécifiques de la requête
        query_terms = set(re.findall(r'\w+', query.lower()))
        important_matches = sum(1 for term in query_terms if term in text.lower() and len(term) > 3)
        query_score = min(important_matches / len(query_terms), 1.0) if query_terms else 0.0
        
        # 6. Bonus pour les textes qui semblent être des définitions juridiques
        definition_score = 0.0
        definition_patterns = [
            r'signifie\s+',
            r'est\s+défini\s+comme',
            r'désigne\s+',
            r'se\s+définit\s+',
            r'on\s+entend\s+par',
            r':\s+\w+'
        ]
        for pattern in definition_patterns:
            if re.search(pattern, text.lower()):
                definition_score = 0.5
                break
        
        # Combiner les scores avec des poids
        score = (
            legal_term_score * 0.25 +
            article_score * 0.15 +
            number_score * 0.10 +
            date_score * 0.10 +
            query_score * 0.30 +
            definition_score * 0.10
        )
        
        return score

    def _calculate_article_match_score(self, query: str, text: str) -> float:
        """
        Score spécial pour les correspondances d'articles spécifiques.
        Donne un score très élevé aux textes qui contiennent exactement l'article demandé.
        """
        # Extraire le numéro d'article de la requête 
        article_match = re.search(r'article\s+(\d+)', query.lower())
        if not article_match:
            return 0.0  # Pas de numéro d'article dans la requête
            
        article_num = article_match.group(1)
        
        # Chercher des modèles d'article dans le texte
        text_lower = text.lower()
        patterns = [
            fr"article\s+{article_num}\b",
            fr"article\s+{article_num}[^\d]",
            fr"article{article_num}\b",
            fr"art\.\s*{article_num}\b",
            fr"art\s+{article_num}\b"
        ]
        
        # Si l'un des modèles correspond exactement, donner un score très élevé
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return 1.0
        
        # Vérifier les variantes (bis, ter, etc.)
        variants = ["bis", "ter", "quater"]
        for variant in variants:
            for pattern in patterns:
                variant_pattern = pattern.replace(fr"{article_num}\b", fr"{article_num}\s+{variant}\b")
                if re.search(variant_pattern, text_lower):
                    return 0.9  # Score légèrement inférieur pour les variantes
                    
        return 0.0

    def rerank(self, query: str, results: List[Dict[str, Any]], use_llm: bool = False) -> List[Dict[str, Any]]:
        """
        Rerank les résultats en utilisant une approche hybride spécialisée pour le droit camerounais.
        
        Args:
            query: Requête utilisateur
            results: Liste de résultats initiaux (dictionnaires avec au moins 'text' et 'score')
            use_llm: Utiliser ou non le LLM pour le reranking
            
        Returns:
            Liste triée des résultats reclassés
        """
        if not results:
            logger.warning("⚠️ Aucun résultat à reclasser.")
            return []
            
        logger.info(f"📊 Début du reranking pour {len(results)} résultats.")
        
        # Vérifier si le LLM est disponible
        use_llm = use_llm and self.llm_available
        if not use_llm and self.llm_available:
            logger.info("LLM disponible mais non utilisé pour le reranking (désactivé par l'utilisateur)")
        elif not self.llm_available and use_llm:
            logger.warning("⚠️ LLM demandé mais non disponible. Utilisation des méthodes alternatives.")
        
        # Extraire les textes et scores originaux
        texts = []
        original_scores = []
        
        for r in results:
            # Gestion des données au format JSON
            if isinstance(r, dict):
                # Si le résultat est dans le format JSON que vous avez partagé
                text = r.get("text", "")
                if not text and "metadata" in r:
                    # Le texte pourrait être dans un autre format
                    text = str(r)  # Convertir tout en texte comme solution de repli
                score = r.get("score", 0.0)
            elif isinstance(r, tuple) and len(r) >= 2:
                # Format (texte, score) pour la compatibilité
                text, score = r[0], r[1]
            else:
                # Format inconnu, essayer de l'interpréter
                try:
                    if hasattr(r, "text") and hasattr(r, "score"):
                        text, score = r.text, r.score
                    else:
                        text, score = str(r), 0.0
                except:
                    text, score = str(r), 0.0
            
            texts.append(text)
            original_scores.append(score)
        
        try:
            # 1. Calculer les scores BM25
            bm25_scores = [self._calculate_bm25_score(query, text) for text in texts]
            bm25_scores = self._normalize_scores(bm25_scores)
            
            # 2. Calculer les scores TF-IDF
            tfidf_scores = self._calculate_tfidf_score(query, texts)
            tfidf_scores = self._normalize_scores(tfidf_scores)
            
            # 3. Normaliser les scores originaux
            normalized_original_scores = self._normalize_scores(original_scores)
            
            # 4. Calculer les scores de pertinence juridique
            legal_scores = [self._calculate_legal_relevance_score(query, text) for text in texts]
            
            # 5. NOUVEAU: Calculer les scores de correspondance d'article
            article_scores = [self._calculate_article_match_score(query, text) for text in texts]
            
            # Vérifier si des scores d'article ont été trouvés
            has_article_matches = any(score > 0 for score in article_scores)
            
            # 6. Calculer les scores LLM si demandé
            llm_scores = []
            if use_llm and self.llm_service:
                logger.info("🧠 Utilisation du LLM pour le reranking...")
                # Ne calculer le score LLM que pour les top N résultats
                # pour réduire la charge sur le LLM
                top_n = min(len(results), 5)  # Limiter à 5 requêtes LLM max
                
                # Trouver les top_n documents avec les meilleurs scores combinés (sans LLM)
                combined_initial = []
                for i in range(len(results)):
                    weighted_score = (
                        self.weights["bm25"] * bm25_scores[i] +
                        self.weights["tfidf"] * tfidf_scores[i] +
                        self.weights["original"] * normalized_original_scores[i] +
                        self.weights["legal"] * legal_scores[i]
                    )
                    
                    # Ajouter un boost pour les correspondances d'article
                    if article_scores[i] > 0:
                        weighted_score += article_scores[i] * 0.5  # Boost pour faciliter la sélection des articles pertinents
                    
                    combined_initial.append((i, weighted_score))
                
                # Tri et sélection des indices des top_n documents
                top_indices = [idx for idx, _ in sorted(combined_initial, key=lambda x: x[1], reverse=True)[:top_n]]
                
                # Initialiser tous les scores LLM à 0
                llm_scores = [0.0] * len(results)
                
                # Calculer les scores LLM uniquement pour les top_n documents
                for idx in top_indices:
                    llm_scores[idx] = self._get_llm_score(query, texts[idx])
                
                logger.info(f"✅ Scores LLM calculés pour {top_n} documents.")
            else:
                # Si LLM non utilisé, mettre tous les poids sur les autres méthodes
                if self.weights["llm"] > 0:
                    llm_weights = self.weights["llm"]
                    adjusted_weights = {
                        "bm25": self.weights["bm25"] / (1 - llm_weights) * 1.2,  # Augmenter BM25
                        "tfidf": self.weights["tfidf"] / (1 - llm_weights) * 1.1,  # Augmenter TFIDF légèrement
                        "original": self.weights["original"] / (1 - llm_weights),
                        "legal": self.weights["legal"] / (1 - llm_weights) * 1.3,  # Augmenter legal score davantage
                        "llm": 0.0
                    }
                    
                    # Normaliser pour que la somme soit 1
                    total = sum(adjusted_weights.values())
                    self.weights = {k: v/total for k, v in adjusted_weights.items()}
                
                # Initialiser les scores LLM à 0
                llm_scores = [0.0] * len(results)
            
            # 7. Combiner tous les scores avec ajout du score d'article
            final_scores = []
            
            # Si correspondance d'article trouvée, ajuster les poids
            article_weight = 0.4 if has_article_matches else 0.0
            
            if has_article_matches:
                # Ajuster les poids pour inclure le score d'article
                remaining_weight = 1.0 - article_weight
                effective_weights = {
                    "bm25": self.weights["bm25"] * remaining_weight,
                    "tfidf": self.weights["tfidf"] * remaining_weight,
                    "original": self.weights["original"] * remaining_weight,
                    "legal": self.weights["legal"] * remaining_weight,
                    "llm": self.weights["llm"] * remaining_weight
                }
                
                # Logger l'ajustement pour le débogage
                logger.info(f"Correspondances d'articles trouvées, ajustement des poids avec article_weight={article_weight}")
                
                for i in range(len(results)):
                    weighted_score = (
                        effective_weights["bm25"] * bm25_scores[i] +
                        effective_weights["tfidf"] * tfidf_scores[i] +
                        effective_weights["original"] * normalized_original_scores[i] +
                        effective_weights["legal"] * legal_scores[i] +
                        effective_weights["llm"] * llm_scores[i] +
                        article_weight * article_scores[i]  # Score de correspondance d'article
                    )
                    final_scores.append(weighted_score)
            else:
                # Utiliser les poids standard si pas de correspondance d'article
                for i in range(len(results)):
                    weighted_score = (
                        self.weights["bm25"] * bm25_scores[i] +
                        self.weights["tfidf"] * tfidf_scores[i] +
                        self.weights["original"] * normalized_original_scores[i] +
                        self.weights["legal"] * legal_scores[i] +
                        self.weights["llm"] * llm_scores[i]
                    )
                    final_scores.append(weighted_score)
            
            # 8. Créer les résultats reclassés avec conservation des formats JSON
            reranked_results = []
            for i, result in enumerate(results):
                # Copier le résultat original
                if isinstance(result, dict):
                    reranked = result.copy()
                else:
                    # Convertir en dictionnaire si ce n'est pas déjà le cas
                    reranked = {
                        "text": texts[i],
                        "score": original_scores[i]
                    }
                
                # Ajouter les scores
                reranked["original_score"] = original_scores[i]
                reranked["rerank_score"] = final_scores[i]
                reranked["score_details"] = {
                    "bm25": bm25_scores[i],
                    "tfidf": tfidf_scores[i],
                    "original": normalized_original_scores[i],
                    "legal": legal_scores[i],
                    "article": article_scores[i], 
                    "llm": llm_scores[i]
                }
                
                reranked_results.append(reranked)
            
            # 9. Trier par score final décroissant
            reranked_results.sort(key=lambda x: x["rerank_score"], reverse=True)
            
            # 10. Grouper par document pour améliorer la cohérence des résultats
            reranked_results = self._group_by_document(reranked_results)
            
            logger.info(f"📈 Reranking terminé: {len(reranked_results)} résultats reclassés.")
            return reranked_results
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du reranking: {e}")
            # En cas d'erreur, retourner les résultats originaux
            return results

    def _group_by_document(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Groupe les résultats par document tout en préservant les plus pertinents.
        Réorganise les résultats pour maximiser la pertinence et la cohérence.
        
        Args:
            results: Liste de résultats reclassés
            
        Returns:
            Liste réorganisée des résultats
        """
        if not results:
            return []
        
        # 1. Organiser les résultats par document
        documents = {}
        for result in results:
            metadata = result.get("metadata", {})
            doc_id = metadata.get("document_id", "unknown")
            if doc_id not in documents:
                documents[doc_id] = []
            documents[doc_id].append(result)
        
        # 2. Identifier les documents avec correspondance d'article (très pertinents)
        priority_docs = {}
        regular_docs = {}
        
        for doc_id, doc_results in documents.items():
            # Vérifier si ce document contient des correspondances d'article
            has_article_match = any(
                r.get("score_details", {}).get("article", 0) > 0.5 
                for r in doc_results
            )
            
            if has_article_match:
                priority_docs[doc_id] = doc_results
            else:
                regular_docs[doc_id] = doc_results
        
        # 3. Trier chaque groupe par score de pertinence
        for doc_id in list(priority_docs.keys()) + list(regular_docs.keys()):
            if doc_id in priority_docs:
                priority_docs[doc_id].sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
            else:
                regular_docs[doc_id].sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        
        # 4. Trier les documents eux-mêmes par leur meilleur score
        sorted_priority_docs = sorted(
            priority_docs.items(),
            key=lambda x: max(r.get("rerank_score", 0.0) for r in x[1]),
            reverse=True
        )
        
        sorted_regular_docs = sorted(
            regular_docs.items(),
            key=lambda x: max(r.get("rerank_score", 0.0) for r in x[1]),
            reverse=True
        )
        
        # 5. Construire la liste finale en privilégiant les documents prioritaires
        final_results = []
        
        # D'abord, ajouter les meilleurs extraits des documents prioritaires
        for doc_id, doc_results in sorted_priority_docs:
            # Ajouter le meilleur extrait de ce document
            final_results.append(doc_results[0])
        
        # Ensuite, ajouter les meilleurs extraits des documents standards
        for doc_id, doc_results in sorted_regular_docs:
            # Ajouter le meilleur extrait de ce document
            if doc_results:
                final_results.append(doc_results[0])
        
        # Maintenant, ajouter d'autres extraits des documents prioritaires pour le contexte
        for doc_id, doc_results in sorted_priority_docs:
            if len(doc_results) > 1:
                # Ajouter jusqu'à 2 extraits supplémentaires des documents prioritaires
                final_results.extend(doc_results[1:min(3, len(doc_results))])
        
        # Enfin, ajouter des extraits supplémentaires des documents standards
        # mais seulement pour les documents très pertinents
        high_score_threshold = 0.6
        for doc_id, doc_results in sorted_regular_docs:
            if doc_results and doc_results[0].get("rerank_score", 0.0) >= high_score_threshold:
                # Ajouter 1 extrait supplémentaire des documents très pertinents
                if len(doc_results) > 1:
                    final_results.append(doc_results[1])
        
        return final_results

# Exemple d'utilisation
if __name__ == "__main__":
    try:
        # Tentative d'initialisation avec LLM
        from llm_service import LlmService
        llm_service = LlmService("llama3.2", server_url="http://10.100.212.118:8001", check_connection=False)
        rerank_service = RerankService(llm_service, cache_size=100)
        logger.info("Service de reranking initialisé avec LLM")
    except Exception as e:
        # Fallback sans LLM
        logger.warning(f"Impossible de créer le service LLM : {e}. Utilisation du reranking sans LLM.")
        rerank_service = RerankService(llm_service=None, cache_size=100)
    
    # Exemple de résultats de recherche au format JSON
    query = "Quelles sont les mesures de promotions de l'import-substitution ?"
    results = [
        {
            "text": "sable à l'importation des « compléments alimentaires » (vitamines, acides aminés et sels minéraux), non produits localement, destinés aux préparations alimentaires de provenderie pour le renforcement de la croissance des animaux.",
            "score": 0.68,
            "metadata": {
                "chunk_id": "0ab7ef1e1dbfab6ee85d9a6ed8bf7305",
                "document_id": "71cf1b815c8bb293c4dd1181666ff190",
                "filename": "CIREX-2025-FR.pdf",
                "page_number": 14
            }
        },
        {
            "text": "Le bénéfice de l'exemption de TVA sur les importations de denrées de première nécessité est accordé dans le cadre des mesures de promotion de l'import-substitution.",
            "score": 0.58,
            "metadata": {
                "chunk_id": "1ab7ef1e1dbfab6ee85d9a6ed8bf7306",
                "document_id": "71cf1b815c8bb293c4dd1181666ff190",
                "filename": "CIREX-2025-FR.pdf",
                "page_number": 22
            }
        }
    ]
    
    # Essayer de reclasser avec LLM
    try:
        print("\n=== Reranking avec LLM ===")
        reranked_with_llm = rerank_service.rerank(query, results, use_llm=True)
        for i, r in enumerate(reranked_with_llm, 1):
            print(f"{i}. [{r['rerank_score']:.4f}] {r['text'][:100]}... (fichier: {r['metadata']['filename']})")
    except Exception as e:
        print(f"Erreur lors du reranking avec LLM: {e}")
    
    # Reclasser sans LLM (mode économique)
    try:
        print("\n=== Reranking sans LLM (économique) ===")
        reranked_without_llm = rerank_service.rerank(query, results, use_llm=False)
        for i, r in enumerate(reranked_without_llm, 1):
            print(f"{i}. [{r['rerank_score']:.4f}] {r['text'][:100]}... (fichier: {r['metadata']['filename']})")
    except Exception as e:
        print(f"Erreur lors du reranking sans LLM: {e}")
        
    # Test avec une requête juridique plus complexe (articles de loi)
    try:
        article_query = "Que dit l'article 40 de la constitution camerounaise ?"
        print(f"\n=== Recherche spécifique: {article_query} ===")
        
        article_results = [
            {
                "text": "ARTICLE 40: Le Président de la République peut, s'il le juge nécessaire, organiser un référendum sur toute question considérée comme d'importance nationale.",
                "score": 0.75,
                "metadata": {
                    "chunk_id": "article40-constitution",
                    "document_id": "constitution-cameroun",
                    "filename": "Constitution.pdf",
                    "page_number": 12
                }
            },
            {
                "text": "ARTICLE 41: Le Président de la République promulgue les lois adoptées par le Parlement dans un délai de quinze (15) jours à compter de leur transmission, s'il ne formule aucune demande de seconde lecture ou ne saisit le Conseil Constitutionnel.",
                "score": 0.70,
                "metadata": {
                    "chunk_id": "article41-constitution",
                    "document_id": "constitution-cameroun",
                    "filename": "Constitution.pdf",
                    "page_number": 12
                }
            }
        ]
        
        reranked_articles = rerank_service.rerank(article_query, article_results)
        for i, r in enumerate(reranked_articles, 1):
            print(f"{i}. [{r['rerank_score']:.4f}] {r['text']}")
            
    except Exception as e:
        print(f"Erreur lors du test spécifique aux articles: {e}")

    # Statistiques sur le cache
    if rerank_service.score_cache:
        print(f"\nNombre d'entrées en cache LLM: {len(rerank_service.score_cache)}")