"""
Service LangChain optimisé pour LexCam, assistant juridique camerounais.
Implémente un Agentic RAG conversationnel, léger, avec analyse d'intention robuste,
outils juridiques spécialisés, mémoire conversationnelle, et conformité RGPD.
Utilise bge-m3 pour les embeddings, gère le bilinguisme français/anglais.
"""

import re
import logging
import time
import json
import os
from typing import List, Dict, Any, Optional, Union
from langdetect import detect

# Imports LangChain
from langchain.llms.base import LLM
from langchain.agents import initialize_agent, AgentType
from langchain_core.messages import HumanMessage, AIMessage
from langchain.schema import BaseMemory
from pydantic import Field, BaseModel

# Imports pour vos services
from app.services.embedding_service import EmbeddingService
from app.services.milvus_service import MilvusService
from app.services.llm_service import LlmService
from app.services.rerank_service import RerankService
from app.services.search_service import SearchService

# Configuration du logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Cache pour les embeddings
EMBEDDING_CACHE = {}

class CustomLLM(LLM):
    """Adaptateur pour LlmService compatible avec LangChain."""
    
    # Déclarer explicitement les champs
    llm_service: Any = Field(default=None)
    
    def __init__(self, llm_service, **kwargs):
        # Initialiser la classe parent d'abord
        super().__init__(**kwargs)
        # Puis assigner notre service
        self.llm_service = llm_service
        logger.info("CustomLLM initialisé avec succès")
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"service_type": "custom_llm_service"}
    
    @property
    def _llm_type(self) -> str:
        return "custom_llm_service"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        try:
            if self.llm_service is None:
                return "Service LLM non disponible"
                
            response = self.llm_service.generate_response(prompt, max_length=1000)
            if isinstance(response, dict):
                text = response.get("generated_text", response.get("response", ""))
                if not text and "choices" in response and response["choices"]:
                    text = response["choices"][0].get("text", "")
            else:
                text = str(response)
            return text
        except Exception as e:
            logger.error(f"Erreur lors de la génération avec LlmService: {e}")
            return "Erreur lors de la génération de la réponse."

class DomainAwareMemory(BaseModel):
    """Mémoire conversationnelle avec persistance et suivi des domaines juridiques."""
    
    # Déclaration des champs avec Pydantic
    save_dir: str
    session_id: int
    messages: List[Dict[str, Any]] = []
    legal_contexts: Dict[str, set] = {}
    domains: set = set()
    
    class Config:
        # Permettre les types arbitraires comme set
        arbitrary_types_allowed = True
    
    def __init__(self, save_dir: str, session_id: Optional[int] = None, **kwargs):
        session_id = session_id or int(time.time())
        super().__init__(
            save_dir=save_dir,
            session_id=session_id,
            messages=[],
            legal_contexts={},
            domains=set(),
            **kwargs
        )
        os.makedirs(save_dir, exist_ok=True)
        if session_id:
            self.load_session(session_id)

    def add_user_message(self, message: str, anonymized: bool = True):
        if anonymized:
            message = self._anonymize_message(message)
        self.messages.append({"role": "user", "content": message, "timestamp": time.time()})
        self._save_session()

    def add_ai_message(self, message: str, legal_context: Optional[Dict] = None, domains: Optional[List] = None):
        self.messages.append({"role": "assistant", "content": message, "timestamp": time.time()})
        if legal_context:
            for domain, refs in legal_context.items():
                if domain not in self.legal_contexts:
                    self.legal_contexts[domain] = set()
                self.legal_contexts[domain].update(refs)
        if domains:
            self.domains.update(domains)
        self._save_session()

    def _anonymize_message(self, message: str) -> str:
        message = re.sub(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', '[ANONYMIZED_NAME]', message)
        message = re.sub(r'\b\d{9,}\b', '[ANONYMIZED_NUMBER]', message)
        return message

    def detect_legal_domains(self, query: str) -> List[str]:
        query_lower = query.lower()
        domain_keywords = {
            "fiscal": ["impôt", "taxe", "tva", "irpp", "cgi", "fiscal", "finance"],
            "travail": ["travail", "salarié", "contrat", "licenciement", "congé", "embauche"],
            "ohada": ["ohada", "société", "commerce", "acte uniforme", "entreprise"],
            "administratif": ["fonction publique", "administration", "fonctionnaire", "décret"],
            "civil": ["mariage", "divorce", "succession", "héritage", "filiation", "propriété"],
            "pénal": ["infraction", "peine", "prison", "amende", "délit", "crime"]
        }
        domains = [domain for domain, keywords in domain_keywords.items() if any(kw in query_lower for kw in keywords)]
        return domains or ["général"]

    def get_conversation_history(self, max_messages: int = 5) -> List[Dict]:
        return self.messages[-max_messages:]

    def _save_session(self):
        try:
            session_file = os.path.join(self.save_dir, f"session_{self.session_id}.json")
            session_data = {
                "session_id": self.session_id,
                "messages": self.messages,
                "legal_contexts": {k: list(v) for k, v in self.legal_contexts.items()},
                "domains": list(self.domains),
                "last_updated": time.time()
            }
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde: {e}")

    def load_session(self, session_id: int) -> bool:
        try:
            session_file = os.path.join(self.save_dir, f"session_{session_id}.json")
            if not os.path.exists(session_file):
                return False
            with open(session_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.session_id = data.get("session_id")
            self.messages = data.get("messages", [])
            self.legal_contexts = {k: set(v) for k, v in data.get("legal_contexts", {}).items()}
            self.domains = set(data.get("domains", []))
            return True
        except Exception as e:
            logger.error(f"Erreur lors du chargement: {e}")
            return False

    def list_available_sessions(self) -> List[Dict]:
        try:
            sessions = []
            for filename in os.listdir(self.save_dir):
                if filename.startswith("session_") and filename.endswith(".json"):
                    session_id = int(filename.replace("session_", "").replace(".json", ""))
                    with open(os.path.join(self.save_dir, filename), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    first_query = next((msg["content"] for msg in data.get("messages", []) if msg["role"] == "user"), "")
                    sessions.append({
                        "session_id": session_id,
                        "first_query": first_query,
                        "last_updated": data.get("last_updated", 0),
                        "interactions": len([m for m in data.get("messages", []) if m["role"] == "user"])
                    })
            sessions.sort(key=lambda x: x["last_updated"], reverse=True)
            return sessions
        except Exception as e:
            logger.error(f"Erreur lors de la liste des sessions: {e}")
            return []

    def get_messages_for_llama(self, max_messages: int = 5) -> list:
        """
        Retourne l'historique au format de messages Llama 3.2.
        """
        if not self.messages:
            return []
        
        # Récupérer les derniers messages sans dépasser max_messages
        recent_messages = self.messages[-max_messages*2:] if len(self.messages) > max_messages*2 else self.messages
        
        formatted_messages = []
        for msg in recent_messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role in ["user", "assistant"]:
                formatted_messages.append({"role": role, "content": content})
        
        return formatted_messages

class CustomConversationMemory(BaseMemory):
    """Custom memory adapter to integrate DomainAwareMemory with LangChain."""
    
    domain_memory: DomainAwareMemory = Field(description="DomainAwareMemory instance")
    memory_key: str = Field(default="chat_history")
    
    def __init__(self, domain_memory: DomainAwareMemory, **kwargs):
        super().__init__(domain_memory=domain_memory, **kwargs)
    
    @property
    def memory_variables(self) -> List[str]:
        """Return the memory variables managed by this class."""
        return [self.memory_key]
    
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        history = self.domain_memory.get_conversation_history(max_messages=5)
        messages = []
        for msg in history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        return {self.memory_key: messages}
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        input_str = inputs.get("input", "")
        output_str = outputs.get("output", "")
        if input_str:
            self.domain_memory.add_user_message(input_str)
        if output_str:
            self.domain_memory.add_ai_message(output_str)
    
    def clear(self) -> None:
        self.domain_memory = DomainAwareMemory(self.domain_memory.save_dir)

class LangChainService:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        milvus_service: MilvusService,
        llm_service: LlmService,
        rerank_service: Optional[RerankService] = None,
        search_service: Optional[SearchService] = None,
        data_path: str = "/home/mea/Documents/modelAi/data",
        metadata_path: str = "/home/mea/Documents/modelAi/metadata",
        save_dir: Optional[str] = None
    ):
        logger.info("Initialisation de LangChainService...")
        if not llm_service or not embedding_service or not milvus_service:
            raise ValueError("Services essentiels manquants")
        
        self.data_path = data_path
        self.metadata_path = metadata_path
        self.save_dir = save_dir or os.path.join(metadata_path, "chat_history")
        os.makedirs(self.save_dir, exist_ok=True)

        self.embedding_service = embedding_service
        self.milvus_service = milvus_service
        self.llm_service = llm_service
        self.rerank_service = rerank_service
        self.search_service = search_service or SearchService(
            milvus_service=milvus_service,
            embedding_service=embedding_service
        )

        self.session_id = int(time.time())
        self.memory = DomainAwareMemory(self.save_dir)
        self.tools = self._create_service_tools()
        self.agent = self._create_agent()
        logger.info("✅ Service LangChain initialisé")

    def _create_service_tools(self) -> List:
        from langchain.tools import Tool
        tools = [
            Tool(
                name="search_article",
                func=self._tool_search_article,
                description="Recherche un article de loi spécifique."
            ),
            Tool(
                name="check_updates",
                func=self._tool_check_updates,
                description="Vérifie les mises à jour d'une loi."
            ),
            Tool(
                name="explain_legal_term",
                func=self._tool_explain_legal_term,
                description="Explique un terme juridique."
            ),
            Tool(
                name="find_procedure",
                func=self._tool_find_procedure,
                description="Recherche une procédure administrative."
            )
        ]
        logger.info(f"Outils créés: {[tool.name for tool in tools]}")
        return tools

    def _tool_search_article(self, article_reference: str) -> str:
        try:
            results = self.search_service.search(f"article {article_reference}", top_k=3)
            if not results:
                return f"Aucun article trouvé pour '{article_reference}'."
            if self.rerank_service:
                results = self.rerank_service.rerank(f"article {article_reference}", results)
            article_text = results[0].get("text", "")
            metadata = results[0].get("metadata", {})
            source = metadata.get("filename", "Source inconnue")
            page = metadata.get("page_number", "?")
            return f"Article {article_reference} (source: {source}, page {page}):\n{article_text}"
        except Exception as e:
            logger.error(f"Erreur dans search_article: {e}")
            return f"Erreur lors de la recherche de l'article {article_reference}."

    def _tool_check_updates(self, law_reference: str) -> str:
        try:
            results = self.search_service.search(f"modification {law_reference} récent", top_k=3)
            if not results:
                return f"Aucune mise à jour trouvée pour '{law_reference}'."
            updates = []
            for result in results:
                text = result.get("text", "").lower()
                if "modif" in text or "amend" in text or "mise à jour" in text:
                    metadata = result.get("metadata", {})
                    source = metadata.get("filename", "Source inconnue")
                    updates.append(f"Source: {source}\n{result.get('text', '')}")
            if not updates:
                return f"Aucune mise à jour spécifique trouvée pour '{law_reference}'."
            return "Mises à jour trouvées:\n\n" + "\n\n---\n\n".join(updates[:2])
        except Exception as e:
            logger.error(f"Erreur dans check_updates: {e}")
            return f"Erreur lors de la recherche des mises à jour pour {law_reference}."

    def _tool_explain_legal_term(self, term: str) -> str:
        try:
            results = self.search_service.search(f"définition {term} juridique cameroun", top_k=3)
            if not results:
                return f"Aucune définition trouvée pour '{term}'."
            if self.rerank_service:
                results = self.rerank_service.rerank(f"définition {term}", results)
            definition = results[0].get("text", "")
            metadata = results[0].get("metadata", {})
            source = metadata.get("filename", "Source inconnue")
            return f"Définition de '{term}' (source: {source}):\n{definition}"
        except Exception as e:
            logger.error(f"Erreur dans explain_legal_term: {e}")
            return f"Erreur lors de la recherche de la définition de {term}."

    def _tool_find_procedure(self, procedure_name: str) -> str:
        try:
            results = self.search_service.search(f"procédure {procedure_name} cameroun étapes", top_k=3)
            if not results:
                return f"Aucune information sur la procédure '{procedure_name}'."
            if self.rerank_service:
                results = self.rerank_service.rerank(f"procédure {procedure_name}", results)
            procedure_text = results[0].get("text", "")
            metadata = results[0].get("metadata", {})
            source = metadata.get("filename", "Source inconnue")
            return f"Procédure '{procedure_name}' (source: {source}):\n{procedure_text}"
        except Exception as e:
            logger.error(f"Erreur dans find_procedure: {e}")
            return f"Erreur lors de la recherche de la procédure {procedure_name}."

    def _create_agent(self):
        """Crée un agent conversationnel léger basé sur LangChain."""
        try:
            llm = CustomLLM(llm_service=self.llm_service)
            
            # Configurer la mémoire conversationnelle
            memory = CustomConversationMemory(domain_memory=self.memory)

            # Prompt système conversationnel
            system_prompt = """
Vous êtes LexCam, un assistant expert sur les documents administratifs camerounais. Répondez de manière formelle, précise et engageante, en vous appuyant sur les documents juridiques fournis par la base vectorielle. Intégrez des exemples locaux camerounais (ex. : pratiques à Douala, Yaoundé, ou contextes ruraux comme Bamenda) pour illustrer vos réponses.

## Directives

- **Conversationnalité** : Adoptez un ton professionnel mais accessible, comme si vous expliquiez à un client camerounais.
- **Contexte local** : Incluez des exemples pertinents (ex. : application d'une loi dans une PME à Douala).
- **Bilinguisme** : Répondez en français, sauf si la requête est en anglais. Traduisez les termes juridiques si nécessaire.
- **Clarification** : Si la requête est ambiguë ou le contexte insuffisant, demandez poliment des précisions.
- **Suggestions** : Terminez par une suggestion contextuelle (ex. : "Souhaitez-vous des détails sur les sanctions associées ?").
- **Conformité RGPD** : Les données sensibles sont anonymisées par DomainAwareMemory.

COMPORTEMENT :
- Adoptez un ton conversationnel et engageant, sans formalités inutiles, mais restez précis et professionnel.
- Tenez compte de l’historique de la conversation. Si l’utilisateur a déjà posé une question, répondez directement sans demander “Quelle est votre question ?” et faites un lien naturel avec les échanges précédents (ex. “Vous avez parlé de la constitution tout à l’heure, voici un résumé…”).
- Structurez vos réponses en paragraphes courts ou avec des puces pour que ce soit clair et facile à lire.
- Adaptez vos explications et résumés au niveau de l’utilisateur : simplifiez pour les débutants, utilisez des termes techniques pour les experts, en devinant leur niveau à partir de leurs questions.
- Basez-vous UNIQUEMENT sur les documents juridiques fournis. Citez toujours la source exacte quand tu te sers de la base vectorielle pour répondre ou une source quelconque (nom du document, article, section ou page) pour les explications et les résumés.
- Si une information n’est pas dans les documents, dites-le honnêtement (ex. “Désolé, je n’ai pas assez d’infos dans mes sources pour résumer ce sujet, mais je peux aider avec autre chose.”).
- Si un terme juridique est complexe, expliquez-le brièvement en langage courant pour le rendre accessible.
- Si l’utilisateur semble inquiet ou utilise des mots comme “stressé” ou “urgent”, montrez de l’empathie (ex. “Je vois que c’est préoccupant, on va clarifier ça ensemble.”).
- Si la question est vague, demandez une précision de manière amicale (ex. “Pour bien vous aider, vous parlez de quel aspect du droit ?”).
- Répondez aux salutations avec un accueil chaleureux.

INSTRUCTIONS SPÉCIFIQUES :
- Pour les résumés, incluez 3 à 5 points clés maximum, en évitant les détails inutiles. Assurez-vous que le résumé est autonome mais invite à poser des questions pour approfondir.
- Utilisez la langue de l’utilisateur (français par défaut, anglais si détecté).
- Restez neutre et objectif, mais ajoutez une touche de chaleur pour rendre l’échange agréable.
- Si c’est la première question de la session, accueillez l’utilisateur avec enthousiasme. Dans une conversation en cours, concentrez-vous sur la continuité et la pertinence.
- Évitez les réponses génériques ou hors sujet. Assurez-vous que vos réponses et résumés s’appuient sur le contexte de la question et de l’historique.

## Contexte juridique
{context}

## Historique de la conversation
{chat_history}
"""

            # Configurer l'agent
            agent_kwargs = {
                "prefix": system_prompt,
                "format_instructions": """
Pour utiliser un outil, utilisez le format :
Action: nom_de_l_outil
Action Input: paramètre_pour_l_outil

Après avoir utilisé un outil, analysez le résultat et décidez si :
1. Un autre outil est nécessaire.
2. Une réponse finale peut être fournie.

Réponse finale :
Final Answer: [votre réponse formelle avec exemple local si pertinent]
""",
                "suffix": "Question: {input}\n{agent_scratchpad}"
            }

            # Initialiser l'agent conversationnel
            agent = initialize_agent(
                tools=self.tools,
                llm=llm,
                agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                memory=memory,
                verbose=True,
                agent_kwargs=agent_kwargs,
                handle_parsing_errors=True
            )

            logger.info("✅ Agent conversationnel créé")
            return agent

        except Exception as e:
            logger.error(f"❌ Erreur lors de la création de l'agent: {e}")
            raise

    def _identify_query_intent(self, query: str) -> Dict:
        intent = {
            "domains": self.memory.detect_legal_domains(query),
            "intent": "information",
            "keywords": []
        }
        query_lower = query.lower()
        if re.search(r'article\s+\d+', query_lower) or "texte" in query_lower or "loi" in query_lower:
            intent["intent"] = "article"
        elif "définition" in query_lower or "signifie" in query_lower or "qu'est-ce" in query_lower:
            intent["intent"] = "definition"
        elif "procédure" in query_lower or "comment" in query_lower or "étapes" in query_lower:
            intent["intent"] = "procedure"
        elif "mise à jour" in query_lower or "modif" in query_lower or "récent" in query_lower:
            intent["intent"] = "update"
        intent["keywords"] = [word for word in re.findall(r'\b\w{4,}\b', query_lower) 
                             if word not in ["pour", "avec", "dans", "comment", "quels", "quelles"]]
        return intent

    def _format_search_results(self, results: List[Dict]) -> str:
        """Formate les résultats de recherche en un contexte structuré pour le LLM."""
        if not results:
            return "Aucun document pertinent trouvé dans la base de connaissances."
        
        context_parts = []
        
        # Organiser par document pour une meilleure lisibilité
        documents = {}
        for i, result in enumerate(results, 1):
            doc_id = result.get("metadata", {}).get("document_id", "unknown")
            if doc_id not in documents:
                documents[doc_id] = []
            documents[doc_id].append(result)
        
        # Formater chaque document et ses extraits
        for doc_id, items in documents.items():
            # Utiliser le premier item pour les informations du document
            first_item = items[0]
            metadata = first_item.get("metadata", {})
            doc_name = metadata.get("filename", "Document inconnu")
            
            # Ajouter l'en-tête du document
            context_parts.append(f"## {doc_name} (ID: {doc_id[:8]}...)")
            
            # Ajouter chaque extrait avec sa page
            for item in items:
                text = item.get("text", "").strip()
                page = item.get("metadata", {}).get("page_number", "?")
                score = item.get("score", 0.0)
                
                # Formater l'extrait
                context_parts.append(f"[Page {page}] {text}")
            
            # Ajouter un séparateur entre les documents
            context_parts.append("---")
        
        return "\n\n".join(context_parts)

    def _extract_legal_context_from_results(self, results: List[Dict]) -> Dict:
        """Extrait le contexte juridique (références aux articles, lois, etc.) des résultats."""
        legal_context = {}
        
        for result in results:
            text = result.get("text", "")
            metadata = result.get("metadata", {})
            
            # Détecter les domaines juridiques
            domains = self.memory.detect_legal_domains(text)
            
            # Formater une citation
            doc_name = metadata.get("filename", "Document inconnu")
            page = metadata.get("page_number", "?")
            citation = f"{doc_name} (p.{page})"
            
            # Détecter les références à des articles
            article_matches = re.findall(r'article\s+(\d+[a-z]*)', text.lower())
            if article_matches:
                for article in article_matches:
                    for domain in domains:
                        key = f"{domain}_articles"
                        if key not in legal_context:
                            legal_context[key] = []
                        legal_context[key].append(f"Article {article} ({citation})")
            
            # Ajouter des références générales par domaine
            for domain in domains:
                if domain not in legal_context:
                    legal_context[domain] = []
                legal_context[domain].append(citation)
        
        return legal_context

    def debug_source_metadata(self, results: List[Dict]) -> None:
        """
        Méthode de debugging pour analyser les métadonnées des sources.
        À utiliser temporairement pour comprendre la structure des données.
        """
        logger.info("🔍 === DEBUG DES MÉTADONNÉES SOURCES ===")
        
        for i, result in enumerate(results[:3]):  # Analyser les 3 premiers résultats
            logger.info(f"📄 Résultat #{i+1}:")
            logger.info(f"   Type: {type(result)}")
            logger.info(f"   Clés disponibles: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
            
            if isinstance(result, dict):
                # Analyser les métadonnées
                metadata = result.get("metadata", {})
                logger.info(f"   Métadonnées type: {type(metadata)}")
                logger.info(f"   Métadonnées clés: {list(metadata.keys()) if isinstance(metadata, dict) else 'N/A'}")
                
                # Afficher les valeurs importantes
                if isinstance(metadata, dict):
                    for key in ["filename", "source", "document_id", "page_number", "page", "path"]:
                        if key in metadata:
                            logger.info(f"   {key}: {metadata[key]} (type: {type(metadata[key])})")
                
                # Analyser le score
                score = result.get("score")
                logger.info(f"   Score: {score} (type: {type(score)})")
                
                # Analyser le texte
                text = result.get("text", "")
                logger.info(f"   Texte: {len(text)} caractères")
        
        logger.info("🔍 === FIN DEBUG ===")
    def _format_source_documents(self, results: List[Dict]) -> List[Dict]:
        """
        Formate les documents sources pour l'interface utilisateur avec des métadonnées améliorées.
        S'assure que chaque document a un nom de fichier et un numéro de page significatifs.
        
        Args:
            results: Liste des résultats de recherche
            
        Returns:
            Liste formatée de documents pour l'affichage
        """
        formatted_docs = []
        
        for result in results:
            # Extraire les métadonnées de différentes sources possibles
            metadata = result.get("metadata", {})
            score = result.get("score", 0.0)
            
            # Debug : afficher les métadonnées reçues
            logger.debug(f"📊 Métadonnées reçues: {metadata}")
            
            # Extraction robuste du nom du fichier
            filename = "Document juridique"
            
            # Essayer plusieurs champs pour le nom du fichier
            possible_filename_fields = [
                "filename", "source", "file_name", "document_name", 
                "path", "file_path", "title", "name"
            ]
            
            for field in possible_filename_fields:
                if field in metadata and metadata[field]:
                    raw_filename = metadata[field]
                    
                    # Nettoyer le chemin si c'est un chemin complet
                    if isinstance(raw_filename, str):
                        # Enlever les chemins Unix/Windows
                        if "/" in raw_filename:
                            filename = raw_filename.split("/")[-1]
                        elif "\\" in raw_filename:
                            filename = raw_filename.split("\\")[-1]
                        else:
                            filename = raw_filename
                        
                        # Enlever les extensions inutiles
                        if filename.endswith(('.pdf', '.PDF')):
                            filename = filename[:-4]
                        
                        # Limiter la longueur
                        if len(filename) > 50:
                            filename = filename[:47] + "..."
                        
                        break
            
            # Si toujours pas de nom valide, utiliser l'ID du document
            if filename == "Document juridique" and metadata.get("document_id"):
                doc_id = metadata["document_id"]
                filename = f"Document_{doc_id[:12]}"
            
            # Extraction robuste du numéro de page
            page_number = 1
            possible_page_fields = [
                "page_number", "page", "page_num", "page_index", "numero_page"
            ]
            
            for field in possible_page_fields:
                if field in metadata and metadata[field]:
                    try:
                        page_number = int(metadata[field])
                        break
                    except (ValueError, TypeError):
                        continue
            
            # Extraire d'autres métadonnées utiles
            section_info = {}
            if metadata.get("section_type"):
                section_info["type"] = metadata["section_type"]
            if metadata.get("section_number"):
                section_info["number"] = metadata["section_number"]
            if metadata.get("section_title"):
                section_info["title"] = metadata["section_title"]
            
            # Créer un objet document formaté avec toutes les métadonnées
            formatted_doc = {
                "text": result.get("text", "")[:2000],  # Limiter la longueur du texte
                "score": score,
                "metadata": {
                    "document_id": metadata.get("document_id", ""),
                    "filename": filename,  # ← NOM PROPRE GARANTI
                    "page_number": page_number,  # ← PAGE VALIDE GARANTIE
                    "extraction_method": metadata.get("extraction_method", ""),
                    "chunk_id": metadata.get("chunk_id", ""),
                    "source": metadata.get("source", ""),
                    # Préserver les métadonnées originales pour le debug
                    "original_metadata": metadata
                }
            }
            
            # Ajouter les informations de section si disponibles
            if section_info:
                formatted_doc["metadata"].update(section_info)
            
            # Créer une description lisible de la source
            source_description = filename
            if section_info.get("type") and section_info.get("number"):
                source_description += f" ({section_info['type']} {section_info['number']})"
            elif section_info.get("title"):
                source_description += f" ({section_info['title']})"
                
            formatted_doc["source"] = source_description
            
            # Log pour debug
            logger.info(f"✅ Document formaté: {filename} (page {page_number})")
            
            formatted_docs.append(formatted_doc)
        
        logger.info(f"📚 {len(formatted_docs)} documents formatés avec succès")
        return formatted_docs
    
    def _format_messages_as_prompt(self, messages: List[Dict]) -> str:
        """Formate les messages pour créer un prompt conversationnel."""
        prompt_parts = []
        
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "user":
                prompt_parts.append(f"<|user|>\n{content}\n")
            elif role == "assistant":
                prompt_parts.append(f"<|assistant|>\n{content}\n")
            elif role == "system":
                prompt_parts.append(f"<|system|>\n{content}\n")
        
        return "".join(prompt_parts)
    
    def _process_contextual_references(self, query: str, conversation_history: str) -> str:
        """Identifie et résout les références contextuelles dans la requête."""
        # Détecter les références comme "cette loi", "l'article mentionné", etc.
        contextual_references = [
            "cet article", "cette loi", "ce document", "mentionné précédemment",
            "comme indiqué", "comme mentionné", "ci-dessus", "précédemment"
        ]
        
        # Si la requête contient une référence contextuelle, ajouter une instruction
        if any(ref in query.lower() for ref in contextual_references):
            instruction = """Remarque: La question fait référence à des éléments mentionnés précédemment dans la conversation. 
            Assurez-vous de prendre en compte tout le contexte de la conversation pour y répondre."""
            return instruction
        
        return ""

    def generate_response(self, query: str, streaming: bool = False, session_id=None) -> Dict:
        """
        Génère une réponse à une question juridique en utilisant la recherche vectorielle et le LLM.
        
        Args:
            query: Question juridique de l'utilisateur
            streaming: Indique si la réponse doit être générée en mode streaming
            session_id: Identifiant de session optionnel pour la conversation
            
        Returns:
            Dictionnaire contenant la réponse et les métadonnées associées
        """
        try:
            # Normalisation et validation de la requête
            if isinstance(query, dict):
                query = query.get('question', query.get('query', str(query)))
            elif not isinstance(query, str):
                query = str(query)
                
            if not query.strip():
                return {
                    "response": "Veuillez poser une question valide.",
                    "error": "Invalid query",
                    "success": False,
                    "session_id": self.session_id
                }
            
            # Gérer la session
            if session_id:
                self.get_or_create_session(session_id)
            else:
                session_id = self.session_id
            
            # Détection de la langue
            try:
                detected_lang = detect(query)
                language = detected_lang if detected_lang in ["fr", "en"] else "fr"
            except Exception:
                language = "fr"
            
            # Analyser l'intention pour comprendre le type de question juridique
            intent_analysis = self._identify_query_intent(query)
            domains = intent_analysis.get("domains", [])
            
            logger.info(f"Question: '{query}' - Domaines: {domains} - Langue: {language}")
            
            # Ajouter la question à l'historique (une seule fois)
            self.memory.add_user_message(query)
            
            # Effectuer la recherche avec gestion d'erreurs robuste
            try:
                # Rechercher les documents pertinents
                search_results = self.search_service.search(query, top_k=5)
                
                # Appliquer le reranking si disponible
                if self.rerank_service and search_results:
                    search_results = self.rerank_service.rerank(query, search_results, use_llm=False)
                
                # Formater le contexte pour le LLM
                context = self._format_search_results(search_results)
                logger.info(f"Recherche effectuée: {len(search_results)} résultats trouvés")
                
            except Exception as e:
                logger.error(f"Erreur lors de la recherche: {e}")
                search_results = []
                context = "Aucun document pertinent trouvé."

            # Récupérer l'historique au format des messages pour Llama 3.2
            previous_messages = self.memory.get_messages_for_llama(max_messages=5)
            
            # Vérifier si la requête contient des références contextuelles
            contextual_instruction = ""
            if hasattr(self, '_process_contextual_references'):
                conversation_history = self.memory.get_conversation_history(max_messages=5)
                contextual_instruction = self._process_contextual_references(query, conversation_history)
            
            # Construction du message système conforme au format Llama 3.2
            system_message = """Vous êtes LexCam, un assistant juridique camerounais conversationnel et précis. Répondez comme un expert juridique amical et accessible.

    COMPORTEMENT :
- Si vous connaissez le nom de l'utilisateur, commencez la première interaction de la session par un message de bienvenue personnalisé (ex. "Bienvenue, Jean ! Content de vous aider aujourd'hui."). Si le nom n'est pas disponible, utilisez un accueil chaleureux mais général (ex. "Ravi de vous aider aujourd'hui !").
- Si l'utilisateur mentionne un article ou une loi spécifique, citez son texte exact, puis expliquez-le en termes simples, comme si vous l'expliquiez à quelqu'un qui découvre le sujet.
- Si l'utilisateur demande un résumé sur un sujet juridique (ex. "Résumez le droit des contrats"), fournissez un aperçu concis et clair du sujet, basé uniquement sur les documents fournis. Structurez le résumé en points clés, adaptés au niveau d'expertise de l'utilisateur, et mentionnez les sources utilisées.
- Adoptez un ton conversationnel et engageant, sans formalités inutiles, mais restez précis et professionnel.
- Tenez compte de l'historique de la conversation. Si l'utilisateur a déjà posé une question, répondez directement sans demander "Quelle est votre question ?" et faites un lien naturel avec les échanges précédents (ex. "Vous avez parlé de la constitution tout à l'heure, voici un résumé…").
- Structurez vos réponses en paragraphes courts ou avec des puces pour que ce soit clair et facile à lire.
- Adaptez vos explications et résumés au niveau de l'utilisateur : simplifiez pour les débutants, utilisez des termes techniques pour les experts, en devinant leur niveau à partir de leurs questions.
- Basez-vous UNIQUEMENT sur les documents juridiques fournis. Citez toujours la source exacte (nom du document, article, section ou page) pour les explications et les résumés.
- Si une information n'est pas dans les documents, dites-le honnêtement (ex. "Désolé, je n'ai pas assez d'infos dans mes sources pour résumer ce sujet, mais je peux aider avec autre chose.").
- Si un terme juridique est complexe, expliquez-le brièvement en langage courant pour le rendre accessible.
- Proposez 1 ou 2 questions de suivi pertinentes, mais seulement si c'est la première question de la session ou si l'utilisateur semble vouloir explorer davantage. Évitez les suggestions inutiles dans une conversation avancée.
- Si l'utilisateur semble inquiet ou utilise des mots comme "stressé" ou "urgent", montrez de l'empathie (ex. "Je vois que c'est préoccupant, on va clarifier ça ensemble.").
- Si la question est vague, demandez une précision de manière amicale (ex. "Pour bien vous aider, vous parlez de quel aspect du droit ?").
- Répondez aux salutations (ex. "Bonjour", "Salut") avec un accueil chaleureux mais unique, sans répéter leur salutation (ex. "Content de vous aider aujourd'hui !").

INSTRUCTIONS SPÉCIFIQUES :
- Pour les résumés, incluez 3 à 5 points clés maximum, en évitant les détails inutiles. Assurez-vous que le résumé est autonome mais invite à poser des questions pour approfondir.
- Utilisez la langue de l'utilisateur (français par défaut, anglais si détecté).
- Restez neutre et objectif, mais ajoutez une touche de chaleur pour rendre l'échange agréable.
- Si c'est la première question de la session, accueillez l'utilisateur avec enthousiasme. Dans une conversation en cours, concentrez-vous sur la continuité et la pertinence.
- Évitez les réponses génériques ou hors sujet. Assurez-vous que vos réponses et résumés s'appuient sur le contexte de la question et de l'historique.
    """
            
            if contextual_instruction:
                system_message += f"\nREMARQUE IMPORTANTE:\n{contextual_instruction}\n"
                
            system_message += f"\nINFORMATION JURIDIQUE:\n{context}"
            
            # Créer la liste complète des messages pour le format Llama 3.2
            messages = [
                {"role": "system", "content": system_message}
            ]
            
            # Ajouter les messages précédents s'ils existent
            if previous_messages:
                messages.extend(previous_messages)
            
            # Ajouter la question actuelle
            messages.append({"role": "user", "content": query})
            
            # Construire le prompt au format Llama 3.2 selon la documentation officielle
            prompt = "<|begin_of_text|>\n"
            
            for message in messages:
                role = message["role"]
                content = message["content"]
                
                if role == "system":
                    prompt += f"<|system|>\n{content}\n"
                elif role == "user":
                    prompt += f"<|user|>\n{content}\n"
                elif role == "assistant":
                    prompt += f"<|assistant|>\n{content}\n"
            
            # Ajouter la balise assistant pour la réponse à générer
            prompt += "<|assistant|>\n"
            
            # Générer la réponse avec le LLM
            logger.info("Génération de la réponse avec le LLM")
            start_time = time.time()

            if search_results:
                self.debug_source_metadata(search_results)
            
            if streaming:
                # Gérer le mode streaming si implémenté
                logger.info("🔄 Mode streaming activé")
                # Créer un générateur pour le streaming
                def response_generator():
                    try:
                        # Appeler le LLM en mode streaming
                        stream = self.llm_service.generate_response(
                            prompt=prompt, 
                            max_length=3000, 
                            stream=True  # ← IMPORTANT: Activer le streaming
                        )
                        
                        full_response = ""
                        for token in stream:
                            full_response += token
                            yield token
                        
                        # Sauvegarder à la fin
                        self.memory.add_ai_message(full_response, {}, domains)
                        
                    except Exception as e:
                        logger.error(f"Erreur streaming: {e}")
                        yield f"Erreur: {str(e)}"
                
                # Retourner avec le générateur
                return {
                    "query": query,
                    "streaming": True,
                    "response_generator": response_generator(),
                    "source_documents": self._format_source_documents(search_results) if hasattr(self, '_format_source_documents') else [],
                    "domains": domains,
                    "intent": intent_analysis.get("intent"),
                    "language": language,
                    "session_id": session_id,
                    "success": True
                }
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération de réponse: {e}")
            return {
                "query": query,
                "response": f"Une erreur s'est produite: {str(e)}",
                "source_documents": [],
                "success": False,
                "error": str(e),
                "session_id": session_id if session_id else self.session_id
            }

    def reset_conversation(self):
        self.memory = DomainAwareMemory(self.save_dir)
        self.session_id = int(time.time())

    def load_conversation_history(self, session_id: int) -> bool:
        if self.memory.load_session(session_id):
            self.session_id = session_id
            logger.info(f"Session {session_id} chargée avec succès")
            return True
        return False

    def list_available_sessions(self) -> List[Dict]:
         
        sessions = self.memory.list_available_sessions()
    
        # Normaliser les données
        normalized_sessions = []
        for session in sessions:
            normalized_session = {
                "session_id": session.get("session_id", int(time.time())),
                "first_query": session.get("first_query", ""),
                "start_time": session.get("start_time", time.time()),
                "last_time": session.get("last_updated", time.time()),
                "interactions": session.get("interactions", 0)
            }
            normalized_sessions.append(normalized_session)
        
        return normalized_sessions

    def get_session_info(self) -> Dict:
        return {
            "session_id": self.session_id,
            "message_count": len(self.memory.messages),
            "domains": list(self.memory.domains),
            "last_updated": time.time()
        }

    def get_or_create_session(self, session_id: int):
        """Charge une session ou crée une nouvelle si elle n'existe pas."""
        if not self.load_conversation_history(session_id):
            # Créer une nouvelle session avec l'ID fourni
            self.session_id = session_id
            self.memory = DomainAwareMemory(self.save_dir, session_id)

def get_langchain_service(
    embedding_service=None,
    milvus_service=None,
    llm_service=None,
    rerank_service=None,
    search_service=None,
    data_path="/home/mea/Documents/modelAi/data",
    metadata_path="/home/mea/Documents/modelAi/metadata",
    save_dir=None
) -> LangChainService:
    logger.info("Appel de get_langchain_service")
    if llm_service is None or not isinstance(llm_service, LlmService):
        logger.error(f"llm_service invalide: {type(llm_service)}")
        raise ValueError("llm_service doit être une instance de LlmService")
    return LangChainService(
        embedding_service=embedding_service,
        milvus_service=milvus_service,
        llm_service=llm_service,
        rerank_service=rerank_service,
        search_service=search_service,
        data_path=data_path,
        metadata_path=metadata_path,
        save_dir=save_dir
    )