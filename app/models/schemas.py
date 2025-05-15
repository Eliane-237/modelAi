from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union

# Modèles pour le service d'embedding
class TextsRequest(BaseModel):
    texts: List[str] = Field(..., description="Textes à transformer en embeddings")

class EmbeddingsResponse(BaseModel):
    embeddings: List[List[float]] = Field(..., description="Embeddings générés")

# Modèles pour la recherche
class SearchRequest(BaseModel):
    query: str = Field(..., description="Requête de l'utilisateur", min_length=1)
    top_k: Optional[int] = Field(10, description="Nombre de résultats à retourner")
    use_rerank: Optional[bool] = Field(True, description="Utiliser le reranking")
    use_llm_rerank: Optional[bool] = Field(False, description="Utiliser le LLM pour le reranking (plus précis mais plus lent)")
    filter: Optional[Dict[str, Any]] = Field(None, description="Filtres à appliquer (ex: {\"filename\": \"document.pdf\"})")

class DocumentMetadata(BaseModel):
    document_id: str = ""
    chunk_id: str = ""
    filename: str = "Document inconnu"
    page_number: int = 0
    extraction_method: Optional[str] = None
    section_type: Optional[str] = None
    section_number: Optional[str] = None
    section_title: Optional[str] = None

class SourceDocument(BaseModel):
    text: str
    score: float
    metadata: DocumentMetadata
    score_details: Optional[Dict[str, float]] = None

class SearchResult(BaseModel):
    text: str
    score: float
    metadata: DocumentMetadata
    original_score: Optional[float] = None
    rerank_score: Optional[float] = None
    score_details: Optional[Dict[str, float]] = None
    matched_query: Optional[str] = None

class SearchResponse(BaseModel):
    results: List[SearchResult]
    query: str
    total_results: int
    search_time: float
    metadata: Dict[str, Any] = {}

# Modèles pour le service LLM
class LLMRequest(BaseModel):
    prompt: str = Field(..., description="Prompt pour le LLM")
    max_length: Optional[int] 
    temperature: Optional[float] = Field(0.7, description="Température pour le sampling")
    stream: Optional[bool] = Field(False, description="Activer le streaming de la réponse")

class LLMResponse(BaseModel):
    response: str = Field(..., description="Réponse générée par le LLM")
    metadata: Optional[Dict[str, Any]] = None

# Modèles pour la conversation avec LLM
# Modèle pour la requête de conversation
class ConversationRequest(BaseModel):
    """
    Modèle pour une requête de conversation avec historique.
    """
    message: str = Field(..., description="Message de l'utilisateur")
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        default_factory=list, 
        description="Historique des conversations précédentes"
    )

    class Config:
        # Configuration supplémentaire si nécessaire
        arbitrary_types_allowed = True
        json_schema_extra = {
            "example": {
                "message": "Pouvez-vous m'expliquer l'article 15 de la constitution ?",
                "conversation_history": [
                    {"query": "Qu'est-ce que la constitution ?", "response": "La constitution est..."}
                ]
            }
        }

class ConversationResponse(BaseModel):
    """
    Modèle pour la réponse d'une conversation.
    """
    query: str = Field(..., description="La requête originale de l'utilisateur")
    answer: str = Field(..., description="La réponse générée")
    explanation: Optional[str] = Field(None, description="Explication de l'origine des informations")
    source_documents: List[Dict] = Field(default_factory=list, description="Documents sources")
    success: bool = Field(..., description="Indicateur de succès de la requête")
    stats: Dict = Field(default_factory=dict, description="Statistiques supplémentaires")
    
# Modèles pour le service RAG
class RAGRequest(BaseModel):
    query: str = Field(..., description="Question de l'utilisateur")
    use_expansion: Optional[bool] = Field(True, description="Utiliser l'expansion de requête")
    use_reranking: Optional[bool] = Field(True, description="Utiliser le reranking")
    max_results: Optional[int] = Field(5, description="Nombre maximum de sources à inclure")

class RAGResponse(BaseModel):
    query: str
    answer: str
    source_documents: List[SourceDocument] = []
    stats: Dict[str, Any] = {}
    success: bool = True
    error: Optional[str] = None

# Modèles pour le service de documents
class ProcessPDFResponse(BaseModel):
    status: str
    document_id: Optional[str] = None
    filename: Optional[str] = None
    chunks_processed: int = 0
    success: bool
    error: Optional[str] = None
    message: str

class RAGRequest(BaseModel):
    query: str
    use_expansion: bool = True
    use_reranking: bool = True
    max_results: int = 5

class DocumentsStatsResponse(BaseModel):
    total_documents: int
    total_chunks: int
    scanned_documents: int
    documents_list: List[Dict[str, Any]]

class ChatRequest(BaseModel):
    """Modèle de requête pour les endpoints de chat."""
    query: str = Field(..., description="Requête ou question de l'utilisateur")
    session_id: Optional[int] = Field(None, description="ID de session pour maintenir la conversation")
    streaming: bool = Field(False, description="Activer le streaming de la réponse")
    max_length: Optional[int] = Field(None, description="Limite optionnelle de longueur (aucune limite si None)")
    user_id: Optional[str] = Field(None, description="Identifiant optionnel de l'utilisateur")



# Tester avec un exemple proche de votre erreur
test_data = {
    "query": "Test query",
    "answer": "Test response",
    "source_documents": [
        {
            "text": "Sample text",
            "score": 0.8,
            "metadata": {
                "document_id": "doc123",
                "chunk_id": "chunk456",
                "filename": "test.pdf",
                "page_number": 1
            },
            "score_details": {"bm25": 0.9, "tfidf": 0.8, "original": 0.7, "legal": 0.6, "article": 0.0, "llm": 0.0}
        }
    ],
    "stats": {},
    "success": True
}

try:
    response = RAGResponse(**test_data)
    print("Validation successful!")
    print(response.model_dump())
except Exception as e:
    print(f"Validation error: {e}")