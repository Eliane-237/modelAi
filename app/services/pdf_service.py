import os
import logging
import numpy as np
import json
from typing import List, Dict, Any, Tuple, Optional
from app.services.embedding_service import EmbeddingService
from app.services.milvus_service import MilvusService
from app.services.pdf_processor import PDFProcessor
from fastapi import FastAPI


# Configuration du logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

class PDFService:
    """
    Service principal pour la gestion des documents PDF:
    - Coordonne l'extraction et le traitement des PDFs
    - Gère la génération d'embeddings
    - Stocke les données dans Milvus
    - Fait le lien entre les différents composants du système
    """
    
    def __init__(
        self, 
        embedding_service: EmbeddingService, 
        milvus_service: MilvusService, 
        data_path: str = "/home/mea/Documents/modelAi/data",
        metadata_path: str = "/home/mea/Documents/modelAi/metadata",
        max_tokens: int = 500, 
        overlap_tokens: int = 100, 
        batch_size: int = 20,
        target_dim: Optional[int] = None,  # Dimension cible des embeddings (None = pas d'ajustement)
        force_ocr: bool = False,           # Nouveau paramètre pour forcer l'OCR sur tous les documents
        ocr_language: str = "fra"          # Langue par défaut pour l'OCR
    ):
        self.data_path = data_path
        self.metadata_path = metadata_path
        self.embedding_service = embedding_service
        self.milvus_service = milvus_service
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.batch_size = batch_size
        self.target_dim = target_dim or milvus_service.dim
        self.force_ocr = force_ocr
        self.ocr_language = ocr_language
        
        # Créer les répertoires nécessaires
        os.makedirs(self.data_path, exist_ok=True)
        os.makedirs(self.metadata_path, exist_ok=True)
        
        # Initialiser le processeur PDF avec les paramètres améliorés
        self.pdf_processor = PDFProcessor(
            metadata_path=metadata_path,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            ocr_language=ocr_language,
            force_ocr=force_ocr
        )

    def _adjust_embeddings(self, embeddings: List[List[float]]) -> List[List[float]]:
        """
        Ajuste les dimensions des embeddings pour correspondre à la dimension cible.
        Utilise des techniques appropriées selon que la dimension cible est plus grande ou plus petite.
        
        Args:
            embeddings: Liste des vecteurs d'embedding à ajuster
            
        Returns:
            Liste des vecteurs d'embedding ajustés
        """
        if not embeddings:
            return []
            
        current_dim = len(embeddings[0])
        target_dim = self.target_dim
        
        # Si les dimensions correspondent déjà, pas d'ajustement nécessaire
        if current_dim == target_dim:
            return embeddings
        
        logger.info(f"Ajustement des embeddings de {current_dim} à {target_dim} dimensions")
        
        adjusted_embeddings = []
        
        for embedding in embeddings:
            if current_dim < target_dim:
                # Augmenter la dimension (padding ou expansion)
                if target_dim % current_dim == 0:
                    # Si la cible est un multiple de la dimension actuelle, répéter
                    factor = target_dim // current_dim
                    adjusted = embedding * factor
                else:
                    # Sinon, padding avec des zéros
                    adjusted = embedding + [0.0] * (target_dim - current_dim)
            else:
                # Réduire la dimension (agrégation ou sélection)
                if current_dim % target_dim == 0:
                    # Si la dimension actuelle est un multiple de la cible, moyenner les groupes
                    factor = current_dim // target_dim
                    adjusted = [
                        sum(embedding[i*factor:(i+1)*factor]) / factor
                        for i in range(target_dim)
                    ]
                else:
                    # Sinon, sélectionner uniformément les valeurs
                    indices = np.linspace(0, current_dim-1, target_dim, dtype=int)
                    adjusted = [embedding[i] for i in indices]
            
            adjusted_embeddings.append(adjusted)
        
        return adjusted_embeddings

    def _process_batch(self, chunks: List[Dict[str, Any]]) -> bool:
        """
        Traite un lot de chunks: génère des embeddings et les stocke dans Milvus.
        
        Args:
            chunks: Liste de dictionnaires contenant les textes et métadonnées
            
        Returns:
            True si le traitement a réussi, False sinon
        """
        if not chunks:
            return True  # Rien à faire
        
        try:
            # Extraire les textes pour génération d'embeddings
            texts = [chunk["text"] for chunk in chunks]
            logger.info(f"📊 Génération d'embeddings pour {len(texts)} chunks")
            
            # Générer les embeddings
            embeddings = self.embedding_service.generate_embeddings(texts)
            
            if not embeddings:
                logger.error("❌ Aucun embedding généré")
                return False
            
            # Ajuster les dimensions si nécessaire
            if len(embeddings[0]) != self.target_dim:
                embeddings = self._adjust_embeddings(embeddings)
            
            # Préparer les métadonnées pour Milvus
            metadata_list = [chunk["metadata"] for chunk in chunks]
            
            # Insérer dans Milvus
            self.milvus_service.insert_documents_with_metadata(embeddings, texts, metadata_list)
            logger.info(f"✅ Lot traité avec succès: {len(chunks)} chunks")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du traitement du lot: {e}")
            return False
    
    def load_documents(self) -> List[str]:
        """
        Charge la liste des fichiers PDF disponibles dans le répertoire de données.
        
        Returns:
            Liste des chemins complets vers les fichiers PDF
        """
        if not os.path.exists(self.data_path):
            logger.error(f"Le répertoire spécifié n'existe pas: {self.data_path}")
            raise FileNotFoundError(f"Le répertoire spécifié n'existe pas: {self.data_path}")
            
        pdf_files = [
            os.path.join(self.data_path, f) 
            for f in os.listdir(self.data_path) 
            if f.lower().endswith('.pdf')
        ]
        
        logger.info(f"Fichiers PDF trouvés: {len(pdf_files)}")
        return pdf_files

    def process_pdf(self, pdf_path: str, force_reprocess: bool = False, detect_scanned: bool = True) -> bool:
        """
        Traite un document PDF en l'analysant, générant des chunks et des embeddings.
        
        Args:
            pdf_path: Chemin vers le fichier PDF
            force_reprocess: Forcer le retraitement même si déjà traité
            detect_scanned: Détecter automatiquement si le PDF est scanné
            
        Returns:
            True si le traitement a réussi, False sinon
        """
        try:
            # Vérifier si le document est un PDF scanné avant le traitement
            if detect_scanned:
                try:
                    import fitz
                    doc = fitz.open(pdf_path)
                    is_scanned = self.pdf_processor.detect_scanned_document(doc)
                    doc.close()
                    
                    if is_scanned:
                        logger.info(f"📑 Le document {os.path.basename(pdf_path)} est détecté comme scanné, activation de l'OCR")
                        self.pdf_processor.force_ocr = True
                    else:
                        # Rétablir la valeur par défaut si elle a été modifiée
                        self.pdf_processor.force_ocr = self.force_ocr
                except Exception as e:
                    logger.warning(f"Erreur lors de la détection de document scanné: {e}")
            
            # Traiter le PDF avec le processeur
            chunks, doc_metadata = self.pdf_processor.process_pdf(pdf_path, force_reprocess)
            
            if not chunks:
                logger.warning(f"Aucun contenu extrait du document: {pdf_path}")
                return False
            
            # Traiter les chunks par lots
            current_batch = []
            success = True
            
            for chunk in chunks:
                current_batch.append(chunk)
                
                # Traiter le lot quand il atteint la taille maximale
                if len(current_batch) >= self.batch_size:
                    batch_success = self._process_batch(current_batch)
                    success = success and batch_success
                    current_batch = []
            
            # Traiter le dernier lot s'il reste des chunks
            if current_batch:
                batch_success = self._process_batch(current_batch)
                success = success and batch_success
            
            # Mettre à jour les métadonnées avec les informations sur le traitement
            metadata_file = os.path.join(self.metadata_path, f"{doc_metadata['document_id']}.json")
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    updated_metadata = json.load(f)
                
                # Ajouter des informations sur les embeddings
                updated_metadata["embeddings_generated"] = success
                updated_metadata["embedding_model"] = "BGE-M3"
                updated_metadata["embedding_dim"] = self.target_dim
                
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(updated_metadata, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error(f"Erreur lors de la mise à jour des métadonnées: {e}")
            
            logger.info(f"Traitement terminé pour {pdf_path}: {len(chunks)} chunks traités")
            return success
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement du PDF {pdf_path}: {e}")
            return False

    def process_pdfs(self, force_reprocess: bool = False, detect_scanned: bool = True) -> Dict[str, Any]:
        """
        Traite uniquement les PDFs qui n'ont pas encore été traités,
        sauf si force_reprocess est True.
        
        Args:
            force_reprocess: Forcer le retraitement même si déjà traité
            detect_scanned: Détecter automatiquement si les PDFs sont scannés
            
        Returns:
            Statistiques de traitement
        """
        # Charger la liste des PDFs
        pdf_files = self.load_documents()
        
        if not pdf_files:
            logger.warning(f"Aucun fichier PDF trouvé dans {self.data_path}")
            return {"status": "completed", "processed_files": 0, "success": True}
        
        # Statistiques de traitement
        stats = {
            "total_files": len(pdf_files),
            "processed_files": 0,
            "skipped_files": 0,
            "successful_files": 0,
            "failed_files": 0,
            "chunks_processed": 0,
            "already_processed_files": 0,
            "scanned_documents": 0
        }
        
        # Traiter chaque PDF
        for pdf_path in pdf_files:
            try:
                # Générer l'ID du document
                document_id = self.pdf_processor.generate_document_id(pdf_path)
                
                # Vérifier si le document a déjà été traité
                metadata_file = os.path.join(self.metadata_path, f"{document_id}.json")
                
                if os.path.exists(metadata_file) and not force_reprocess:
                    # Charger les métadonnées pour vérifier si le traitement a été complet
                    try:
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            doc_metadata = json.load(f)
                        
                        # Vérifier si le document a été traité et que les embeddings ont été générés
                        if doc_metadata.get("processed", False) and doc_metadata.get("embeddings_generated", False):
                            # Vérifier les chunks
                            chunks_dir = os.path.join(self.metadata_path, "chunks")
                            chunk_count = 0
                            
                            if os.path.exists(chunks_dir):
                                for chunk_filename in os.listdir(chunks_dir):
                                    if chunk_filename.endswith('.json'):
                                        try:
                                            chunk_path = os.path.join(chunks_dir, chunk_filename)
                                            with open(chunk_path, 'r', encoding='utf-8') as f:
                                                chunk_data = json.load(f)
                                                if chunk_data.get("metadata", {}).get("document_id") == document_id:
                                                    chunk_count += 1
                                        except Exception:
                                            continue
                            
                            # Si le nombre de chunks correspond aux métadonnées, considérer comme traité
                            if chunk_count == doc_metadata.get("chunk_count", 0) and chunk_count > 0:
                                logger.info(f"Document déjà traité, ignoré: {os.path.basename(pdf_path)} "
                                        f"({chunk_count} chunks)")
                                stats["skipped_files"] += 1
                                stats["already_processed_files"] += 1
                                continue
                            else:
                                logger.warning(f"Document marqué comme traité mais chunks incohérents: "
                                            f"{os.path.basename(pdf_path)}. "
                                            f"Trouvé {chunk_count} chunks, attendu {doc_metadata.get('chunk_count', 0)}. "
                                            f"Retraitement...")
                    except Exception as e:
                        logger.warning(f"Erreur lors de la vérification des métadonnées: {e}. Retraitement...")
                
                logger.info(f"Traitement du fichier {os.path.basename(pdf_path)}")
                
                # Vérifier si le document est scanné avant de le traiter
                if detect_scanned:
                    try:
                        import fitz
                        doc = fitz.open(pdf_path)
                        is_scanned = self.pdf_processor.detect_scanned_document(doc)
                        doc.close()
                        
                        if is_scanned:
                            stats["scanned_documents"] += 1
                            logger.info(f"📑 Le document {os.path.basename(pdf_path)} est détecté comme scanné, activation de l'OCR")
                            self.pdf_processor.force_ocr = True
                        else:
                            # Rétablir la valeur par défaut
                            self.pdf_processor.force_ocr = self.force_ocr
                    except Exception as e:
                        logger.warning(f"Erreur lors de la détection de document scanné: {e}")
                
                # Processus principal: traiter le PDF
                if self.process_pdf(pdf_path, force_reprocess, detect_scanned=False):  # detect_scanned=False car déjà fait
                    stats["successful_files"] += 1
                else:
                    stats["failed_files"] += 1
                
                stats["processed_files"] += 1
                
                # Récupérer les statistiques de chunks traités
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        doc_metadata = json.load(f)
                    stats["chunks_processed"] += doc_metadata.get("chunk_count", 0)
                except Exception:
                    pass
                        
            except Exception as e:
                logger.error(f"Erreur lors du traitement de {pdf_path}: {e}")
                stats["failed_files"] += 1
        
        # Calculer les statistiques finales
        stats["success_rate"] = (stats["successful_files"] / stats["processed_files"]) * 100 if stats["processed_files"] > 0 else 0
        stats["status"] = "completed"
        
        logger.info(f"Traitement terminé. Statistiques: {stats}")
        return stats

    def get_document_info(self, document_id: str) -> Dict[str, Any]:
        """
        Récupère les informations sur un document spécifique.
        
        Args:
            document_id: Identifiant du document
            
        Returns:
            Dictionnaire avec les métadonnées du document
        """
        metadata_file = os.path.join(self.metadata_path, f"{document_id}.json")
        
        if not os.path.exists(metadata_file):
            logger.warning(f"Métadonnées non trouvées pour le document ID: {document_id}")
            return {}
        
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Erreur lors de la lecture des métadonnées: {e}")
            return {}
            
    def reindex_document(self, document_id: str) -> bool:
        """
        Supprime et réindexe un document spécifique.
        Utile après des mises à jour de modèles d'embedding.
        
        Args:
            document_id: Identifiant du document
            
        Returns:
            True si la réindexation a réussi, False sinon
        """
        try:
            # 1. Supprimer les entrées existantes de Milvus
            logger.info(f"Suppression des embeddings existants pour le document {document_id}")
            self.milvus_service.delete_by_document_id(document_id)
            
            # 2. Récupérer les informations du document
            doc_info = self.get_document_info(document_id)
            if not doc_info:
                logger.error(f"Impossible de trouver les métadonnées du document {document_id}")
                return False
                
            # 3. Retrouver le chemin du PDF
            pdf_path = doc_info.get("path")
            if not pdf_path or not os.path.exists(pdf_path):
                logger.error(f"Fichier PDF introuvable pour le document {document_id}: {pdf_path}")
                return False
            
            # 4. Retraiter le document
            logger.info(f"Réindexation du document {document_id}: {pdf_path}")
            success = self.process_pdf(pdf_path, force_reprocess=True)
            
            return success
            
        except Exception as e:
            logger.error(f"Erreur lors de la réindexation du document {document_id}: {e}")
            return False

# Fonction utilitaire pour tester le service
def test_pdf_service():
    from embedding_service import EmbeddingService
    from milvus_service import MilvusService
    
    # Initialiser les services
    embedding_service = EmbeddingService()
    milvus_service = MilvusService("documents_collection", dim=1024)  # Ajuster à votre modèle
    
    # Initialiser le service PDF avec les nouveaux paramètres
    pdf_service = PDFService(
        embedding_service,
        milvus_service,
        max_tokens=512,        # Nombre de tokens maximum par chunk
        overlap_tokens=100,    # Nombre de tokens de chevauchement
        force_ocr=False,       # Utiliser l'OCR seulement si nécessaire
        ocr_language="fra"     # Langue française pour les documents
    )
    
    # Traiter les PDFs
    stats = pdf_service.process_pdfs(detect_scanned=True)
    print(f"Résultats du traitement: {stats}")

    # Tester le retraitement forcé
    # stats = pdf_service.process_pdfs(force_reprocess=True)
    # print(f"🔍 Résultats détaillés du retraitement : {json.dumps(stats, indent=2)}")

if __name__ == "__main__":
    test_pdf_service()