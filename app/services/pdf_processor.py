import os
import fitz  # PyMuPDF
import io
import json
import logging
import hashlib
import re
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from fastapi import FastAPI
from PIL import Image

# Import du module ocr_utils
from app.services.ocr_utils import (
    extract_text_from_pdf_page,
    post_process_ocr_text,
    is_likely_scanned,
    preprocess_image_for_ocr,
    analyze_document_language,
    enhance_legal_document_ocr
)

# Configuration du logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

class PDFProcessor:
    def __init__(
       self,
        metadata_path: str = "/home/mea/Documents/modelAi/metadata",
        max_tokens: int = 512,
        overlap_tokens: int = 100,
        ocr_language: str = "fra",
        ocr_threshold: int = 50,
        force_ocr: bool = False  # Nouveau paramètre pour forcer l'OCR
    ):
        self.metadata_path = metadata_path
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.ocr_language = ocr_language
        self.ocr_threshold = ocr_threshold
        self.force_ocr = force_ocr
        
        # Créer le répertoire de métadonnées s'il n'existe pas
        os.makedirs(self.metadata_path, exist_ok=True)

    def extract_text_from_page(self, page) -> Tuple[str, str]:
        """
        Extrait le texte d'une page PDF en utilisant le module ocr_utils.
        
        Args:
            page: Page PyMuPDF
            
        Returns:
            Tuple (texte extrait, méthode d'extraction)
        """
        # Utiliser la fonction de ocr_utils pour extraire le texte
        text, extraction_method = extract_text_from_pdf_page(
            page,
            ocr_language=self.ocr_language,
            perform_ocr=self.force_ocr
        )
        
        # Détecter la langue du document si nécessaire
        if text and len(text) > 100 and (self.ocr_language == "auto" or not self.ocr_language):
            detected_language = analyze_document_language(text)
            
            # Si la langue détectée est différente, réessayer l'extraction
            if detected_language != self.ocr_language:
                logger.info(f"Langue détectée: {detected_language}, réextraction avec cette langue")
                text, extraction_method = extract_text_from_pdf_page(
                    page,
                    ocr_language=detected_language,
                    perform_ocr=self.force_ocr
                )
        
        # Améliorer le texte pour les documents juridiques
        if extraction_method == "ocr":
            # Post-traitement OCR pour documents juridiques
            text = enhance_legal_document_ocr(text)
        
        return text, extraction_method

    def chunk_text_by_tokens(self, text: str, document_id: str, page_number: int) -> List[Dict[str, Any]]:
        """
        Découpe le texte en segments basés sur les tokens, avec essai préalable des méthodes spécialisées.
        
        Args:
            text: Texte à découper
            document_id: ID du document
            page_number: Numéro de la page
            
        Returns:
            Liste de chunks avec métadonnées
        """
        try:
            # Prétraiter le texte
            processed_text = self.preprocess_text(text)
            
            # 1. Détecter les caractéristiques du document
            is_legal_doc = any(pattern in processed_text.lower() for pattern in 
                          ["article", "chapitre", "section", "alinéa", "paragraphe", "loi", "décret"])
            
            # Vérifier si le document contient des motifs caractéristiques de CIREX
            cirex_patterns = [
                r'(?:^|\n)\s*TITRE\s+([IVX]+)(?:\s*[-–:.]|\s+)([^\n]+)',
                r'(?:^|\n)\s*([IVX]+)(?:\s*[-–:.]|\s+)([^\n]+)',
                r'(?:^|\n)\s*([A-Z])(?:\s*[-–:.]|\s+)([^\n]+)',
                r'(?:^|\n)\s*(\d+)(?:\s*[-–:.]|\s+)([^\n]+)'
            ]
            
            is_cirex_doc = False
            pattern_matches = 0
            for pattern in cirex_patterns:
                if re.search(pattern, processed_text):
                    pattern_matches += 1
                    if pattern_matches >= 2:
                        is_cirex_doc = True
                        break
            
            # 2. Essayer d'abord les méthodes spécialisées
            
            # Si c'est un document juridique avec articles, essayer le chunking sémantique
            if is_legal_doc:
                semantic_chunks = self.chunk_text_semantic(processed_text, document_id, page_number)
                if semantic_chunks:
                    return semantic_chunks
            
            # Si c'est un document CIREX, essayer le chunking document
            if is_cirex_doc:
                cirex_chunks = self.chunk_text_document(processed_text, document_id, page_number)
                if cirex_chunks:
                    return cirex_chunks
            
            # Vérifier la présence de tableaux et les traiter spécialement
            #table_chunks = self.chunk_text_with_context_preservation(processed_text, document_id, page_number)
            #if table_chunks:
                #return table_chunks
             # 3. Vérifier la présence de tableaux et les traiter spécialement si disponible
            if hasattr(self, 'chunk_text_with_context_preservation'):
                table_patterns = [
                    r'(?:\s*\|[^|]+\|[^|]+\|[^|]*\s*\n){3,}',  # Lignes avec plusieurs |
                    r'(?:\s*\w+\s{2,}\w+\s{2,}\w+\s*\n){3,}',  # Lignes avec espaces alignés
                    r'[+\-=]{5,}[\s\n]+(?:[^+\-=\n]*[\s\n]+[+\-=]{5,}[\s\n]+){2,}'  # Séparateurs de lignes
                ]
                
                has_tables = any(re.search(pattern, processed_text) for pattern in table_patterns)
                
                if has_tables:
                    table_chunks = self.chunk_text_with_context_preservation(processed_text, document_id, page_number)
                    if table_chunks:
                        return table_chunks
            
            # Essayer le chunking par paragraphes pour les discours/documents généraux
            if not is_legal_doc and not is_cirex_doc:
                doc_type = "DISCOURS" if "discours" in processed_text.lower() else "GENERAL"
                paragraph_chunks = self._chunk_by_paragraphs(processed_text, document_id, page_number, doc_type)
                if paragraph_chunks:
                    return paragraph_chunks
            
            # 3. En dernier recours, utiliser le chunking par tokens
            
            # Importer tiktoken pour une tokenisation adaptée aux LLMs
            import tiktoken
            
            # Utiliser l'encodeur cl100k_base qui est utilisé par beaucoup de modèles
            encoder = tiktoken.get_encoding("cl100k_base")
            
            # Encoder le texte en tokens
            tokens = encoder.encode(processed_text)
            
            # Si aucun token, retourner liste vide
            if not tokens:
                return []
            
            chunks = []
            start = 0
            
            # Adapter la taille des chunks et le chevauchement selon la complexité du texte
            max_tokens = self.max_tokens
            overlap_tokens = self.overlap_tokens
            
            if is_legal_doc:
                # Réduire la taille des chunks pour les documents juridiques complexes
                max_tokens = min(self.max_tokens, 400)
                # Augmenter le chevauchement pour maintenir le contexte
                overlap_tokens = min(max_tokens // 3, 150)
            
            while start < len(tokens):
                # Extraire un segment de tokens avec la taille maximale
                end = min(start + max_tokens, len(tokens))
                
                # Ajustement intelligent de la fin du chunk pour ne pas couper en plein milieu d'une phrase
                if end < len(tokens):
                    # Encoder les symboles de fin de phrase
                    end_sentence_tokens = encoder.encode(". ")[0:1]
                    
                    # Chercher la fin de phrase la plus proche dans la zone de chevauchement
                    cutoff_zone_start = end - min(end - start, overlap_tokens * 2)
                    cutoff_zone = tokens[cutoff_zone_start:end]
                    
                    # Chercher en partant de la fin
                    for i in range(len(cutoff_zone) - 1, -1, -1):
                        if cutoff_zone[i] in end_sentence_tokens:
                            # Ajuster la fin au niveau de la fin de phrase
                            end = cutoff_zone_start + i + 1
                            break
                
                # Décoder les tokens en texte
                chunk_tokens = tokens[start:end]
                chunk_text = encoder.decode(chunk_tokens)
                
                # Générer un ID unique pour le chunk
                chunk_id = hashlib.md5(
                    f"{document_id}_{page_number}_{start}".encode()
                ).hexdigest()
                
                # Extraire un titre pour le chunk si possible
                chunk_title = ""
                first_line = chunk_text.split('\n', 1)[0].strip()
                if len(first_line) < 100:
                    chunk_title = first_line
                
                # Déterminer le type de document pour les métadonnées
                doc_type = "GENERAL"
                if is_legal_doc:
                    doc_type = "LEGAL"
                elif is_cirex_doc:
                    doc_type = "CIREX"
                
                # Créer les métadonnées du chunk
                chunk_data = {
                    "text": chunk_text,
                    "metadata": {
                        "chunk_id": chunk_id,
                        "document_id": document_id,
                        "page_number": page_number,
                        "start_token": start,
                        "end_token": end,
                        "token_count": len(chunk_tokens),
                        "chunk_title": chunk_title,
                        "is_legal_document": is_legal_doc,
                        "document_type": doc_type,
                        "extraction_method": "text_tokenized"
                    }
                }
                
                chunks.append(chunk_data)
                
                # Avancer avec chevauchement
                start += max_tokens - overlap_tokens
                
                # Arrêter si on a tout traité ou si le chevauchement ne laisse pas assez de tokens
                if start >= len(tokens) or (end - start) < overlap_tokens // 2:
                    break
            
            return chunks
            
        except ImportError:
            logger.warning("tiktoken non disponible, utilisation de la méthode de secours")
            return self._fallback_chunk_text(text, document_id, page_number)

    def chunk_text_semantic(self, text: str, document_id: str, page_number: int) -> List[Dict[str, Any]]:
        
        """
        Découpe le texte en segments basés sur la structure sémantique des documents juridiques.
        
        Args:
            text: Texte à découper
            document_id: ID du document
            page_number: Numéro de la page
            
        Returns:
            Liste de chunks avec métadonnées
        """
        if not text.strip():
            return []
            
        chunks = []
        
        try:
            
            # 1. Identifier les marqueurs de structure juridique
            section_patterns = [
                (r'(?i)article\s+(\d+[a-z]*)(?:\s*[-:]\s*|\s*\.\s*|\s+)', "ARTICLE"),
                (r'(?i)section\s+(\d+[a-z]*)(?:\s*[-:]\s*|\s*\.\s*|\s+)', "SECTION"),
                (r'(?i)chapitre\s+(\d+[a-z]*)(?:\s*[-:]\s*|\s*\.\s*|\s+)', "CHAPITRE"),
                (r'(?i)titre\s+(\d+[a-z]*)(?:\s*[-:]\s*|\s*\.\s*|\s+)', "TITRE"),
                (r'(?i)partie\s+(\d+[a-z]*)(?:\s*[-:]\s*|\s*\.\s*|\s+)', "PARTIE"),
                (r'(?i)paragraphe\s+(\d+[a-z]*)(?:\s*[-:]\s*|\s*\.\s*|\s+)', "PARAGRAPHE"),
                (r'(?i)alinéa\s+(\d+[a-z]*)(?:\s*[-:]\s*|\s*\.\s*|\s+)', "ALINEA"),
            ]
            
            # 2. Trouver toutes les occurrences des marqueurs
            section_matches = []
            for pattern, section_type in section_patterns:
                for match in re.finditer(pattern, text):
                    section_num = match.group(1) if match.groups() else ""
                    section_matches.append({
                        "type": section_type,
                        "number": section_num,
                        "start": match.start(),
                        "match_text": match.group(0)
                    })
            
            # 3. Trier les marqueurs par position dans le texte
            section_matches.sort(key=lambda x: x["start"])
            
            # Si aucun marqueur n'est trouvé, essayer d'autres approches de chunking
            if not section_matches:
                # Essayer de découper par paragraphes pour les circulaires et notes
                paragraphs = re.split(r'\n\s*\n', text)
                if len(paragraphs) > 1:
                    for i, para in enumerate(paragraphs):
                        if len(para.strip()) < 20:  # Ignorer les paragraphes trop courts
                            continue
                        
                        chunk_id = hashlib.md5(
                            f"{document_id}_{page_number}_para{i}".encode()
                        ).hexdigest()
                        
                        # Créer un chunk pour ce paragraphe
                        chunk_data = {
                            "text": para,
                            "metadata": {
                                "chunk_id": chunk_id,
                                "document_id": document_id,
                                "page_number": page_number,
                                "paragraph_index": i,
                                "char_count": len(para),
                                "is_legal_document": True,
                                "extraction_method": "semantic_paragraph"
                            }
                        }
                        chunks.append(chunk_data)
                    
                    return chunks
                
                # Si peu de paragraphes, retourner vide pour utiliser le chunking par tokens
                return []
            
            # 4. Créer les chunks basés sur les sections identifiées
            for i, section in enumerate(section_matches):
                # Déterminer la fin de cette section (début de la prochaine ou fin du texte)
                next_start = section_matches[i+1]["start"] if i < len(section_matches) - 1 else len(text)
                
                # Extraire le texte de la section
                section_text = text[section["start"]:next_start]
                
                # Vérifier que le texte n'est pas trop court
                if len(section_text.strip()) < 20:
                    continue
                
                # Générer un ID unique pour le chunk
                section_id = f"{section['type']}_{section['number']}"
                chunk_id = hashlib.md5(
                    f"{document_id}_{page_number}_{section_id}".encode()
                ).hexdigest()
                
                # Créer les métadonnées du chunk
                chunk_data = {
                    "text": section_text,
                    "metadata": {
                        "chunk_id": chunk_id,
                        "document_id": document_id,
                        "page_number": page_number,
                        "section_type": section["type"],
                        "section_number": section["number"],
                        "section_id": section_id,
                        "char_count": len(section_text),
                        "is_legal_document": True,
                        "extraction_method": "semantic_section"
                    }
                }
                
                chunks.append(chunk_data)
            
            # 5. Si des sections ont été identifiées mais certaines parties restent non couvertes
            if chunks and section_matches[0]["start"] > 0:
                # Traiter le préambule ou introduction avant la première section
                intro_text = text[:section_matches[0]["start"]]
                if len(intro_text.strip()) > 100:  # S'assurer qu'il y a un contenu substantiel
                    intro_id = hashlib.md5(
                        f"{document_id}_{page_number}_intro".encode()
                    ).hexdigest()
                    
                    chunk_data = {
                        "text": intro_text,
                        "metadata": {
                            "chunk_id": intro_id,
                            "document_id": document_id,
                            "page_number": page_number,
                            "section_type": "INTRO",
                            "section_id": "introduction",
                            "char_count": len(intro_text),
                            "is_legal_document": True,
                            "extraction_method": "semantic_intro"
                        }
                    }
                    chunks.insert(0, chunk_data)  # Ajouter l'intro au début
            
            return chunks
            
        except Exception as e:
            logger.error(f"Erreur lors du chunking sémantique: {e}")
            return []
        
    def chunk_text_document(self, text: str, document_id: str, page_number: int) -> List[Dict[str, Any]]:
        """
        Méthode spécialisée pour découper les documents de type CIREX avec structure
        de titres, sections numérotées et points.
        
        Args:
            text: Texte à découper
            document_id: ID du document
            page_number: Numéro de la page
                
        Returns:
            Liste de chunks avec métadonnées
        """
        if not text.strip():
            return []
            
        chunks = []
        
        try:
            # Patterns pour les structures documents
            cirex_patterns = [
                # Titres principaux (TITRE I, TITRE II, etc.)
                (r'(?:^|\n)\s*TITRE\s+([IVX]+)(?:\s*[-–:.]|\s+)([^\n]+)', "TITRE"),
                
                # Sections principales (I, II, III...)
                (r'(?:^|\n)\s*([IVX]+)(?:\s*[-–:.]|\s+)([^\n]+)', "SECTION"),
                
                # Sous-sections alphabétiques (A, B, C...)
                (r'(?:^|\n)\s*([A-Z])(?:\s*[-–:.]|\s+)([^\n]+)', "SOUS_SECTION"),
                
                # Points numérotés (1, 2, 3...)
                (r'(?:^|\n)\s*(\d+)(?:\s*[-–:.]|\s+)([^\n]+)', "POINT"),
                
                # Sous-points (1.1, 1.2, a), b), etc.)
                (r'(?:^|\n)\s*(?:\d+\.\d+|\w+\))(?:\s*[-–:.]|\s+)([^\n]+)', "SOUS_POINT"),
            ]
            
            # Trouver toutes les occurrences des marqueurs
            section_matches = []
            for pattern, section_type in cirex_patterns:
                for match in re.finditer(pattern, text):
                    # Obtenir la position et le texte complet du match
                    number = match.group(1) if match.groups() else ""
                    title = match.group(2) if len(match.groups()) > 1 else ""
                    section_matches.append({
                        "type": section_type,
                        "number": number,
                        "title": title,
                        "start": match.start(),
                        "match_text": match.group(0)
                    })
            
            # Si pas de sections trouvées, essayer de découper par paragraphes
            if not section_matches:
                return self._chunk_by_paragraphs(text, document_id, page_number, "CIREX")
            
            # Trier les marqueurs par position dans le texte
            section_matches.sort(key=lambda x: x["start"])
            
            # Créer les chunks basés sur les sections
            for i, section in enumerate(section_matches):
                # Déterminer la fin de la section (début de la suivante ou fin du texte)
                next_start = section_matches[i+1]["start"] if i < len(section_matches) - 1 else len(text)
                
                # Extraire le texte de la section
                section_text = text[section["start"]:next_start].strip()
                
                # Vérifier que le texte est assez long
                if len(section_text) < 30:
                    continue
                    
                # Générer un ID unique pour le chunk
                section_id = f"{section['type']}_{section['number']}_{page_number}"
                chunk_id = hashlib.md5(
                    f"{document_id}_{section_id}".encode()
                ).hexdigest()
                
                # Créer les métadonnées du chunk
                chunk_data = {
                    "text": section_text,
                    "metadata": {
                        "chunk_id": chunk_id,
                        "document_id": document_id,
                        "page_number": page_number,
                        "section_type": section["type"],
                        "section_number": section["number"],
                        "section_title": section["title"],
                        "document_type": "CIREX",
                        "char_count": len(section_text),
                        "extraction_method": "cirex_section"
                    }
                }
                
                chunks.append(chunk_data)
            
            # Traiter le texte avant la première section (préambule/introduction)
            if section_matches and section_matches[0]["start"] > 0:
                intro_text = text[:section_matches[0]["start"]].strip()
                if len(intro_text) > 100:
                    intro_id = hashlib.md5(
                        f"{document_id}_{page_number}_intro".encode()
                    ).hexdigest()
                    
                    chunk_data = {
                        "text": intro_text,
                        "metadata": {
                            "chunk_id": intro_id,
                            "document_id": document_id,
                            "page_number": page_number,
                            "section_type": "INTRO",
                            "document_type": "CIREX",
                            "char_count": len(intro_text),
                            "extraction_method": "cirex_intro"
                        }
                    }
                    chunks.insert(0, chunk_data)
            
            return chunks
                
        except Exception as e:
            logger.error(f"Erreur lors du chunking du document CIREX: {e}")
            return []

    def _chunk_by_paragraphs(self, text: str, document_id: str, page_number: int, doc_type: str = "GENERAL") -> List[Dict[str, Any]]:
        """
        Découpe le texte par paragraphes distincts.
        Utile pour les discours, programmes et documents sans structure claire.
        
        Args:
            text: Texte à découper
            document_id: ID du document
            page_number: Numéro de la page
            doc_type: Type de document (GENERAL, DISCOURS, PROGRAMME, etc.)
                
        Returns:
            Liste de chunks avec métadonnées
        """
        chunks = []
        
        # Découper par paragraphes distincts (séparés par une ou plusieurs lignes vides)
        paragraphs = re.split(r'\n\s*\n', text)
        
        for i, para in enumerate(paragraphs):
            # Ignorer les paragraphes trop courts qui pourraient être du bruit
            if len(para.strip()) < 50:
                continue
            
            # Générer un ID unique pour le chunk
            chunk_id = hashlib.md5(
                f"{document_id}_{page_number}_para{i}".encode()
            ).hexdigest()
            
            # Créer un chunk pour ce paragraphe
            chunk_data = {
                "text": para.strip(),
                "metadata": {
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "page_number": page_number,
                    "chunk_type": "paragraph",
                    "paragraph_index": i,
                    "document_type": doc_type,
                    "char_count": len(para),
                    "extraction_method": f"{doc_type.lower()}_paragraph"
                }
            }
            chunks.append(chunk_data)
        
        return chunks


    def process_pdf(self, pdf_path: str, force_reprocess: bool = False) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Traite un document PDF en utilisant une méthode de chunking adaptative qui préserve 
        le contexte des tableaux et détecte automatiquement le type de document.
        
        Args:
            pdf_path: Chemin vers le fichier PDF
            force_reprocess: Forcer le retraitement même si déjà traité
                
        Returns:
            Tuple de (liste des chunks, métadonnées du document)
        """
        if not os.path.exists(pdf_path):
            logger.error(f"Le fichier PDF n'existe pas: {pdf_path}")
            return [], {}
        
        try:
          
            # Générer l'ID du document
            document_id = self.generate_document_id(pdf_path)
            logger.info(f"🔍 Traitement du document {os.path.basename(pdf_path)} (ID: {document_id})")
            
            # Toujours forcer le retraitement si demandé
            #force_reprocess = True
            
            # Supprimer le fichier de métadonnées existant si force_reprocess est True
            metadata_file = os.path.join(self.metadata_path, f"{document_id}.json")
            if force_reprocess and os.path.exists(metadata_file):
                os.remove(metadata_file)
                logger.info(f"📄 Fichier de métadonnées supprimé : {metadata_file}")
            
            # Vérifier si le document a déjà été traité (sauf si force_reprocess)
            if os.path.exists(metadata_file) and not force_reprocess:
                logger.info(f"Document déjà traité, chargement des métadonnées: {os.path.basename(pdf_path)}")
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    doc_metadata = json.load(f)
                    
                if doc_metadata.get("processed", False):
                    # Charger les chunks existants
                    chunks = []
                    chunks_dir = os.path.join(self.metadata_path, "chunks")
                    if os.path.exists(chunks_dir):
                        for chunk_filename in os.listdir(chunks_dir):
                            if chunk_filename.endswith('.json'):
                                chunk_path = os.path.join(chunks_dir, chunk_filename)
                                try:
                                    with open(chunk_path, 'r', encoding='utf-8') as f:
                                        chunk_data = json.load(f)
                                        if chunk_data.get("metadata", {}).get("document_id") == document_id:
                                            chunks.append(chunk_data)
                                except Exception as e:
                                    logger.warning(f"Erreur lors du chargement du chunk {chunk_filename}: {e}")
                                    
                    logger.info(f"✅ {len(chunks)} chunks chargés pour le document déjà traité")
                    return chunks, doc_metadata
            
             # Ouvrir le document PDF
            doc = fitz.open(pdf_path)
            
            # Extraire les métadonnées
            doc_metadata = self.extract_document_metadata(doc, pdf_path)
            doc_metadata["document_id"] = document_id
            
            # Traiter chaque page
            all_chunks = []
            page_metadata = []
            
            for page_number, page in enumerate(doc, start=1):
                logger.info(f"📖 Traitement de la page {page_number}/{len(doc)}")
                
                # Extraire le texte (avec OCR si nécessaire)
                text, extraction_method = self.extract_text_from_page(page)
                
                # Prétraiter le texte
                processed_text = self.preprocess_text(text)
                
                # Ajouter un log pour le texte extrait
                logger.info(f"📝 Longueur du texte extrait : {len(processed_text)} caractères")
                
                # Créer les métadonnées de la page
                page_meta = {
                    "page_number": page_number,
                    "width": page.rect.width,
                    "height": page.rect.height,
                    "rotation": page.rotation,
                    "extraction_method": extraction_method,
                    "char_count": len(processed_text)
                }
                page_metadata.append(page_meta)
                
                # Découper en chunks
                page_chunks = self.chunk_text_by_tokens(processed_text, document_id, page_number)
                
                # Log du nombre de chunks pour cette page
                logger.info(f"📦 Nombre de chunks générés pour la page {page_number}: {len(page_chunks)}")
                
                # Mettre à jour la méthode d'extraction pour tous les chunks
                for chunk in page_chunks:
                    chunk["metadata"]["extraction_method"] = extraction_method
                
                all_chunks.extend(page_chunks)
            
            # Fermer le document
            doc.close()
            
            # Sauvegarder les métadonnées du document
            doc_metadata["pages"] = page_metadata
            doc_metadata["chunk_count"] = len(all_chunks)
            doc_metadata["processed"] = True
            doc_metadata["chunking_method"] = "token_based"
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(doc_metadata, f, ensure_ascii=False, indent=2)
            
            # AJOUTER CE LOG
            logger.info(f"✅ Document traité avec succès: {len(all_chunks)} chunks générés")
            # AJOUTER CE LOG POUR VOIR UN EXEMPLE DE CHUNK
            if all_chunks:
                logger.info(f"Exemple de chunk: {json.dumps(all_chunks[0], ensure_ascii=False)[:500]}...")
            
            # AJOUTER LA SAUVEGARDE DES CHUNKS
            self.save_chunks_metadata(all_chunks)
            
            return all_chunks, doc_metadata
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du traitement du PDF {pdf_path}: {e}")
            import traceback
            traceback.print_exc()  # Imprimer la trace complète de l'erreur
            return [], {}
    
    def detect_scanned_document(self, doc) -> bool:
        """
        Détecte si un document est scanné en analysant ses premières pages.
        Utilise is_likely_scanned de ocr_utils.
        
        Args:
            doc: Document PyMuPDF
            
        Returns:
            Boolean indiquant si le document est probablement scanné
        """
        # Examiner les 3 premières pages ou toutes les pages si moins de 3
        pages_to_check = min(3, len(doc))
        scanned_pages = 0
        
        for i in range(pages_to_check):
            page = doc[i]
            text = page.get_text()
            
            # Utiliser la fonction de ocr_utils
            if is_likely_scanned(text, page):
                scanned_pages += 1
                
        # Si plus de la moitié des pages examinées sont scannées, considérer tout le document comme scanné
        return scanned_pages >= (pages_to_check / 2)

    def detect_document_language(self, pdf_path):
        """
        Détecte la langue principale du document et met à jour l'attribut ocr_language.
        Utilise analyze_document_language de ocr_utils.
        
        Args:
            pdf_path: Chemin vers le fichier PDF
        """
        try:
            doc = fitz.open(pdf_path)
            sample_text = ""
            
            # Récupérer le texte des 2 premières pages pour analyse
            for i in range(min(2, len(doc))):
                page_text = doc[i].get_text()
                sample_text += page_text + "\n"
                
                # Si on a assez de texte, arrêter
                if len(sample_text) > 1000:
                    break
            
            # Fermer le document
            doc.close()
            
            # Détecter la langue avec ocr_utils
            if sample_text.strip():
                detected_language = analyze_document_language(sample_text)
                if detected_language:
                    logger.info(f"Langue détectée pour le document: {detected_language}")
                    self.ocr_language = detected_language
            
        except Exception as e:
            logger.warning(f"Erreur lors de la détection de la langue: {e}. Utilisation de la langue par défaut.")
    
    def detect_document_type(self, text_or_doc, filename: str = "") -> str:
        """
        Détecte automatiquement le type de document pour choisir la méthode
        de chunking appropriée.
        
        Args:
            text_or_doc: Texte du document ou objet Document PyMuPDF
            filename: Nom du fichier (optionnel)
        
        Returns:
            Type de document détecté (LEGAL, CIREX, DISCOURS, PROGRAMME, etc.)
        """
        # Gestion des différents types d'entrée
        if hasattr(text_or_doc, 'get_text'):
            # Si c'est un objet Document PyMuPDF, extraire le texte
            text = text_or_doc.get_text()
        elif isinstance(text_or_doc, str):
            text = text_or_doc
        else:
            raise ValueError("L'entrée doit être un texte ou un document PyMuPDF")
        
        # Normaliser le texte
        text_lower = text.lower()
        
        # Détection multilingue
        try:
            from langdetect import detect
            lang = detect(text[:500])
        except (ImportError, Exception):
            lang = 'fr'  # Fallback au français
        
        # Mots-clés multilingues
        multilingual_keywords = {
            'fr': {
                'legal': ['loi', 'code', 'constitution', 'décret', 'article'],
                'cirex': ['circulaire', 'ministère', 'administration', 'titre', 'section'],
                'discours': ['discours', 'allocution', 'président', 'excellence', 'chef de l\'état'],
                'programme': ['programme', 'gouvernement', 'orientation', 'politique'],
                'finances': ['loi de finances', 'budget', 'exercice', 'recettes', 'dépenses'],
                'cameroon': ['cameroun', 'république', 'assemblée nationale']
            },
            'en': {
                'legal': ['law', 'code', 'constitution', 'decree', 'article'],
                'cirex': ['circular', 'ministry', 'administration', 'title', 'section'],
                'discours': ['speech', 'address', 'president', 'excellence', 'head of state'],
                'programme': ['program', 'government', 'orientation', 'policy'],
                'finances': ['finance law', 'budget', 'fiscal', 'revenue', 'expenditure'],
                'cameroon': ['cameroon', 'republic', 'national assembly']
            }
        }
        
        # Sélectionner les mots-clés en fonction de la langue détectée
        keywords = multilingual_keywords.get(lang, multilingual_keywords['fr'])
        
        # Règles de détection basées sur le contenu et les mots-clés
        
        # 1. Documents Légaux
        if (re.search(r'(ARTICLE|SECTION)\s+\d+', text, re.IGNORECASE) and 
            any(kw in text_lower for kw in keywords['legal'])):
            return "LEGAL"
        
        # 2. Circulaires et Documents Administratifs
        if ((re.search(r'TITRE\s+[IVX]+', text, re.IGNORECASE) or 
            re.search(r'SECTION\s+[IVX]+', text, re.IGNORECASE)) and 
            any(kw in text_lower for kw in keywords['cirex'])):
            return "CIREX"
        
        # 3. Discours Officiels
        if (any(kw in text_lower[:3000] for kw in keywords['discours']) and 
            ("cameroon" in text_lower[:3000] or "cameroun" in text_lower[:3000])):
            return "DISCOURS"
        
        
        
        # Fallback
        return "GENERAL"

    def chunk_text_with_context_preservation(self, text: str, document_id: str, page_number: int) -> List[Dict[str, Any]]:
        """
        Découpe le texte en préservant le contexte des tableaux et la structure sémantique.
        Cette approche améliorée garantit que les tableaux sont inclus avec leur contexte explicatif.
        
        Args:
            text: Texte à découper
            document_id: ID du document
            page_number: Numéro de la page
                
        Returns:
            Liste de chunks avec métadonnées incluant le contexte des tableaux
        """
        chunks = []
        
        # Patrons pour détecter les tableaux
        table_patterns = [
            # Lignes avec plusieurs | ou + (tableaux ASCII)
            r'(?:\s*\|[^|]+\|[^|]+\|[^|]*\s*\n){3,}',
            # Lignes avec espaces alignés (pseudo-tableaux)
            r'(?:\s*\w+\s{2,}\w+\s{2,}\w+\s*\n){3,}',
            # Tableaux avec séparateurs de lignes (+----- ou ======)
            r'[+\-=]{5,}[\s\n]+(?:[^+\-=\n]*[\s\n]+[+\-=]{5,}[\s\n]+){2,}'
        ]
        
        # Identifier les positions des tableaux dans le texte
        table_positions = []
        for pattern in table_patterns:
            for match in re.finditer(pattern, text, re.MULTILINE):
                table_positions.append((match.start(), match.end(), match.group(0)))
        
        # Trier les positions des tableaux
        table_positions.sort()
        
        # Si aucun tableau trouvé, retourner une liste vide pour signaler qu'il faut utiliser d'autres méthodes
        if not table_positions:
            return []
        
        # 1. MÉTHODE AMÉLIORÉE: Préserver le contexte des tableaux
        
        # Extraire des sections qui contiennent un tableau avec son contexte environnant
        contextual_sections = []
        last_end = 0
        
        for start, end, table_content in table_positions:
            # Rechercher le contexte avant le tableau (jusqu'à 3 paragraphes ou 500 caractères)
            context_start = start
            
            # Trouver le début du contexte en cherchant 2-3 paragraphes en arrière
            paragraphs_before = re.finditer(r'\n\s*\n', text[max(0, start-1000):start])
            paragraph_positions = [m.end() + max(0, start-1000) for m in paragraphs_before]
            
            # Si nous avons des séparations de paragraphes, prendre les 2-3 derniers
            if paragraph_positions and len(paragraph_positions) > 2:
                context_start = paragraph_positions[-3]  # Prendre les 3 derniers paragraphes
            elif paragraph_positions and len(paragraph_positions) > 0:
                context_start = paragraph_positions[0]   # Prendre ce qu'on a
            elif start > 500:
                context_start = max(last_end, start - 500)  # Ou prendre ~500 caractères avant
            
            # Rechercher le contexte après le tableau (jusqu'à 2 paragraphes ou 300 caractères)
            context_end = end
            
            paragraphs_after = list(re.finditer(r'\n\s*\n', text[end:min(end+500, len(text))]))
            
            if paragraphs_after and len(paragraphs_after) >= 2:
                context_end = end + paragraphs_after[1].start()  # Prendre les 2 premiers paragraphes
            elif paragraphs_after and len(paragraphs_after) == 1:
                context_end = end + paragraphs_after[0].start()  # Prendre le premier paragraphe
            elif len(text) > end + 300:
                context_end = min(len(text), end + 300)  # Ou prendre ~300 caractères après
            else:
                context_end = len(text)  # Prendre jusqu'à la fin
                
            # Créer une section contextualisée
            contextual_section = text[context_start:context_end]
            contextual_sections.append((context_start, context_end, contextual_section))
            
            last_end = context_end
        
        # Traiter les sections contextuelles (tableau + contexte)
        for i, (start, end, section) in enumerate(contextual_sections):
            # Nettoyer et normaliser le texte
            cleaned_section = self.preprocess_text(section)
            
            # Générer un ID unique pour le chunk contextuel
            context_id = hashlib.md5(
                f"{document_id}_{page_number}_context_{i}".encode()
            ).hexdigest()
            
            # Détecter le type de document basé sur le contenu plutôt que d'utiliser detect_document_type
            # Vérifier la présence d'éléments typiques des documents
            doc_type = "GENERAL"
            if re.search(r'article\s+\d+', cleaned_section.lower()) or re.search(r'chapitre\s+[IVX]+', cleaned_section):
                doc_type = "LEGAL"
            elif re.search(r'titre\s+[IVX]+', cleaned_section) or "circulaire" in cleaned_section.lower():
                doc_type = "CIREX"
            elif "discours" in cleaned_section.lower() and ("excellence" in cleaned_section.lower() or "président" in cleaned_section.lower()):
                doc_type = "DISCOURS"
            
            # Extraire un titre si possible
            section_title = ""
            for line in cleaned_section.split('\n')[:5]:  # Chercher dans les 5 premières lignes
                # Chercher un titre potentiel (e.g., "Tableau 1: Taux d'imposition")
                if re.search(r'tableau|article|barème|annexe', line.lower()) and len(line) < 100:
                    section_title = line.strip()
                    break
            
            # Détecter le modèle de tableau utilisé
            table_type = "unknown"
            if "|" in cleaned_section:
                table_type = "ascii"
            elif re.search(r'[+\-=]{5,}', cleaned_section):
                table_type = "separator"
            else:
                table_type = "aligned"
            
            # Créer un chunk pour la section contextuelle
            chunk_data = {
                "text": cleaned_section,
                "metadata": {
                    "chunk_id": context_id,
                    "document_id": document_id,
                    "page_number": page_number,
                    "chunk_type": "table_with_context",
                    "table_index": i,
                    "table_type": table_type,
                    "section_title": section_title,
                    "document_type": doc_type,
                    "char_count": len(cleaned_section),
                    "extraction_method": "contextual_table_extraction"
                }
            }
            chunks.append(chunk_data)
        
        # 2. Traiter les portions de texte entre les sections contextuelles
        
        # Identifier les zones de texte entre les sections contextuelles
        text_sections = []
        last_end = 0
        
        for start, end, _ in contextual_sections:
            if start > last_end:
                # Il y a du texte entre la fin précédente et le début de cette section
                text_sections.append((last_end, start, text[last_end:start]))
            last_end = end
        
        # Ajouter la dernière section de texte si nécessaire
        if last_end < len(text):
            text_sections.append((last_end, len(text), text[last_end:]))
        
        # Traiter les sections de texte normales
        for i, (start, end, section) in enumerate(text_sections):
            if len(section.strip()) < 50:  # Ignorer les petites sections (probablement du bruit)
                continue
            
            # Au lieu d'utiliser detect_document_type, détecter le type basé sur le contenu
            # Vérifier la présence d'éléments typiques des documents
            doc_type = "GENERAL"
            if re.search(r'article\s+\d+', section.lower()) or re.search(r'chapitre\s+[IVX]+', section.lower()):
                doc_type = "LEGAL"
                # Essayer le chunking sémantique pour LEGAL
                section_chunks = self.chunk_text_semantic(section, document_id, page_number)
                if not section_chunks:  # Si ça échoue, utiliser le chunking par paragraphes
                    section_chunks = self._chunk_by_paragraphs(section, document_id, page_number, doc_type)
            elif re.search(r'titre\s+[IVX]+', section) or "circulaire" in section.lower():
                doc_type = "CIREX"
                # Essayer le chunking document pour CIREX
                section_chunks = self.chunk_text_document(section, document_id, page_number)
                if not section_chunks:  # Si ça échoue, utiliser le chunking par paragraphes
                    section_chunks = self._chunk_by_paragraphs(section, document_id, page_number, doc_type)
            else:
                # Pour tous les autres types, utiliser le chunking par paragraphes
                section_chunks = self._chunk_by_paragraphs(section, document_id, page_number, doc_type)
            
            # Ajouter un indicateur que ces chunks sont entre des sections tabulaires
            for chunk in section_chunks:
                chunk["metadata"]["between_tables"] = True
                chunk["metadata"]["preceding_table_index"] = i if i > 0 else None
                chunk["metadata"]["following_table_index"] = i if i < len(text_sections) - 1 else None
            
            chunks.extend(section_chunks)
        
        # Trier tous les chunks par leur position dans le document
        # Utiliser une méthode sûre pour le tri qui ne dépend pas de start_position
        # Utilisation de l'ordre dans lequel ils ont été créés
        
        return chunks

    def _detect_table_references(self, text: str) -> List[Dict]:
        """
        Détecte les références à des tableaux dans le texte.
        Utile pour établir des liens entre le texte et les tableaux.
        
        Args:
            text: Texte à analyser
                
        Returns:
            Liste des références de tableaux trouvées
        """
        table_references = []
        
        # Patrons pour détecter les références aux tableaux
        reference_patterns = [
            r'(?:tableau|table)\s+(\d+|[IVX]+)',  # ex: "tableau 1" ou "table II"
            r'(?:voir|cf\.?|confère)\s+(?:le\s+)?(?:tableau|table)',  # ex: "voir tableau" ou "cf. table"
            r'(?:illustr(?:é|er))\s+(?:dans|par)\s+(?:le\s+)?(?:tableau|table)',  # ex: "illustré dans le tableau"
            r'(?:présenté|indiqué|montré)\s+(?:dans|au)\s+(?:le\s+)?(?:tableau|table)',  # ex: "présenté dans le tableau"
        ]
        
        for pattern in reference_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                table_number = None
                # Extraire le numéro du tableau si présent
                if match.groups() and len(match.groups()) > 0:
                    table_number = match.group(1)
                
                table_references.append({
                    "start": match.start(),
                    "end": match.end(),
                    "text": match.group(0),
                    "table_number": table_number
                })
    
        return table_references

    def _clean_table_content(self, table_content: str) -> str:
        """
        Nettoie et normalise le contenu du tableau pour une meilleure
        lisibilité et récupération.
        
        Args:
            table_content: Contenu brut du tableau
                
        Returns:
            Contenu du tableau nettoyé et formaté
        """
        # Supprimer les lignes vides
        lines = [line for line in table_content.split('\n') if line.strip()]
        
        # Détecter si c'est un tableau ASCII avec |
        if any('|' in line for line in lines):
            # Normaliser l'espacement dans les cellules
            normalized_lines = []
            for line in lines:
                # Ignorer les lignes de séparation (+--)
                if re.match(r'^[\s+\-=|]+$', line):
                    continue
                # Normaliser les espaces entre les cellules
                cells = [cell.strip() for cell in line.split('|')]
                normalized_lines.append('| ' + ' | '.join(cells) + ' |')
            
            return '\n'.join(normalized_lines)
        
        # Pour les tableaux alignés par espaces
        else:
            # Essayer de détecter les colonnes en analysant l'espacement
            positions = []
            for line in lines[:5]:  # Utiliser les premières lignes pour détecter la structure
                pos = 0
                for char in line:
                    if char != ' ' and pos > 0 and line[pos-1] == ' ':
                        positions.append(pos)
                    pos += 1
            
            # Compter les occurrences de chaque position
            from collections import Counter
            pos_counts = Counter(positions)
            
            # Trouver les positions les plus fréquentes (colonnes probables)
            common_positions = [pos for pos, count in pos_counts.items() if count >= 2]
            common_positions.sort()
            
            # Si on ne peut pas identifier de colonnes claires, retourner tel quel
            if len(common_positions) < 2:
                return table_content
            
            # Reformater en tableau avec séparateurs |
            formatted_lines = []
            for line in lines:
                if not line.strip():
                    continue
                    
                cells = []
                prev_pos = 0
                for pos in common_positions + [len(line)]:
                    if pos > prev_pos:
                        cell = line[prev_pos:pos].strip()
                        cells.append(cell)
                    prev_pos = pos
                
                formatted_lines.append('| ' + ' | '.join(cells) + ' |')
            
            return '\n'.join(formatted_lines)

    def _fallback_chunk_text(self, text: str, document_id: str, page_number: int) -> List[Dict[str, Any]]:
        """
        Méthode de secours pour découper le texte si tiktoken n'est pas disponible.
        Utilise une approche simple basée sur les mots.
        """
        chunks = []
        words = text.split()
        
        if not words:
            return []
            
        start = 0
        
        while start < len(words):
            end = min(start + self.max_tokens, len(words))
            
            # Créer le chunk de texte
            chunk_text = ' '.join(words[start:end])
            
            # Générer un ID unique pour le chunk
            chunk_id = hashlib.md5(
                f"{document_id}_{page_number}_{start}".encode()
            ).hexdigest()
            
            # Créer les métadonnées du chunk
            chunk_data = {
                "text": chunk_text,
                "metadata": {
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "page_number": page_number,
                    "start_word": start,
                    "end_word": end,
                    "word_count": end - start,
                    "extraction_method": "fallback_word_tokenized"
                }
            }
            
            chunks.append(chunk_data)
            
            # Avancer avec chevauchement
            start += self.max_tokens - self.overlap_tokens
            
            if start >= len(words) or (end - start) < self.overlap_tokens // 2:
                break
        
        return chunks

    # Les autres méthodes de la classe restent inchangées
    def generate_document_id(self, pdf_path: str) -> str:
        """Génère un identifiant unique pour un document basé sur son chemin et ses attributs."""
        try:
            file_stat = os.stat(pdf_path)
            content_hash = hashlib.md5(
                f"{pdf_path}_{file_stat.st_size}_{file_stat.st_mtime}".encode()
            ).hexdigest()
            return content_hash
        except Exception as e:
            logger.error(f"Erreur lors de la génération de l'ID pour {pdf_path}: {e}")
            # Fallback simple en cas d'erreur
            return hashlib.md5(pdf_path.encode()).hexdigest()

    def preprocess_text(self, text: str) -> str:
        """
        Nettoie et normalise le texte avant l'indexation.
        """
        if not text:
            return ""
        
        # Remplacer les sauts de ligne multiples par un espace
        cleaned = re.sub(r'\n+', ' ', text)
        
        # Supprimer les espaces multiples et normaliser
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Traiter les séquences de nombres mélangées (avec ou sans points, avec point-virgule)
        # Cette expression complexe détecte les séquences de nombres suivis de différents séparateurs
        # \d+\.?\s+ : un nombre suivi ou non d'un point, puis d'un espace
        # [\s;,]* : suivi éventuellement d'un espace, point-virgule ou virgule
        cleaned = re.sub(r'(\d+\.?\s+)([\s;,]*\d+\.?\s+)+(?=\w)', r'\1', cleaned)

        # Nettoyer les séquences standard de numéros avec points
        cleaned = re.sub(r'(\d+\.\s+)(\d+\.\s+)+', r'\1', cleaned)

        # Correction des tirets suivis d'un espace
        cleaned = re.sub(r'-\s+', '- ', cleaned)
        
        # Supprimer les caractères de contrôle
        cleaned = re.sub(r'[\x00-\x1F\x7F]', '', cleaned)
        
        # Normaliser les guillemets et apostrophes
        cleaned = cleaned.replace(''', "'").replace(''', "'").replace('"', '"').replace('"', '"')
        
        # Supprimer les espaces au début et à la fin
        cleaned = cleaned.strip()
        
        return cleaned

    def extract_document_metadata(self, doc, pdf_path: str) -> Dict[str, Any]:
        """
        Extrait les métadonnées d'un document PDF.
        
        Args:
            doc: Document PyMuPDF
            pdf_path: Chemin du fichier PDF
            
        Returns:
            Dictionnaire des métadonnées du document
        """
        # Récupérer les métadonnées du document
        doc_info = doc.metadata
        
        # Obtenir le nom de fichier sans le chemin complet
        filename = os.path.basename(pdf_path)
        
        # Créer un dictionnaire de métadonnées avec le nom de fichier mis en évidence
        return {
            "filename": filename,
            "path": pdf_path,
            "display_name": filename,  # Champ dédié pour l'affichage
            "title": doc_info.get("title", filename),  # Utiliser le nom du fichier si pas de titre
            "author": doc_info.get("author", ""),
            "subject": doc_info.get("subject", ""),
            "keywords": doc_info.get("keywords", ""),
            "creation_date": doc_info.get("creationDate", ""),
            "modification_date": doc_info.get("modDate", ""),
            "page_count": len(doc),
            "file_size_bytes": os.path.getsize(pdf_path),
            "processed_date": datetime.now().isoformat()
        }

    def save_chunks_metadata(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Sauvegarde les métadonnées des chunks individuellement.
        """
        try:
            # Créer le répertoire de métadonnées de chunks
            chunks_metadata_dir = os.path.join(self.metadata_path, "chunks")
            os.makedirs(chunks_metadata_dir, exist_ok=True)
            
            logger.info(f"📂 Sauvegarde des métadonnées de {len(chunks)} chunks")
            
            for i, chunk in enumerate(chunks):
                try:
                    chunk_id = chunk["metadata"]["chunk_id"]
                    chunk_file = os.path.join(chunks_metadata_dir, f"{chunk_id}.json")
                    
                    with open(chunk_file, 'w', encoding='utf-8') as f:
                        json.dump(chunk, f, ensure_ascii=False, indent=2)
                    
                    # Log tous les 100 chunks pour éviter de spammer
                    if i % 100 == 0:
                        logger.info(f"💾 {i}/{len(chunks)} chunks sauvegardés")
                except Exception as e:
                    logger.error(f"❌ Erreur lors de la sauvegarde d'un chunk: {e}")
            
            logger.info(f"✅ Tous les chunks ont été sauvegardés ({len(chunks)} au total)")
        except Exception as e:
            logger.error(f"❌ Erreur lors de la sauvegarde des métadonnées des chunks: {e}")

# Fonction utilitaire pour tester le module
def test_pdf_processor(pdf_path):
    processor = PDFProcessor(max_tokens=500, overlap_tokens=100)
    chunks, metadata = processor.process_pdf(pdf_path)
    
    print(f"Document: {metadata.get('filename')}")
    print(f"Pages: {metadata.get('page_count')}")
    print(f"Chunks générés: {len(chunks)}")
    
    if chunks:
        print("\nExemple de chunk:")
        sample_chunk = chunks[0]
        print(f"ID: {sample_chunk['metadata']['chunk_id']}")
        print(f"Page: {sample_chunk['metadata']['page_number']}")
        print(f"Extraction: {sample_chunk['metadata']['extraction_method']}")
        print(f"Texte: {sample_chunk['text'][:100]}...")

if __name__ == "__main__":
    test_pdf_processor("/home/mea/Documents/modelAi/data/CIREX-2025-FR.pdf")