import logging
import re
import io
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import pytesseract
from typing import Tuple, Optional, List
from skimage.filters import threshold_otsu

# Configuration du logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Spécifiez le chemin de l'exécutable Tesseract si nécessaire
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

# Liste des termes juridiques camerounais pour améliorer la détection OCR
LEGAL_TERMS = [
    "impôt", "taxe", "revenu", "contribuable", "déclaration", "irpp", "tva", 
    "article", "chapitre", "section", "alinéa", "paragraphe", "titre", 
    "loi", "décret", "arrêté", "circulaire", "code", "cameroun", 
    "fiscalité", "exonération", "redevance", "cgdci", "dgi", "cgi"
]

def preprocess_image_for_ocr(image):
    """
    Prétraite une image pour améliorer les résultats de l'OCR.
    Optimisé pour les documents juridiques et fiscaux camerounais.
    Gestion spéciale pour les images extrêmement larges.
    
    Args:
        image: Image PIL
        
    Returns:
        Image prétraitée
    """
    try:
        # Vérifier et logger les dimensions originales
        original_width, original_height = image.size
        logger.info(f"Dimensions originales de l'image: {original_width}x{original_height}")
        
        # SOLUTION POUR IMAGES TRÈS LARGES: Définir une limite maximale absolue
        MAX_DIMENSION = 3000  # Limite stricte pour Tesseract
        
        # Si l'image est trop grande dans une dimension, la redimensionner
        if original_width > MAX_DIMENSION or original_height > MAX_DIMENSION:
            # Calculer le ratio tout en préservant les proportions
            width_ratio = MAX_DIMENSION / original_width if original_width > MAX_DIMENSION else 1
            height_ratio = MAX_DIMENSION / original_height if original_height > MAX_DIMENSION else 1
            scale_factor = min(width_ratio, height_ratio)
            
            # Calculer les nouvelles dimensions
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            
            logger.warning(f"Image trop grande ({original_width}x{original_height}) redimensionnée à {new_width}x{new_height}")
            
            # Pour les images extrêmement larges (plus de 30000 pixels), utiliser NEAREST pour économiser la mémoire
            if original_width > 30000 or original_height > 30000:
                image = image.resize((new_width, new_height), Image.NEAREST)
            else:
                # Pour les images plus petites, utiliser LANCZOS pour une meilleure qualité
                image = image.resize((new_width, new_height), Image.LANCZOS)
        
        # Convertir en niveaux de gris après redimensionnement
        if image.mode != 'L':
            image = image.convert('L')
        
        # Pour les images plus petites, améliorer la résolution si nécessaire
        elif image.width < 1000 or image.height < 1000:
            factor = 1500 / min(image.width, image.height)
            new_size = (int(image.width * factor), int(image.height * factor))
            image = image.resize(new_size, Image.LANCZOS)
        
        # OPTIMISATION: Traitement différent selon la taille de l'image
        current_width, current_height = image.size
        
        if current_width > 2000 or current_height > 2000:
            # Pour les images encore grandes après redimensionnement, simplifier le traitement
            logger.info("Image encore grande après redimensionnement, traitement simplifié")
            
            # Augmenter légèrement le contraste seulement
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)  # Contraste modéré
            
            # Binarisation simple basée sur la moyenne
            threshold = np.mean(np.array(image)) * 0.9
            binary_img = image.point(lambda p: 255 if p > threshold else 0)
            
            return binary_img
        else:
            # Traitement complet pour les images de taille raisonnable
            # Augmenter le contraste
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)
            
            # Réduire le bruit
            image = image.filter(ImageFilter.MedianFilter(size=3))
            
            # Binarisation adaptative pour les documents juridiques
            img_array = np.array(image)
            
            # Utiliser la méthode d'Otsu pour la binarisation
            try:
                threshold = threshold_otsu(img_array)
            except Exception:
                # Méthode de secours si skimage n'est pas disponible
                threshold = np.mean(img_array) * 0.9
            
            # Appliquer le seuil
            binary_img = image.point(lambda p: 255 if p > threshold else 0)
            
            # Appliquer une légère dilatation pour améliorer la reconnaissance des caractères
            binary_img = binary_img.filter(ImageFilter.MaxFilter(size=3))
            
            return binary_img
            
    except Exception as e:
        logger.error(f"Erreur critique lors du prétraitement: {str(e)}")
        
        # SOLUTION DE DERNIER RECOURS: redimensionnement brutal en cas d'erreur
        try:
            logger.warning("Tentative de redimensionnement d'urgence")
            
            # Créer une image de secours de taille limitée
            emergency_size = (2000, 2000)
            if hasattr(image, 'width') and hasattr(image, 'height'):
                if image.width > image.height:
                    emergency_size = (2000, int(2000 * image.height / image.width))
                else:
                    emergency_size = (int(2000 * image.width / image.height), 2000)
            
            # Redimensionner avec la méthode la plus simple
            return image.resize(emergency_size, Image.NEAREST).convert('L')
        
        except Exception as fallback_error:
            logger.critical(f"Échec total du prétraitement: {fallback_error}")
            
            # En dernier recours, créer une image blanche
            from PIL import Image
            return Image.new('L', (1000, 1000), 255)


def post_process_ocr_text(text: str) -> str:
    """
    Corrige les erreurs OCR courantes dans les textes juridiques et fiscaux camerounais.
    
    Args:
        text: Texte OCR brut
        
    Returns:
        Texte corrigé
    """
    if not text:
        return ""
    
    # Corrections courantes pour documents juridiques camerounais
    corrections = {
        # Remplacements d'erreurs OCR communes dans les documents juridiques
        "lmpôt": "Impôt",
        "l'lmpôt": "l'Impôt",
        "Articie": "Article",
        "ARTlCLE": "ARTICLE",
        "Chapiire": "Chapitre",
        "CHAPlTRE": "CHAPITRE",
        "Seciion": "Section",
        "SECTlON": "SECTION",
        "Aliriéa": "Alinéa",
        "ALlNEA": "ALINEA",
        "Títre": "Titre",
        "TlTRE": "TITRE",
        "déciaration": "déclaration",
        "IRFP": "IRPP",
        "lRPP": "IRPP",
        "camérouna": "camerounai",
        "flscal": "fiscal",
        "Flscal": "Fiscal",
        "TVÀ": "TVA",
        "IS.": "IS",
        "l.S.": "IS",
        "l'Import": "l'import",
        "l'Exporl": "l'Export",
        "l.R.P.P.": "IRPP",
        "camerounals": "camerounais",
        
        # Séparateurs
        ",:": ":",
        ";:": ":",
        ",;": ";",
        ".,": ".",
        
        # Espaces et ponctuations
        "  ": " ",
        " ,": ",",
        " .": ".",
        " ;": ";",
        " :": ":",
        " )": ")",
        "( ": "(",
        "« ": "«",
        " »": "»",
    }
    
    # Appliquer les corrections
    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)
    
    # Nettoyer les retours à la ligne excessifs
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Corriger les numéros d'articles (ex: Article l devrait être Article 1)
    text = re.sub(r'(?i)article\s+l', 'Article 1', text)
    text = re.sub(r'(?i)article\s+l(\d+)', r'Article 1\1', text)
    
    # Traiter les erreurs courantes pour les nombres
    text = re.sub(r'(\d)\.(\d)', r'\1,\2', text)  # Format français pour les décimales
    
    # Corriger les erreurs OCR sur les pourcentages
    text = re.sub(r'(\d+)o/o', r'\1%', text)
    text = re.sub(r'(\d+)o/', r'\1%', text)
    
    # Remplacer les caractères non imprimables et caractères spéciaux erronés
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
    
    # Correction des références légales
    text = re.sub(r'(l|I)oi n(°|º|o)(\s*\d+)', r'Loi n°\3', text)
    text = re.sub(r'(d|D)écret n(°|º|o)(\s*\d+)', r'Décret n°\3', text)
    
    # Corriger la mise en forme des listes numérotées
    text = re.sub(r'(\d+)\)\s*', r'\1) ', text)
    
    # Corriger les termes spécifiques juridiques camerounais
    for term in LEGAL_TERMS:
        # Créer une regex qui ignore la casse et peut gérer quelques erreurs courantes de substitution
        term_pattern = ''.join([f'[{c.lower()}{c.upper()}1lI]' if c.lower() in 'il' else 
                                f'[{c.lower()}{c.upper()}0O]' if c.lower() == 'o' else 
                                f'[{c.lower()}{c.upper()}]' for c in term])
        # Remplacer par le terme correct
        text = re.sub(fr'\b{term_pattern}\b', term, text)
    
    return text.strip()


def is_likely_scanned(text: str, page=None) -> bool:
    """
    Détermine si un document est probablement scanné en analysant le texte et/ou la page.
    
    Args:
        text: Texte extrait
        page: Objet page de PyMuPDF (optionnel)
        
    Returns:
        True si le document semble être scanné
    """
    # Si peu ou pas de texte extrait, c'est probablement scanné
    if not text or len(text.strip()) < 50:
        return True
    
    # Vérifier le ratio texte/taille de la page si disponible
    if page:
        char_density = len(text) / (page.rect.width * page.rect.height)
        if char_density < 0.005:  # Seuil encore plus strict
            return True
        
        # Vérifier le nombre d'images sur la page
        image_list = page.get_images(full=True)
        if len(image_list) > 0 and len(text) < 200:
            return True
    
    # Vérifier les caractéristiques textuelles d'un scan
    ocr_artifacts = ['�', '|', '/', '\\', '*', '#', '@', '~', '±', '§', '¶']
    artifact_count = sum(text.count(artifact) for artifact in ocr_artifacts)
    if artifact_count > 5 or (artifact_count > 0 and len(text) < 200):
        return True
    
    # Détecter les erreurs OCR typiques
    ocr_error_patterns = [
        r'[a-z][A-Z]{2,}[a-z]',  # Lettres majuscules au milieu des mots
        r'\d[a-zA-Z]\d',         # Lettre entre deux chiffres
        r'[a-zA-Z]\d[a-zA-Z]',   # Chiffre au milieu d'un mot
        r'[lI]mpo[rt]',          # Erreurs sur 'impôt'
        r'[cC]amer[0oO]un'       # Erreurs sur 'cameroun'
    ]
    
    error_count = 0
    for pattern in ocr_error_patterns:
        matches = re.findall(pattern, text)
        error_count += len(matches)
    
    if error_count > 3 or (error_count > 0 and len(text) < 300):
        return True
    
    # Vérifier si le document contient du texte substantiel
    # Les documents scannés mal OCRisés ont souvent un texte fragmenté
    if len(text.split()) < 10 and len(text) > 50:
        return True
    
    return False


def extract_text_from_pdf_page(page, ocr_language='fra', perform_ocr=False):
    """
    Extrait le texte d'une page PDF avec OCR si nécessaire.
    Optimisée pour les documents juridiques camerounais.
    Solution améliorée pour les images extrêmement larges.
    
    Args:
        page: Page PyMuPDF
        ocr_language: Code de langue pour l'OCR
        perform_ocr: Forcer l'OCR même si du texte est disponible
        
    Returns:
        Tuple (texte extrait, méthode d'extraction)
    """
    # Essayer d'abord l'extraction de texte standard
    text = page.get_text()
    
    # Vérifier si le document est probablement scanné
    scanned = is_likely_scanned(text, page)
    
    # Si peu ou pas de texte, ou si OCR forcé, ou si document scanné, utiliser OCR
    if perform_ocr or scanned or len(text.strip()) < 50:
        ocr_text = ""
        
        # APPROCHE 1: Traiter directement la page entière au lieu des images individuelles
        try:
            logger.info("Tentative d'OCR sur la page entière...")
            
            # Adapter la résolution en fonction de la taille de la page
            page_rect = page.rect
            page_width, page_height = page_rect.width, page_rect.height
            
            # Réduire la résolution pour les pages très grandes
            adaptive_dpi = 200  # DPI par défaut
            
            if page_width > 1000 or page_height > 1000:
                adaptive_dpi = 150  # Réduire pour grandes pages
            if page_width > 2000 or page_height > 2000:
                adaptive_dpi = 100  # Réduire encore pour très grandes pages
            
            logger.info(f"OCR page complète: dimensions {page_width}x{page_height}, DPI={adaptive_dpi}")
            
            # Rendre la page avec DPI adaptatif
            pix = page.get_pixmap(dpi=adaptive_dpi)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Prétraitement avec notre fonction améliorée
            img = preprocess_image_for_ocr(img)
            
            # Configuration OCR adaptée
            custom_config = '--oem 1 --psm 1 -l ' + ocr_language
            
            # Appliquer OCR
            page_ocr_text = pytesseract.image_to_string(img, config=custom_config)
            
            # Post-traitement du texte OCR
            page_ocr_text = post_process_ocr_text(page_ocr_text)
            
            # Si du texte a été extrait, l'utiliser directement
            if page_ocr_text.strip():
                logger.info(f"OCR page entière réussi: {len(page_ocr_text)} caractères extraits")
                return page_ocr_text, "ocr_page"
                
        except Exception as page_error:
            logger.warning(f"Erreur lors de l'OCR sur la page entière: {page_error}")
        
        # APPROCHE 2: Si l'approche page entière échoue, essayer l'extraction des images individuelles
        # mais avec gestion spéciale pour les images très larges
        image_list = page.get_images(full=True)
        
        if image_list:
            logger.info(f"Tentative d'OCR sur {len(image_list)} images individuelles...")
            
            for img_index, img in enumerate(image_list):
                try:
                    # Extraire les métadonnées de l'image d'abord
                    xref = img[0]
                    base_image = page.parent.extract_image(xref)
                    
                    # VÉRIFICATION PRÉLIMINAIRE: Vérifier les dimensions si disponibles
                    if "width" in base_image and "height" in base_image:
                        width, height = base_image["width"], base_image["height"]
                        
                        logger.info(f"Image {img_index+1}/{len(image_list)}: dimensions {width}x{height}")
                        
                        # Sauter les images excessivement grandes pour éviter le traitement direct
                        if width > 50000 or height > 50000:
                            logger.warning(f"Image {img_index} extrêmement large, traitement spécial requis")
                            # Ne pas essayer l'OCR direct, traiter plus tard avec approche par tuiles
                            continue
                    
                    # Extraire et charger l'image
                    image_bytes = base_image["image"]
                    image = Image.open(io.BytesIO(image_bytes))
                    
                    # Prétraitement pour améliorer l'OCR
                    image = preprocess_image_for_ocr(image)
                    
                    # Configuration OCR adaptée aux documents juridiques
                    custom_config = '--oem 1 --psm 6 -l ' + ocr_language
                    
                    # Pour les grandes images, utiliser PSM 1 (analyse de page complète)
                    if image.width > 1000:
                        custom_config = '--oem 1 --psm 1 -l ' + ocr_language
                    
                    # Tenter l'OCR avec gestion d'erreur
                    try:
                        extracted_text = pytesseract.image_to_string(image, config=custom_config)
                    except Exception as ocr_error:
                        logger.warning(f"Erreur OCR: {ocr_error} - tentative avec config simplifiée")
                        
                        # Essayer avec une configuration plus simple
                        try:
                            extracted_text = pytesseract.image_to_string(image, config='--psm 3')
                        except Exception:
                            logger.error(f"Échec OCR total pour l'image {img_index}")
                            continue
                    
                    # Post-traitement du texte OCR
                    extracted_text = post_process_ocr_text(extracted_text)
                    
                    # Ajouter au texte OCR global
                    ocr_text += extracted_text + " "
                    
                except Exception as e:
                    logger.warning(f"Erreur lors de l'OCR sur l'image {img_index}: {e}")
        
        # APPROCHE 3: Si nous n'avons toujours pas de texte, essayer l'approche par tuiles
        # pour les images très larges
        if not ocr_text.strip() and (perform_ocr or scanned):
            try:
                # Approche par tuiles pour pages très grandes
                logger.info("Tentative d'approche par tuiles pour les grandes images...")
                
                # Fonction pour traiter par tuiles - à implémenter ou intégrer ici
                def process_large_image_by_tiles(image, tile_size=1000):
                    """
                    Traite une image large en la découpant en tuiles plus petites.
                    """
                    width, height = image.size
                    logger.info(f"Traitement par tuiles: image {width}x{height}, tuile {tile_size}px")
                    
                    # Calculer le nombre de tuiles
                    cols = (width + tile_size - 1) // tile_size
                    rows = (height + tile_size - 1) // tile_size
                    
                    combined_text = ""
                    
                    # Traiter chaque tuile
                    for row in range(rows):
                        for col in range(cols):
                            try:
                                # Calculer les coordonnées de la tuile
                                left = col * tile_size
                                upper = row * tile_size
                                right = min(left + tile_size, width)
                                lower = min(upper + tile_size, height)
                                
                                # Extraire la tuile
                                tile = image.crop((left, upper, right, lower))
                                
                                # Traiter uniquement les tuiles non vides
                                if tile.getbbox():  # Vérifie que la tuile n'est pas vide
                                    # Prétraiter la tuile
                                    processed_tile = preprocess_image_for_ocr(tile)
                                    
                                    # OCR avec configuration simple
                                    tile_text = pytesseract.image_to_string(
                                        processed_tile, 
                                        config=f'--oem 1 --psm 6 -l {ocr_language}'
                                    )
                                    
                                    # Ajouter au texte combiné
                                    if tile_text.strip():
                                        combined_text += tile_text + " "
                            except Exception as tile_error:
                                logger.warning(f"Erreur tuile ({row},{col}): {tile_error}")
                    
                    return combined_text
                
                # Pour les images individuelles extrêmement larges
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        base_image = page.parent.extract_image(xref)
                        
                        # Vérifier si l'image est extrêmement large
                        if "width" in base_image and "height" in base_image:
                            width, height = base_image["width"], base_image["height"]
                            
                            if width > 30000 or height > 30000:
                                # Traiter cette image par tuiles
                                logger.info(f"Traitement par tuiles de l'image {img_index}: {width}x{height}")
                                
                                # Charger l'image
                                image_bytes = base_image["image"]
                                image = Image.open(io.BytesIO(image_bytes))
                                
                                # Redimensionner légèrement pour accélérer le traitement
                                if width > 50000 or height > 50000:
                                    scale = 0.2  # Réduction drastique pour images énormes
                                    image = image.resize(
                                        (int(width * scale), int(height * scale)), 
                                        Image.NEAREST
                                    )
                                
                                # Traiter par tuiles
                                tile_text = process_large_image_by_tiles(image)
                                
                                if tile_text.strip():
                                    ocr_text += tile_text + " "
                    except Exception as e:
                        logger.warning(f"Erreur traitement tuiles image {img_index}: {e}")
                
                # Si toujours pas de texte, traiter la page entière par tuiles
                if not ocr_text.strip():
                    logger.info("Dernier essai: page entière par tuiles à basse résolution")
                    
                    # Rendre la page à basse résolution
                    pix = page.get_pixmap(dpi=72)  # Très basse résolution
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    
                    # Traiter par tuiles
                    page_tile_text = process_large_image_by_tiles(img)
                    
                    if page_tile_text.strip():
                        ocr_text = page_tile_text
            
            except Exception as e:
                logger.error(f"Erreur lors de l'approche par tuiles: {e}")
        
        # Si OCR a produit du texte, l'utiliser
        if ocr_text.strip() and (not text.strip() or scanned):
            return ocr_text, "ocr"
    
    # Retourner le texte standard si tout échoue ou s'il n'y a pas besoin d'OCR
    return text, "text"


def analyze_document_language(text: str) -> str:
    """
    Détermine la langue du document basée sur une analyse plus fine.
    """
    # Mots-clés français juridiques
    french_keywords = ['le', 'la', 'les', 'un', 'une', 'des', 'et', 'ou',
                     'pour', 'par', 'sur', 'dans', 'avec', 'est', 'sont',
                     'loi', 'décret', 'article', 'disposition', 'texte']
    
    # Mots-clés anglais juridiques
    english_keywords = ['the', 'a', 'an', 'of', 'to', 'in', 'and', 'or',
                     'for', 'by', 'on', 'with', 'is', 'are', 'was',
                     'law', 'decree', 'article', 'provision', 'text']
    
    # Termes juridiques spécifiques au Cameroun
    french_legal = ['impôt', 'loi', 'décret', 'article', 'cameroun', 'fiscal']
    english_legal = ['tax', 'law', 'decree', 'article', 'cameroon', 'fiscal']
    
    # Compter avec une pondération
    french_count = sum(3 if word in french_legal else 1 
                     for word in french_keywords if word in text.lower().split())
    english_count = sum(3 if word in english_legal else 1 
                      for word in english_keywords if word in text.lower().split())
    
    # Utiliser une méthode plus robuste
    if french_count > english_count * 1.2:  # Seuil avec marge
        return 'fra'
    elif english_count > french_count * 1.2:
        return 'eng'
    else:
        # Analyse plus approfondie pour les cas ambigus
        # [Implémentation d'analyse plus détaillée]
        return 'fra'  # Par défaut français

def batch_process_images(images: List[Image.Image], ocr_language='fra') -> str:
    """
    Traite un lot d'images pour extraire et combiner le texte avec OCR.
    Utile pour les documents multi-pages.
    
    Args:
        images: Liste d'images PIL
        ocr_language: Code de langue pour l'OCR
        
    Returns:
        Texte combiné extrait
    """
    combined_text = ""
    
    for i, image in enumerate(images):
        try:
            logger.info(f"Traitement de l'image {i+1}/{len(images)}")
            
            # Prétraitement
            processed_image = preprocess_image_for_ocr(image)
            
            # Configuration OCR adaptée à la taille de l'image
            custom_config = '--oem 1 --psm 6 -l ' + ocr_language
            if image.width > 1000:
                custom_config = '--oem 1 --psm 1 -l ' + ocr_language
            
            # Extraire le texte
            extracted_text = pytesseract.image_to_string(
                processed_image, config=custom_config
            )
            
            # Post-traitement
            processed_text = post_process_ocr_text(extracted_text)
            
            # Ajouter au texte combiné
            combined_text += processed_text + "\n\n"
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement de l'image {i+1}: {e}")
    
    return combined_text.strip()


def enhance_legal_document_ocr(text: str) -> str:
    """
    Améliore spécifiquement la reconnaissance des termes juridiques camerounais.
    
    Args:
        text: Texte OCR à améliorer
        
    Returns:
        Texte amélioré
    """
    # Termes juridiques spécifiques et corrections
    legal_terms_corrections = {
        "Code General des 1mpots": "Code Général des Impôts",
        "Direction Generale des 1mpots": "Direction Générale des Impôts",
        "1mpot sur le Revenu des Personnes Physiques": "Impôt sur le Revenu des Personnes Physiques",
        "1mpot sur les Societes": "Impôt sur les Sociétés",
        "Taxe sur la Valeur Ajoutee": "Taxe sur la Valeur Ajoutée",
        "1mport-Substitution": "Import-Substitution",
        "Declaration d'1mpot": "Déclaration d'Impôt",
        "Code des Douanes": "Code des Douanes",
        "Loi de Finances": "Loi de Finances",
        "Administration Fiscale": "Administration Fiscale",
        "Ministere des Finances": "Ministère des Finances",
        "Circulaire d'Application": "Circulaire d'Application"
    }
    
    # Appliquer les corrections
    for wrong, correct in legal_terms_corrections.items():
        # Créer un pattern insensible à la casse
        pattern = re.compile(re.escape(wrong), re.IGNORECASE)
        text = pattern.sub(correct, text)
    
    # Corriger les formats de nombres selon les conventions françaises
    # (point décimal -> virgule)
    text = re.sub(r'(\d+)\.(\d+)%', r'\1,\2%', text)
    text = re.sub(r'(\d+)\.(\d+)\s*(?:FCFA|XAF|francs)', r'\1,\2 FCFA', text)
    
    # Correction des références à des articles de loi
    text = re.sub(r'art(?:\.|icle)\s*(\d+)', r'Article \1', text, flags=re.IGNORECASE)
    
    return text


if __name__ == "__main__":
    # Test du module
    test_image_path = "test_scan.png"  # Remplacez par un chemin réel
    try:
        from PIL import Image
        test_image = Image.open(test_image_path)
        result = pytesseract.image_to_string(
            preprocess_image_for_ocr(test_image), 
            config='--oem 1 --psm 6 -l fra'
        )
        processed_result = post_process_ocr_text(result)
        
        print("=== Texte extrait brut ===")
        print(result[:500])
        print("\n=== Texte après post-traitement ===")
        print(processed_result[:500])
    except Exception as e:
        print(f"Erreur lors du test: {e}")
   