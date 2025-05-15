import os
import logging
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, Form, Query, BackgroundTasks, FastAPI
from fastapi.responses import FileResponse, JSONResponse
from typing import List, Optional

from app.core.config import get_settings, Settings
from app.core.dependencies import get_pdf_service
from app.services.pdf_service import PDFService
from app.models.schemas import ProcessPDFResponse, DocumentsStatsResponse

# Configuration du logger
logger = logging.getLogger(__name__)

router = APIRouter()
app = FastAPI()

@router.post("/process", response_model=ProcessPDFResponse)
async def process_pdfs(
    background_tasks: BackgroundTasks,
    force_reprocess: bool = Form(False),
    detect_scanned: bool = Form(True),
    pdf_service: PDFService = Depends(get_pdf_service),
    settings: Settings = Depends(get_settings)
):
    """
    Traite tous les PDFs dans le répertoire de données configuré.
    Le traitement s'effectue en arrière-plan.
    """
    try:
        # Lancer le traitement en arrière-plan
        background_tasks.add_task(
            pdf_service.process_pdfs,
            force_reprocess=force_reprocess,
            detect_scanned=detect_scanned
        )
        
        return ProcessPDFResponse(
            status="processing",
            success=True,
            chunks_processed=0,
            message="Le traitement des PDFs a été lancé en arrière-plan. Utilisez /documents/status pour vérifier l'avancement."
        )
    except Exception as e:
        logger.error(f"Erreur lors du lancement du traitement des PDFs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process/{pdf_name}", response_model=ProcessPDFResponse)
async def process_single_pdf(
    pdf_name: str,
    force_reprocess: bool = Form(False),
    detect_scanned: bool = Form(True),
    pdf_service: PDFService = Depends(get_pdf_service),
    settings: Settings = Depends(get_settings)
):
    """
    Traite un seul PDF spécifique par son nom.
    """
    try:
        # Construire le chemin complet vers le PDF
        pdf_path = os.path.join(settings.DATA_PATH, pdf_name)
        
        # Vérifier si le fichier existe
        if not os.path.exists(pdf_path):
            raise HTTPException(status_code=404, detail=f"Le fichier PDF '{pdf_name}' n'existe pas.")
            
        # Traiter le PDF
        document_id = pdf_service.pdf_processor.generate_document_id(pdf_path)
        success = pdf_service.process_pdf(
            pdf_path=pdf_path,
            force_reprocess=force_reprocess,
            detect_scanned=detect_scanned
        )
        
        if success:
            # Récupérer les métadonnées
            metadata_file = os.path.join(settings.METADATA_PATH, f"{document_id}.json")
            chunks_processed = 0
            
            if os.path.exists(metadata_file):
                import json
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    chunks_processed = metadata.get("chunk_count", 0)
                    
            return ProcessPDFResponse(
                status="completed",
                document_id=document_id,
                filename=pdf_name,
                chunks_processed=chunks_processed,
                success=True,
                message=f"Le PDF '{pdf_name}' a été traité avec succès."
            )
        else:
            return ProcessPDFResponse(
                status="failed",
                document_id=document_id,
                filename=pdf_name,
                chunks_processed=0,
                success=False,
                error="Échec du traitement du PDF",
                message=f"Le traitement du PDF '{pdf_name}' a échoué."
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors du traitement du PDF '{pdf_name}': {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload", response_model=ProcessPDFResponse)
async def upload_and_process_pdf(
    file: UploadFile = File(...),
    process_immediately: bool = Form(True),
    force_reprocess: bool = Form(False),
    detect_scanned: bool = Form(True),
    pdf_service: PDFService = Depends(get_pdf_service),
    settings: Settings = Depends(get_settings)
):
    """
    Télécharge un nouveau fichier PDF et optionnellement le traite immédiatement.
    """
    try:
        # Vérifier que c'est bien un PDF
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Le fichier doit être un PDF.")
            
        # Sauvegarder le fichier
        file_path = os.path.join(settings.DATA_PATH, file.filename)
        
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        logger.info(f"Fichier '{file.filename}' téléchargé avec succès à '{file_path}'")
        
        # Traiter immédiatement si demandé
        if process_immediately:
            success = pdf_service.process_pdf(
                pdf_path=file_path,
                force_reprocess=True,  # Toujours forcer pour les nouveaux fichiers
                detect_scanned=detect_scanned
            )
            
            document_id = pdf_service.pdf_processor.generate_document_id(file_path)
            
            if success:
                # Récupérer les métadonnées
                metadata_file = os.path.join(settings.METADATA_PATH, f"{document_id}.json")
                chunks_processed = 0
                
                if os.path.exists(metadata_file):
                    import json
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        chunks_processed = metadata.get("chunk_count", 0)
                
                return ProcessPDFResponse(
                    status="completed",
                    document_id=document_id,
                    filename=file.filename,
                    chunks_processed=chunks_processed,
                    success=True,
                    message=f"Le PDF '{file.filename}' a été téléchargé et traité avec succès."
                )
            else:
                return ProcessPDFResponse(
                    status="upload_success_process_failed",
                    document_id=document_id,
                    filename=file.filename,
                    chunks_processed=0,
                    success=False,
                    error="Échec du traitement",
                    message=f"Le PDF '{file.filename}' a été téléchargé mais son traitement a échoué."
                )
        
        # Si le traitement n'est pas demandé immédiatement
        return ProcessPDFResponse(
            status="uploaded",
            filename=file.filename,
            chunks_processed=0,
            success=True,
            message=f"Le PDF '{file.filename}' a été téléchargé avec succès. Utilisez /documents/process/{file.filename} pour le traiter."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors du téléchargement ou du traitement du PDF: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list", response_model=DocumentsStatsResponse)
async def list_documents(
    pdf_service: PDFService = Depends(get_pdf_service),
    settings: Settings = Depends(get_settings)
):
    """
    Liste tous les documents disponibles avec leurs statistiques.
    """
    try:
        # Charger la liste des PDFs
        pdf_files = pdf_service.load_documents()
        
        # Statistiques globales
        total_documents = len(pdf_files)
        total_chunks = 0
        scanned_documents = 0
        documents_list = []
        
        # Récupérer les informations de chaque document
        for pdf_path in pdf_files:
            document_id = pdf_service.pdf_processor.generate_document_id(pdf_path)
            doc_info = pdf_service.get_document_info(document_id)
            
            if doc_info:
                # Ajouter aux statistiques globales
                if doc_info.get("chunk_count"):
                    total_chunks += doc_info.get("chunk_count", 0)
                
                if doc_info.get("is_scanned", False):
                    scanned_documents += 1
                
                # Enrichir avec quelques informations supplémentaires
                doc_info["filename"] = os.path.basename(pdf_path)
                doc_info["full_path"] = pdf_path
                
                documents_list.append(doc_info)
        
        return DocumentsStatsResponse(
            total_documents=total_documents,
            total_chunks=total_chunks,
            scanned_documents=scanned_documents,
            documents_list=documents_list
        )
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération de la liste des documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/info/{document_id}")
async def get_document_info(
    document_id: str,
    pdf_service: PDFService = Depends(get_pdf_service)
):
    """
    Récupère les informations détaillées sur un document spécifique.
    """
    try:
        doc_info = pdf_service.get_document_info(document_id)
        
        if not doc_info:
            raise HTTPException(status_code=404, detail=f"Document non trouvé: {document_id}")
            
        return doc_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des informations du document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/delete/{document_id}")
async def delete_document(
    document_id: str,
    delete_file: bool = Query(False, description="Supprimer également le fichier PDF"),
    pdf_service: PDFService = Depends(get_pdf_service),
    settings: Settings = Depends(get_settings)
):
    """
    Supprime un document de l'index et optionnellement le fichier PDF.
    """
    try:
        # Récupérer les informations du document
        doc_info = pdf_service.get_document_info(document_id)
        
        if not doc_info:
            raise HTTPException(status_code=404, detail=f"Document non trouvé: {document_id}")
        
        # Supprimer les embeddings de Milvus
        pdf_service.milvus_service.delete_by_document_id(document_id)
        
        # Supprimer les métadonnées
        metadata_file = os.path.join(settings.METADATA_PATH, f"{document_id}.json")
        if os.path.exists(metadata_file):
            os.remove(metadata_file)
        
        # Supprimer les chunks associés
        chunks_dir = os.path.join(settings.METADATA_PATH, "chunks")
        if os.path.exists(chunks_dir):
            for chunk_filename in os.listdir(chunks_dir):
                if chunk_filename.endswith('.json'):
                    try:
                        chunk_path = os.path.join(chunks_dir, chunk_filename)
                        with open(chunk_path, 'r', encoding='utf-8') as f:
                            import json
                            chunk_data = json.load(f)
                            if chunk_data.get("metadata", {}).get("document_id") == document_id:
                                os.remove(chunk_path)
                    except:
                        pass
        
        # Supprimer le fichier PDF si demandé
        if delete_file and "path" in doc_info and os.path.exists(doc_info["path"]):
            os.remove(doc_info["path"])
            file_deleted = True
        else:
            file_deleted = False
        
        return {
            "success": True,
            "document_id": document_id,
            "metadata_deleted": True,
            "file_deleted": file_deleted,
            "message": f"Document {document_id} supprimé avec succès."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la suppression du document: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/upload-multiple")
async def upload_and_process_multiple_pdfs(
    files: List[UploadFile] = File(...),
    process_immediately: bool = Form(True),
    force_reprocess: bool = Form(False),
    detect_scanned: bool = Form(True),
    pdf_service: PDFService = Depends(get_pdf_service),
    settings: Settings = Depends(get_settings)
):
    """
    Télécharge plusieurs fichiers PDF et optionnellement les traite immédiatement.
    """
    results = []
    failed_files = []
    
    try:
        for file in files:
            try:
                # Vérifier que c'est bien un PDF
                if not file.filename.lower().endswith('.pdf'):
                    failed_files.append({
                        "filename": file.filename,
                        "error": "Le fichier doit être un PDF."
                    })
                    continue
                    
                # Sauvegarder le fichier
                file_path = os.path.join(settings.DATA_PATH, file.filename)
                
                with open(file_path, "wb") as f:
                    f.write(await file.read())
                
                logger.info(f"Fichier '{file.filename}' téléchargé avec succès à '{file_path}'")
                
                # Traiter immédiatement si demandé
                if process_immediately:
                    success = pdf_service.process_pdf(
                        pdf_path=file_path,
                        force_reprocess=force_reprocess,
                        detect_scanned=detect_scanned
                    )
                    
                    document_id = pdf_service.pdf_processor.generate_document_id(file_path)
                    
                    if success:
                        # Récupérer les métadonnées
                        metadata_file = os.path.join(settings.METADATA_PATH, f"{document_id}.json")
                        chunks_processed = 0
                        
                        if os.path.exists(metadata_file):
                            import json
                            with open(metadata_file, 'r', encoding='utf-8') as f:
                                metadata = json.load(f)
                                chunks_processed = metadata.get("chunk_count", 0)
                        
                        results.append({
                            "status": "completed",
                            "document_id": document_id,
                            "filename": file.filename,
                            "chunks_processed": chunks_processed,
                            "success": True,
                            "message": f"Le PDF '{file.filename}' a été téléchargé et traité avec succès."
                        })
                    else:
                        results.append({
                            "status": "upload_success_process_failed",
                            "document_id": document_id,
                            "filename": file.filename,
                            "chunks_processed": 0,
                            "success": False,
                            "error": "Échec du traitement",
                            "message": f"Le PDF '{file.filename}' a été téléchargé mais son traitement a échoué."
                        })
                else:
                    # Si le traitement n'est pas demandé immédiatement
                    results.append({
                        "status": "uploaded",
                        "filename": file.filename,
                        "chunks_processed": 0,
                        "success": True,
                        "message": f"Le PDF '{file.filename}' a été téléchargé avec succès. Utilisez /documents/process/{file.filename} pour le traiter."
                    })
                    
            except Exception as e:
                logger.error(f"Erreur lors du traitement de '{file.filename}': {e}")
                failed_files.append({
                    "filename": file.filename,
                    "error": str(e)
                })
        
        return {
            "total_files": len(files),
            "successful_files": len(results),
            "failed_files": len(failed_files),
            "results": results,
            "failures": failed_files
        }
        
    except Exception as e:
        logger.error(f"Erreur globale lors de l'upload multiple: {e}")
        raise HTTPException(status_code=500, detail=str(e))