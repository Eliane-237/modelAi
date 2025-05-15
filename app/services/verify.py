import sys
import os
import json

# Ajouter le chemin du projet au PYTHONPATH
project_root = "/home/mea/Documents/modelAi"
sys.path.append(project_root)

from app.services.milvus_service import MilvusService

def detailed_milvus_inspection():
    # Initialiser le service Milvus
    milvus_service = MilvusService("documents_collection", dim=1024)
    
    try:
        # Charger la collection
        milvus_service.collection.load()
        
        # Afficher le schéma de la collection pour comprendre sa structure
        print("🔍 Schéma de la collection :")
        schema = milvus_service.collection.schema
        for field in schema.fields:
            print(f"- Champ: {field.name}, Type: {field.dtype}")
        
        # Récupérer tous les champs disponibles
        all_fields = [field.name for field in schema.fields]
        print("\n📋 Champs disponibles :", all_fields)
        
        # Effectuer une requête pour récupérer les données
        print("\n🔬 Tentative de requête :")
        query_expression = "page_number >= 0"  # Une condition qui devrait être vraie pour tous les documents
        
        try:
            query_result = milvus_service.collection.query(
                expr=query_expression,
                output_fields=all_fields,
                limit=10
            )
            
            print(f"\n📊 Nombre de résultats : {len(query_result)}")
            
            for i, result in enumerate(query_result, 2):
                print(f"\n{i}. Résultat :")
                for field in all_fields:
                    try:
                        value = result.get(field, "N/A")
                        print(f"   {field}: {str(value)[:512]}...")
                    except Exception as field_err:
                        print(f"   Erreur pour {field}: {field_err}")
        
        except Exception as query_err:
            print(f"❌ Erreur lors de la requête : {query_err}")
        
    except Exception as e:
        print(f"❌ Erreur lors de l'inspection de la collection : {e}")
    finally:
        # Libérer la collection
        try:
            milvus_service.collection.release()
        except:
            pass

def verify_milvus_insertion():
    """
    Vérification détaillée du processus d'insertion.
    """
    from app.services.pdf_service import PDFService
    from app.services.embedding_service import EmbeddingService
    
    # Initialiser les services
    embedding_service = EmbeddingService()
    milvus_service = MilvusService("documents_collection", dim=1024)
    
    # Initialiser le service PDF
    pdf_service = PDFService(
        embedding_service,
        milvus_service,
        max_tokens=500,
        overlap_tokens=100
    )
    
    # Charger les documents PDF
    pdf_files = pdf_service.load_documents()
    
    print("🔍 Vérification du processus d'insertion :")
    print(f"Nombre de fichiers PDF : {len(pdf_files)}")
    
    for pdf_path in pdf_files:
        print(f"\n📄 Analyse du document : {os.path.basename(pdf_path)}")
        
        try:
            # Traiter le PDF et obtenir les chunks
            chunks, doc_metadata = pdf_service.pdf_processor.process_pdf(pdf_path, force_reprocess=True)
            
            print(f"📦 Nombre de chunks générés : {len(chunks)}")
            
            # Générer les embeddings
            texts = [chunk['text'] for chunk in chunks]
            embeddings = embedding_service.generate_embeddings(texts)
            
            print(f"🧩 Nombre d'embeddings générés : {len(embeddings)}")
            print(f"Dimension des embeddings : {len(embeddings[0])}")
            
            # Préparer les métadonnées
            metadata_list = [chunk['metadata'] for chunk in chunks]
            
            # Insérer dans Milvus
            milvus_service.insert_documents_with_metadata(embeddings, texts, metadata_list)
            
        except Exception as e:
            print(f"❌ Erreur lors du traitement de {pdf_path}: {e}")

if __name__ == "__main__":
    # Choisissez l'une des deux méthodes
    detailed_milvus_inspection()
    # verify_milvus_insertion()