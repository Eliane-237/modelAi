# Système RAG Juridique Camerounais

## 📋 Description

Ce projet est un système d'assistance juridique intelligent spécialisé dans les lois camerounaises. Il utilise une architecture RAG (Retrieval-Augmented Generation) pour fournir des informations précises et contextualisées sur la législation camerounaise.

## 🚀 Fonctionnalités

- **Recherche sémantique avancée** dans les documents juridiques
- **Extraction et traitement intelligent** des PDF juridiques
- **OCR spécialisé** pour les documents scannés
- **Question-réponse** sur des sujets juridiques camerounais
- **Segmentation sémantique** des textes juridiques (articles, sections, etc.)
- **Reranking optimisé** pour les documents juridiques camerounais
- **API RESTful complète** pour l'intégration avec différentes applications

## 🛠️ Architecture technique

Le système utilise les technologies suivantes:

- **Base vectorielle**: Milvus pour le stockage et la recherche vectorielle
- **Modèle d'embedding**: BGE-M3 pour la transformation des textes en vecteurs
- **LLM**: DeepSeek pour la génération de réponses
- **Backend**: FastAPI pour l'API REST
- **Traitement PDF**: PyMuPDF et OCR spécialisé
- **Reranking**: Algorithmes hybrides adaptés aux textes juridiques

## 📂 Structure du projet

```
project/
├── app/
│   ├── api/
│   │   ├── endpoints/     # Points d'entrée API par domaine fonctionnel
│   │   └── api.py         # Routeur principal de l'API
│   ├── core/              # Configuration et dépendances
│   ├── models/            # Modèles Pydantic pour validation de données
│   ├── services/          # Services métier
│   └── main.py            # Point d'entrée principal
├── requirements.txt
└── startup.py             # Script de démarrage
```

## 🚀 Installation et démarrage

### Prérequis

- Python 3.8+
- Accès au serveur d'embedding à l'adresse `10.100.212.118`
- Accès à une instance Milvus à l'adresse `10.100.212.118:19530`
- Accès au service LLM à l'adresse `10.100.212.118:8001`

### Installation

1. Cloner le dépôt
   ```bash
   git clone <repository-url>
   cd rag-juridique-camerounais
   ```

2. Installer les dépendances
   ```bash
   pip install -r requirements.txt
   ```

3. Démarrer l'API
   ```bash
   python startup.py
   ```

4. Accéder à la documentation de l'API
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

## 📚 Utilisation de l'API

### Traitement de documents

```bash
# Télécharger et traiter un document
curl -X POST "http://localhost:8000/api/documents/upload" \
  -F "file=@document.pdf" \
  -F "process_immediately=true"

# Lister les documents disponibles
curl -X GET "http://localhost:8000/api/documents/list"
```

### Recherche

```bash
# Recherche simple
curl -X GET "http://localhost:8000/api/search/simple?query=Quelles+sont+les+exonérations+fiscales"

# Recherche avancée
curl -X POST "http://localhost:8000/api/search/" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Quelles sont les exonérations fiscales pour les PME?",
    "top_k": 5,
    "use_rerank": true,
    "use_llm_rerank": false,
    "filter": {"filename": "code_general_impots.pdf"}
  }'
```

curl -X POST "http://10.100.212.118:8001/generate"   -H "Content-Type: application/json"   -d '{
    "model": "llama3.2:latest",
    "prompt": "Test du modèle",
    "max_length": 10,
    "temperature": 0.7
  }'
{"generated_text":"Je suis prêt ! Comment puis-je vous"}

### Question-réponse (RAG)

```bash
# Poser une question juridique
curl -X GET "http://localhost:8000/api/rag/question?query=Quels+sont+les+taux+de+TVA+applicables+au+Cameroun"

# Version avancée
curl -X POST "http://localhost:8000/api/rag/question" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Quelles sont les sanctions en cas de fraude fiscale au Cameroun?",
    "use_expansion": true,
    "use_reranking": true,
    "max_results": 3
  }'
```

## 🧩 Services principaux

- **PDFService**: Traitement et indexation des documents
- **SearchService**: Recherche sémantique avec expansion de requête
- **RerankService**: Réordonnement optimisé des résultats
- **RAGSystem**: Génération de réponses basées sur les documents

## 📊 Points forts du système

1. **Chunking sémantique juridique**: Segmentation intelligente préservant la structure des textes légaux
2. **Reranking spécialisé**: Algorithmes de réordonnement optimisés pour le domaine juridique camerounais
3. **Détection d'articles spécifiques**: Identification précise des références légales
4. **Support multilingue**: Traitement du français et de l'anglais (les deux langues officielles)
5. **OCR optimisé**: Reconnaissance de texte spécialisée pour documents juridiques scannés

## 👥 Contributeurs

- MEA

## 🧠 Fonctionnalités juridiques spécialisées

Le système intègre plusieurs fonctionnalités spécifiquement optimisées pour le domaine juridique camerounais :

### Détection intelligente des textes juridiques
- Reconnaissance automatique des articles, sections, chapitres
- Identification des références légales croisées
- Traitement adapté des documents scannés (très fréquents pour les textes juridiques anciens)

### Expansion de requête juridique
- Prise en compte des termes juridiques camerounais et leurs synonymes
- Expansion des codes et acronymes juridiques (IRPP, TVA, IS, CGI, etc.)
- Reconnaissance des variantes orthographiques des termes juridiques

### Reranking adapté au droit camerounais
- Scoring spécial pour les correspondances d'articles
- Priorisation des textes juridiques officiels
- Détection de pertinence juridique basée sur le contexte

### Génération de réponses explicatives
- Formulation de réponses adaptées au type de question juridique
- Citation précise des sources et références légales
- Explication des dispositions légales dans un langage accessible

## 🔄 Cycle de vie des requêtes

1. **Requête utilisateur** - L'utilisateur pose une question sur un sujet juridique camerounais
2. **Expansion de requête** - Le système génère des variantes pertinentes de la requête
3. **Recherche vectorielle** - Les documents pertinents sont récupérés via Milvus
4. **Reranking juridique** - Les résultats sont réordonnés selon leur pertinence juridique
5. **Formatage du contexte** - Les extraits sont organisés de manière cohérente
6. **Génération de réponse** - Le LLM génère une réponse basée sur les sources
7. **Citation des sources** - La réponse inclut les références aux textes juridiques

## 🔄 Mise à jour des données

Pour maintenir la base de connaissance à jour :

```bash
# Télécharger et traiter un nouveau document
curl -X POST "http://localhost:8000/api/documents/upload" \
  -F "file=@nouveau_decret.pdf" \
  -F "process_immediately=true"

# Retraiter tous les documents (par exemple, après mise à jour du modèle d'embedding)
curl -X POST "http://localhost:8000/api/documents/process" \
  -F "force_reprocess=true"
```

## 🛠️ Configuration avancée

Vous pouvez personnaliser le comportement du système en modifiant le fichier `app/core/config.py`. Les principaux paramètres sont :

- `DATA_PATH` et `METADATA_PATH` - Chemins des données
- `EMBEDDING_DIM` - Dimension des vecteurs d'embedding
- `LLM_MODEL` - Modèle DeepSeek à utiliser
- `MAX_TOKENS` et `OVERLAP_TOKENS` - Configuration du chunking

## 🔍 Débogage et monitoring

Le système inclut plusieurs endpoints utiles pour le débogage :

- `/api/search/expand?query=...` - Visualise l'expansion de requête
- `/api/rerank/legal-score?query=...&text=...` - Calcule le score de pertinence juridique
- `/api/rag/history` - Affiche l'historique des questions-réponses

## 🔒 Sécurité et limitations

- L'API ne gère pas actuellement l'authentification des utilisateurs
- Les résultats fournis sont à titre informatif et ne constituent pas un avis juridique
- Le système a été optimisé pour le corpus juridique camerounais et pourrait nécessiter des adaptations pour d'autres juridictions

## 📚 Ressources additionnelles

- [Documentation DeepSeek](https://deepseek.ai/docs/)
- [Guide Milvus](https://milvus.io/docs)
- [Spécifications BGE Embeddings](https://huggingface.co/BAAI/bge-m3)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

## 📄 Licence

Ce projet est sous licence propriétaire. Tous droits réservés.

---

Pour toute question ou suggestion, contactez [armelle.mfegue@example.com]