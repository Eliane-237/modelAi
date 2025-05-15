// Configuration de l'API
const API_BASE_URL = 'http://localhost:8000/api';
const API_ENDPOINTS = {
    rag: '/rag/question',
    search: '/search',
    documents: '/documents/list',
    upload: '/documents/upload',
    processDoc: '/documents/process',
    history: '/rag/history',
    // Nouveaux endpoints pour l'Agent RAG
    agent: '/agent/chat',
    agentStream: '/agent/chat/stream',
    agentSessions: '/agent/sessions',
    agentReset: '/agent/reset'
};

// Sélecteurs DOM
const DOM = {
    // Navigation
    navItems: document.querySelectorAll('.nav-menu li'),
    sections: document.querySelectorAll('.content-section'),
    rightPanel: document.getElementById('rightPanel'),
    closePanel: document.getElementById('closePanel'),
    
    // Chat section
    chatMessages: document.getElementById('chatMessages'),
    questionInput: document.getElementById('questionInput'),
    sendQuestion: document.getElementById('sendQuestion'),
    resetChatBtn: document.getElementById('resetChat'), // Bouton pour réinitialiser le chat
    
    // Documents section
    uploadBtn: document.getElementById('uploadBtn'),
    fileUpload: document.getElementById('fileUpload'),
    documentsList: document.getElementById('documentsList'),
    docSearch: document.getElementById('docSearch'),
    docFilter: document.getElementById('docFilter'),
    
    // Search section
    searchInput: document.getElementById('searchInput'),
    searchButton: document.getElementById('searchButton'),
    searchResults: document.getElementById('searchResults'),
    useRerank: document.getElementById('useRerank'),
    topK: document.getElementById('topK'),
    
    // History section
    historyList: document.getElementById('historyList'),
    
    // Modals
    loadingModal: document.getElementById('loadingModal'),
    loadingMessage: document.getElementById('loadingMessage'),
    
    // Panel
    panelContent: document.getElementById('panelContent'),
    previewTitle: document.getElementById('previewTitle')
};

// État de l'application
const appState = {
    activeSection: 'chat',
    documents: [],
    searchResults: [],
    history: [],
    sessions: [],
    currentSessionId: null,
    rightPanelOpen: false,
    isLoading: false,
    useAgent: true // Utiliser l'agent au lieu du RAG traditionnel
};

// Utilitaires génériques
const utils = {
    // Formater une date
    formatDate: (timestamp) => {
        const date = new Date(timestamp * 1000);
        return date.toLocaleString();
    },
    
    // Tronquer un texte
    truncateText: (text, maxLength = 150) => {
        if (text.length <= maxLength) return text;
        return text.substr(0, maxLength) + '...';
    },
    
    // Afficher un message de chargement
    showLoading: (message = 'Traitement en cours...') => {
        DOM.loadingMessage.textContent = message;
        DOM.loadingModal.classList.add('active');
        appState.isLoading = true;
    },
    
    // Masquer le message de chargement
    hideLoading: () => {
        DOM.loadingModal.classList.remove('active');
        appState.isLoading = false;
    },
    
    // Requête API
    fetchAPI: async (endpoint, options = {}) => {
        try {
            const url = API_BASE_URL + endpoint;
            const response = await fetch(url, options);
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `Erreur ${response.status}: ${response.statusText}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('Erreur API:', error);
            throw error;
        }
    }
};

// Gestionnaires pour chaque section
const handlers = {
    // Section Chat - Assistant juridique
    chat: {
        init: () => {
            // Initial message already in HTML
            
            // Event listener for send button
            DOM.sendQuestion.addEventListener('click', handlers.chat.sendQuestion);
            
            // Event listener for Enter key in textarea
            DOM.questionInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handlers.chat.sendQuestion();
                }
            });
            
            // Ajouter l'event listener pour le bouton de réinitialisation
            if (DOM.resetChatBtn) {
                DOM.resetChatBtn.addEventListener('click', handlers.chat.resetChat);
            }
        },
        
        sendQuestion: async () => {
            const question = DOM.questionInput.value.trim();
            if (!question) return;
            
            // Add user message to chat
            handlers.chat.addMessage(question, 'user');
            
            // Clear input
            DOM.questionInput.value = '';
            
            try {
                utils.showLoading('Recherche de la réponse...');
                
                // Utiliser soit l'agent RAG soit le RAG standard selon la configuration
                if (appState.useAgent) {
                    // Send question to Agent API
                    const response = await utils.fetchAPI(API_ENDPOINTS.agent, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            query: question,
                            session_id: appState.currentSessionId,
                            streaming: false
                        })
                    });
                    
                    utils.hideLoading();
                    
                    // Stocker l'ID de session
                    appState.currentSessionId = response.session_id;
                    
                    // Add AI response to chat with additional metadata
                    handlers.chat.addMessage(response.response, 'ai', response.source_documents, {
                        domains: response.domains,
                        intent: response.intent,
                        responseTime: response.response_time
                    });
                } else {
                    // Send question to traditional RAG API
                    const response = await utils.fetchAPI(API_ENDPOINTS.rag, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            query: question,
                            use_expansion: true,
                            use_reranking: true
                        })
                    });
                    
                    utils.hideLoading();
                    
                    // Add AI response to chat
                    handlers.chat.addMessage(response.answer, 'ai', response.source_documents);
                }
                
                // Scroll to bottom
                DOM.chatMessages.scrollTop = DOM.chatMessages.scrollHeight;
                
            } catch (error) {
                utils.hideLoading();
                handlers.chat.addMessage(
                    `Désolé, une erreur est survenue lors de la recherche de la réponse: ${error.message}`,
                    'system'
                );
            }
        },
        
        addMessage: (text, type, sources = [], metadata = {}) => {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${type}`;
            
            // Message content
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            
            // Format text with line breaks
            text.split('\n').forEach(line => {
                if (line.trim()) {
                    const p = document.createElement('p');
                    p.textContent = line;
                    contentDiv.appendChild(p);
                } else {
                    contentDiv.appendChild(document.createElement('br'));
                }
            });
            
            messageDiv.appendChild(contentDiv);
            
            // Add sources if available
            if (sources && sources.length > 0 && type === 'ai') {
                const sourceListDiv = document.createElement('div');
                sourceListDiv.className = 'source-list';
                
                const sourceHeader = document.createElement('p');
                sourceHeader.innerHTML = '<strong>Sources:</strong>';
                sourceListDiv.appendChild(sourceHeader);
                
                sources.slice(0, 3).forEach(source => {
                    const sourceItem = document.createElement('div');
                    sourceItem.className = 'source-item';
                    
                    const metadata = source.metadata || {};
                    let citation = source.citation;
                    
                    if (!citation) {
                        const filename = metadata.filename || 'Document inconnu';
                        const page = metadata.page_number || '?';
                        citation = `${filename} (page ${page})`;
                    }
                    
                    sourceItem.innerHTML = `<i class="fas fa-file-alt"></i> ${citation}`;
                    sourceListDiv.appendChild(sourceItem);
                });
                
                contentDiv.appendChild(sourceListDiv);
                
                // Add domains if available
                if (metadata.domains && metadata.domains.length) {
                    const domainsDiv = document.createElement('div');
                    domainsDiv.className = 'domains-list';
                    domainsDiv.style.marginTop = "8px";
                    domainsDiv.style.fontSize = "0.85rem";
                    domainsDiv.style.color = "#64748b";
                    domainsDiv.innerHTML = `<strong>Domaines juridiques:</strong> ${metadata.domains.join(', ')}`;
                    contentDiv.appendChild(domainsDiv);
                }
                
                // Add response time if available
                if (metadata.responseTime) {
                    const timeDiv = document.createElement('div');
                    timeDiv.className = 'response-time';
                    timeDiv.style.marginTop = "4px";
                    timeDiv.style.fontSize = "0.75rem";
                    timeDiv.style.color = "#94a3b8";
                    timeDiv.textContent = `Temps de réponse: ${metadata.responseTime.toFixed(2)}s`;
                    contentDiv.appendChild(timeDiv);
                }
            }
            
            // Add timestamp
            const metaDiv = document.createElement('div');
            metaDiv.className = 'message-meta';
            metaDiv.textContent = new Date().toLocaleTimeString();
            messageDiv.appendChild(metaDiv);
            
            // Add to chat
            DOM.chatMessages.appendChild(messageDiv);
            
            // Scroll to bottom
            DOM.chatMessages.scrollTop = DOM.chatMessages.scrollHeight;
        },
        
        resetChat: async () => {
            try {
                utils.showLoading('Réinitialisation de la conversation...');
                
                // Si nous avons un ID de session, envoyer une requête pour réinitialiser
                if (appState.currentSessionId) {
                    await utils.fetchAPI(API_ENDPOINTS.agentReset, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            session_id: appState.currentSessionId
                        })
                    });
                    
                    // Effacer l'ID de session actuel
                    appState.currentSessionId = null;
                }
                
                utils.hideLoading();
                
                // Effacer tous les messages du chat
                DOM.chatMessages.innerHTML = '';
                
                // Ajouter un message système pour indiquer que la conversation a été réinitialisée
                handlers.chat.addMessage(
                    "Conversation réinitialisée. Vous pouvez commencer une nouvelle discussion.",
                    'system'
                );
                
            } catch (error) {
                utils.hideLoading();
                handlers.chat.addMessage(
                    `Erreur lors de la réinitialisation de la conversation: ${error.message}`,
                    'system'
                );
            }
        },
        
        loadSession: async (sessionId) => {
            try {
                utils.showLoading('Chargement de la session...');
                
                // Obtenir les informations de la session
                const response = await utils.fetchAPI(`/agent/sessions/${sessionId}`);
                
                utils.hideLoading();
                
                // Stocker l'ID de session
                appState.currentSessionId = sessionId;
                
                // Effacer tous les messages du chat
                DOM.chatMessages.innerHTML = '';
                
                // Ajouter un message système pour indiquer que la session a été chargée
                handlers.chat.addMessage(
                    `Session #${sessionId} chargée avec succès. Cette conversation contient ${response.message_count} messages.`,
                    'system'
                );
                
                // Ajouter les messages de l'historique
                response.messages.forEach(msg => {
                    handlers.chat.addMessage(
                        msg.content,
                        msg.role === "user" ? "user" : "ai"
                    );
                });
                
            } catch (error) {
                utils.hideLoading();
                handlers.chat.addMessage(
                    `Erreur lors du chargement de la session: ${error.message}`,
                    'system'
                );
            }
        }
    },
    
    // Section Documents
    documents: {
        init: async () => {
            // Event listeners
            DOM.uploadBtn.addEventListener('click', () => DOM.fileUpload.click());
            DOM.fileUpload.addEventListener('change', handlers.documents.uploadDocument);
            DOM.fileUpload.addEventListener('change', handlers.documents.handleFileSelection);
            DOM.docSearch.addEventListener('input', handlers.documents.filterDocuments);
            DOM.docFilter.addEventListener('change', handlers.documents.filterDocuments);

            // Ajouter un écouteur pour le glisser-déposer
            const uploadZone = document.querySelector('.upload-zone');
            uploadZone.addEventListener('dragover', e => {
                e.preventDefault();
                uploadZone.classList.add('drag-over');
            });
            uploadZone.addEventListener('dragleave', () => {
                uploadZone.classList.remove('drag-over');
            });
            uploadZone.addEventListener('drop', e => {
                e.preventDefault();
                uploadZone.classList.remove('drag-over');
                
                if (e.dataTransfer.files.length > 0) {
                    handlers.documents.handleFileSelection({ target: { files: e.dataTransfer.files } });
                }
            });
            
            // Load documents
            await handlers.documents.loadDocuments();

        },

        handleFileSelection: (e) => {
            const files = Array.from(e.target.files).filter(file => file.type === 'application/pdf');
            
            if (files.length === 0) {
                alert('Veuillez sélectionner des fichiers PDF.');
                return;
            }
            
            // Afficher les fichiers sélectionnés
            const selectedFilesContainer = document.getElementById('selectedFiles');
            selectedFilesContainer.innerHTML = '';
            
            files.forEach(file => {
                const fileItem = document.createElement('div');
                fileItem.className = 'file-item';
                fileItem.innerHTML = `
                    <div class="file-name">${file.name}</div>
                    <i class="fas fa-times remove-file" data-name="${file.name}"></i>
                `;
                selectedFilesContainer.appendChild(fileItem);
            });
            
            // Ajouter un bouton pour télécharger tous les fichiers
            if (files.length > 0) {
                const uploadActions = document.createElement('div');
                uploadActions.className = 'upload-actions';
                uploadActions.innerHTML = `
                    <button id="uploadSelectedBtn" class="btn primary">
                        <i class="fas fa-upload"></i> Télécharger ${files.length} fichier(s)
                    </button>
                    <button id="clearSelectedBtn" class="btn secondary">
                        <i class="fas fa-times"></i> Annuler
                    </button>
                    <div class="upload-progress">
                        <div class="progress-bar" id="uploadProgressBar"></div>
                    </div>
                `;
                selectedFilesContainer.appendChild(uploadActions);
                
                // Ajouter les écouteurs d'événements
                document.getElementById('uploadSelectedBtn').addEventListener('click', () => {
                    handlers.documents.uploadSelectedFiles(files);
                });
                
                document.getElementById('clearSelectedBtn').addEventListener('click', () => {
                    selectedFilesContainer.innerHTML = '';
                    DOM.fileUpload.value = ''; // Réinitialiser l'input file
                });
                
                // Ajouter les gestionnaires pour supprimer des fichiers individuels
                document.querySelectorAll('.remove-file').forEach(btn => {
                    btn.addEventListener('click', (e) => {
                        const fileName = e.target.getAttribute('data-name');
                        // Filtrer le fichier du tableau
                        const newFileList = Array.from(files).filter(f => f.name !== fileName);
                        // Mettre à jour l'affichage
                        handlers.documents.handleFileSelection({ target: { files: newFileList } });
                    });
                });
            }
        },

        uploadSelectedFiles: async (files) => {
            const totalFiles = files.length;
            let successCount = 0;
            let failCount = 0;
            
            // Initialiser la barre de progression
            const progressBar = document.getElementById('uploadProgressBar');
            progressBar.style.width = '0%';
            
            // Désactiver le bouton d'upload pendant le processus
            const uploadBtn = document.getElementById('uploadSelectedBtn');
            const clearBtn = document.getElementById('clearSelectedBtn');
            uploadBtn.disabled = true;
            clearBtn.disabled = true;
            uploadBtn.innerHTML = `<i class="fas fa-spinner fa-spin"></i> Téléchargement...`;
            
            utils.showLoading(`Téléchargement de ${totalFiles} fichier(s)...`);
            
            for (let i = 0; i < files.length; i++) {
                const file = files[i];
                try {
                    // Mettre à jour la barre de progression
                    const progress = Math.round(((i) / totalFiles) * 100);
                    progressBar.style.width = `${progress}%`;
                    
                    // Créer un FormData pour chaque fichier
                    const formData = new FormData();
                    formData.append('file', file);
                    formData.append('process_immediately', 'true');
                    
                    // Télécharger le fichier
                    const response = await fetch(API_BASE_URL + API_ENDPOINTS.upload, {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (!response.ok) {
                        const errorData = await response.json().catch(() => ({}));
                        throw new Error(errorData.detail || `Erreur ${response.status}: ${response.statusText}`);
                    }
                    
                    const data = await response.json();
                    successCount++;
                    
                } catch (error) {
                    console.error(`Erreur lors du téléchargement de ${file.name}:`, error);
                    failCount++;
                }
                
                // Mettre à jour la progression
                utils.loadingMessage.textContent = `Téléchargement ${i + 1}/${totalFiles}...`;
            }
            
            // Mettre à jour la progression finale
            progressBar.style.width = '100%';
            
            // Réactiver les boutons
            uploadBtn.disabled = false;
            clearBtn.disabled = false;
            uploadBtn.innerHTML = `<i class="fas fa-upload"></i> Télécharger`;
            
            utils.hideLoading();
            
            // Afficher le résultat
            alert(`Téléchargement terminé. ${successCount} fichier(s) traité(s) avec succès, ${failCount} échec(s).`);
            
            // Réinitialiser l'affichage
            document.getElementById('selectedFiles').innerHTML = '';
            DOM.fileUpload.value = '';
            
            // Recharger la liste des documents
            await handlers.documents.loadDocuments();
        },
        
        loadDocuments: async () => {
            try {
                DOM.documentsList.innerHTML = '<div class="loading-indicator">Chargement des documents...</div>';
                
                const data = await utils.fetchAPI(API_ENDPOINTS.documents);
                appState.documents = data.documents_list || [];
                
                handlers.documents.renderDocuments();
                
            } catch (error) {
                DOM.documentsList.innerHTML = `
                    <div class="error-message">
                        <i class="fas fa-exclamation-circle"></i> 
                        Erreur lors du chargement des documents: ${error.message}
                    </div>
                `;
            }
        },
        
        renderDocuments: () => {
            if (!appState.documents.length) {
                DOM.documentsList.innerHTML = `
                    <div class="loading-indicator">
                        <i class="fas fa-file-alt"></i> 
                        Aucun document disponible. Téléchargez votre premier document!
                    </div>
                `;
                return;
            }
            
            // Filter documents based on search and filter
            const searchText = DOM.docSearch.value.toLowerCase();
            const filterValue = DOM.docFilter.value;
            
            const filteredDocs = appState.documents.filter(doc => {
                const filename = doc.filename?.toLowerCase() || '';
                const isScanned = doc.is_scanned || false;
                
                // Apply text search
                const textMatch = !searchText || filename.includes(searchText);
                
                // Apply type filter
                let typeMatch = true;
                if (filterValue === 'scanned') typeMatch = isScanned;
                if (filterValue === 'digital') typeMatch = !isScanned;
                
                return textMatch && typeMatch;
            });
            
            // Render filtered documents
            DOM.documentsList.innerHTML = '';
            
            filteredDocs.forEach(doc => {
                const card = document.createElement('div');
                card.className = 'document-card';
                card.dataset.id = doc.document_id;
                
                const filename = doc.filename || 'Document sans nom';
                const chunkCount = doc.chunk_count || 0;
                const isScanned = doc.is_scanned ? 'Document scanné' : 'Document numérique';
                const dateAdded = doc.processed_date ? utils.formatDate(new Date(doc.processed_date).getTime() / 1000) : 'Date inconnue';
                
                card.innerHTML = `
                    <div class="document-header">
                        <span>${utils.truncateText(filename, 20)}</span>
                        <i class="fas ${doc.is_scanned ? 'fa-file-image' : 'fa-file-alt'}"></i>
                    </div>
                    <div class="document-body">
                        <div>${utils.truncateText(filename, 30)}</div>
                        <div class="document-meta">
                            <div><i class="fas fa-puzzle-piece"></i> ${chunkCount} segments</div>
                            <div><i class="fas ${doc.is_scanned ? 'fa-file-image' : 'fa-file-alt'}"></i> ${isScanned}</div>
                            <div><i class="fas fa-calendar"></i> ${dateAdded}</div>
                        </div>
                    </div>
                    <div class="document-actions">
                        <button class="btn secondary view-doc"><i class="fas fa-eye"></i> Aperçu</button>
                        <button class="btn secondary process-doc"><i class="fas fa-sync"></i> Retraiter</button>
                    </div>
                `;
                
                // Add event listeners
                card.querySelector('.view-doc').addEventListener('click', (e) => {
                    e.stopPropagation();
                    handlers.documents.viewDocument(doc);
                });
                
                card.querySelector('.process-doc').addEventListener('click', (e) => {
                    e.stopPropagation();
                    handlers.documents.processDocument(doc);
                });
                
                DOM.documentsList.appendChild(card);
            });
        },
        
        uploadSelectedFiles: async (files) => {
            const totalFiles = files.length;
            let successCount = 0;
            let failCount = 0;
            
            // Créer un conteneur pour le suivi visuel
            const selectedFilesContainer = document.getElementById('selectedFiles');
            selectedFilesContainer.innerHTML = '';
            
            // Créer un panneau de suivi des téléchargements
            const uploadPanel = document.createElement('div');
            uploadPanel.className = 'upload-panel';
            uploadPanel.innerHTML = `
                <div class="upload-panel-header">
                    <h3>Traitement des documents (${totalFiles} fichiers)</h3>
                    <div class="global-progress">
                        <div class="progress-track">
                            <div class="progress-bar" id="globalProgressBar"></div>
                        </div>
                        <div class="progress-status">0/${totalFiles}</div>
                    </div>
                </div>
                <div class="upload-files-list" id="uploadFilesList"></div>
                <div class="upload-panel-footer">
                    <button id="cancelUploadBtn" class="btn secondary">Annuler</button>
                </div>
            `;
            selectedFilesContainer.appendChild(uploadPanel);
            
            // Référence à la barre de progression globale
            const globalProgressBar = document.getElementById('globalProgressBar');
            const globalProgressStatus = uploadPanel.querySelector('.progress-status');
            
            // Créer une entrée pour chaque fichier
            const uploadFilesList = document.getElementById('uploadFilesList');
            const fileEntries = {};
            
            files.forEach(file => {
                const fileEntry = document.createElement('div');
                fileEntry.className = 'file-entry';
                fileEntry.innerHTML = `
                    <div class="file-entry-header">
                        <div class="file-icon"><i class="fas fa-file-pdf"></i></div>
                        <div class="file-info">
                            <div class="file-name">${file.name}</div>
                            <div class="file-size">${(file.size / 1024).toFixed(1)} KB</div>
                        </div>
                        <div class="file-status pending">En attente</div>
                    </div>
                    <div class="file-progress">
                        <div class="progress-track">
                            <div class="progress-bar" id="progress-${file.name.replace(/\s+/g, '_')}"></div>
                        </div>
                        <div class="progress-percentage">0%</div>
                    </div>
                    <div class="file-message">En attente de traitement...</div>
                `;
                uploadFilesList.appendChild(fileEntry);
                
                fileEntries[file.name] = {
                    element: fileEntry,
                    progressBar: fileEntry.querySelector('.progress-bar'),
                    percentage: fileEntry.querySelector('.progress-percentage'),
                    status: fileEntry.querySelector('.file-status'),
                    message: fileEntry.querySelector('.file-message')
                };
            });
            
            // Fonction pour mettre à jour l'état d'un fichier
            const updateFileStatus = (fileName, progress, status, message) => {
                const entry = fileEntries[fileName];
                if (!entry) return;
                
                // Mettre à jour la barre de progression
                entry.progressBar.style.width = `${progress}%`;
                entry.percentage.textContent = `${progress}%`;
                
                // Mettre à jour le statut
                entry.status.className = `file-status ${status}`;
                entry.status.textContent = {
                    'pending': 'En attente',
                    'uploading': 'Téléchargement',
                    'processing': 'Traitement',
                    'indexing': 'Indexation',
                    'complete': 'Terminé',
                    'error': 'Erreur'
                }[status] || status;
                
                // Mettre à jour le message
                if (message) {
                    entry.message.textContent = message;
                }
                
                // Mettre à jour la progression globale
                const completedCount = successCount + failCount;
                globalProgressBar.style.width = `${(completedCount / totalFiles) * 100}%`;
                globalProgressStatus.textContent = `${completedCount}/${totalFiles}`;
            };
            
            // Gérer l'annulation
            document.getElementById('cancelUploadBtn').addEventListener('click', () => {
                if (confirm('Êtes-vous sûr de vouloir annuler le traitement ?')) {
                    // Réinitialiser l'affichage
                    selectedFilesContainer.innerHTML = '';
                    DOM.fileUpload.value = '';
                }
            });
            
            // Traiter les fichiers par lots de 3 simultanément
            const batchSize = 3;
            const processFile = async (file) => {
                try {
                    // Mise à jour du statut initial
                    updateFileStatus(file.name, 5, 'uploading', 'Préparation du téléchargement...');
                    
                    // Créer un FormData pour le fichier
                    const formData = new FormData();
                    formData.append('file', file);
                    formData.append('process_immediately', 'true');
                    
                    // Créer une requête avec suivi de progression
                    const xhr = new XMLHttpRequest();
                    xhr.open('POST', API_BASE_URL + API_ENDPOINTS.upload);
                    
                    // Suivre la progression du téléchargement
                    xhr.upload.onprogress = (event) => {
                        if (event.lengthComputable) {
                            const uploadProgress = Math.round((event.loaded / event.total) * 40); // 40% pour l'upload
                            updateFileStatus(file.name, uploadProgress, 'uploading', 'Téléchargement en cours...');
                        }
                    };
                    
                    // Gérer la réponse
                    const response = await new Promise((resolve, reject) => {
                        xhr.onload = () => {
                            if (xhr.status >= 200 && xhr.status < 300) {
                                updateFileStatus(file.name, 50, 'processing', 'Traitement du document...');
                                try {
                                    resolve(JSON.parse(xhr.responseText));
                                } catch (e) {
                                    reject(new Error('Réponse invalide du serveur'));
                                }
                            } else {
                                updateFileStatus(file.name, 100, 'error', `Erreur ${xhr.status}: ${xhr.statusText}`);
                                reject(new Error(`Erreur ${xhr.status}: ${xhr.statusText}`));
                            }
                        };
                        xhr.onerror = () => {
                            updateFileStatus(file.name, 100, 'error', 'Erreur réseau lors du téléchargement');
                            reject(new Error('Erreur réseau lors du téléchargement'));
                        };
                        xhr.send(formData);
                    });
                    
                    // Si nous avons un ID de document, suivre son traitement
                    if (response && response.document_id) {
                        const documentId = response.document_id;
                        
                        // Suivre le traitement avec des vérifications périodiques
                        let processingComplete = false;
                        let retryCount = 0;
                        let progress = 50;
                        
                        updateFileStatus(file.name, progress, 'processing', 'Analyse du document...');
                        
                        while (!processingComplete && retryCount < 15) {
                            await new Promise(resolve => setTimeout(resolve, 2000)); // Attendre 2 secondes
                            
                            try {
                                // Vérifier l'état du document
                                const statusResponse = await fetch(`${API_BASE_URL}/documents/info/${documentId}`);
                                
                                if (statusResponse.ok) {
                                    const docInfo = await statusResponse.json();
                                    
                                    // Vérifier si le traitement est terminé
                                    if (docInfo.processed && docInfo.chunk_count > 0) {
                                        processingComplete = true;
                                        updateFileStatus(file.name, 100, 'complete', 
                                            `Traitement terminé. ${docInfo.chunk_count} segments indexés.`);
                                        successCount++;
                                    } else {
                                        // Augmenter progressivement la barre de progression
                                        progress = Math.min(90, progress + 3);
                                        const statusMessage = retryCount < 5 ? 'Extraction du texte...' :
                                                            retryCount < 10 ? 'Génération des embeddings...' :
                                                            'Finalisation de l\'indexation...';
                                                            
                                        updateFileStatus(file.name, progress, 'indexing', statusMessage);
                                    }
                                } else {
                                    // Erreur lors de la vérification de l'état
                                    throw new Error(`Erreur lors de la vérification: ${statusResponse.status}`);
                                }
                            } catch (error) {
                                console.warn(`Erreur lors de la vérification de l'état de ${file.name}:`, error);
                            }
                            
                            retryCount++;
                        }
                        
                        // Si le traitement n'est pas terminé après le timeout, considérer comme succès partiel
                        if (!processingComplete) {
                            updateFileStatus(file.name, 95, 'complete', 
                                'Document traité (le statut final sera visible après actualisation)');
                            successCount++;
                        }
                    } else {
                        // Pas d'ID de document dans la réponse
                        updateFileStatus(file.name, 100, 'error', 'Erreur: Réponse incomplète du serveur');
                        failCount++;
                    }
                    
                } catch (error) {
                    console.error(`Erreur lors du traitement de ${file.name}:`, error);
                    updateFileStatus(file.name, 100, 'error', `Erreur: ${error.message}`);
                    failCount++;
                }
            };
            
            // Traiter les fichiers par lots
            const fileBatches = [];
            for (let i = 0; i < files.length; i += batchSize) {
                fileBatches.push(Array.from(files).slice(i, i + batchSize));
            }
            
            // Traiter les lots séquentiellement, mais les fichiers de chaque lot en parallèle
            for (const batch of fileBatches) {
                await Promise.all(batch.map(file => processFile(file)));
            }
            
            // Ajouter un bouton pour finaliser et recharger la liste des documents
            const completeButton = document.createElement('button');
            completeButton.className = 'btn primary';
            completeButton.style.marginTop = '15px';
            completeButton.innerHTML = '<i class="fas fa-check-circle"></i> Traitement terminé - Voir tous les documents';
            completeButton.addEventListener('click', async () => {
                // Réinitialiser l'affichage
                selectedFilesContainer.innerHTML = '';
                DOM.fileUpload.value = '';
                
                // Recharger la liste des documents
                await handlers.documents.loadDocuments();
            });
            
            // Ajouter un résumé du traitement
            const summaryDiv = document.createElement('div');
            summaryDiv.className = successCount === totalFiles ? 'success-message' : 'warning-message';
            summaryDiv.innerHTML = `<i class="fas fa-info-circle"></i> Traitement terminé: ${successCount} fichier(s) traité(s) avec succès, ${failCount} échec(s).`;
            
            // Ajouter le résumé et le bouton
            uploadPanel.querySelector('.upload-panel-footer').appendChild(summaryDiv);
            uploadPanel.querySelector('.upload-panel-footer').appendChild(completeButton);
            
            // Mettre à jour l'état de l'application en arrière-plan
            try {
                const data = await utils.fetchAPI(API_ENDPOINTS.documents);
                appState.documents = data.documents_list || [];
            } catch (e) {
                console.error("Erreur lors du rechargement des données:", e);
            }
        },
        
        filterDocuments: () => {
            handlers.documents.renderDocuments();
        },
        
        viewDocument: (doc) => {
            DOM.previewTitle.textContent = doc.filename || 'Détails du document';
            
            let content = `
                <h3>${doc.filename || 'Document sans nom'}</h3>
                <div class="document-meta" style="margin: 20px 0;">
                    <p><strong>ID:</strong> ${doc.document_id}</p>
                    <p><strong>Type:</strong> ${doc.is_scanned ? 'Document scanné' : 'Document numérique'}</p>
                    <p><strong>Segments:</strong> ${doc.chunk_count || 0}</p>
                    <p><strong>Date d'ajout:</strong> ${doc.processed_date ? utils.formatDate(new Date(doc.processed_date).getTime() / 1000) : 'Date inconnue'}</p>
                </div>
            `;
            
            if (doc.page_count) {
                content += `<p><strong>Nombre de pages:</strong> ${doc.page_count}</p>`;
            }
            
            // Add sample extracts if available
            if (doc.samples && doc.samples.length) {
                content += `
                    <div style="margin-top: 20px;">
                        <h4>Extraits du document:</h4>
                        <div class="document-samples">
                `;
                
                doc.samples.forEach((sample, index) => {
                    content += `
                        <div class="sample-item" style="margin-bottom: 15px; padding: 10px; background-color: #f8fafc; border-radius: 8px;">
                            <p><strong>Extrait ${index + 1}:</strong></p>
                            <p>${sample}</p>
                        </div>
                    `;
                });
                
                content += `</div></div>`;
            }
            
            // Add actions
            content += `
                <div style="margin-top: 30px; display: flex; gap: 10px;">
                    <button id="searchDocBtn" class="btn primary">
                        <i class="fas fa-search"></i> Rechercher dans ce document
                    </button>
                    <button id="deleteDocBtn" class="btn secondary">
                        <i class="fas fa-trash"></i> Supprimer
                    </button>
                </div>
            `;
            
            DOM.panelContent.innerHTML = content;
            
            // Add event listeners
            document.getElementById('searchDocBtn').addEventListener('click', () => {
                // Switch to search view
                navigation.activateSection('search');
                // Pre-populate search box
                const searchTerm = `filename:${doc.filename}`;
                DOM.searchInput.value = searchTerm;
                // Close panel
                handlers.ui.toggleRightPanel(false);
            });
            
            document.getElementById('deleteDocBtn').addEventListener('click', () => {
                if (confirm(`Êtes-vous sûr de vouloir supprimer ce document: ${doc.filename}?`)) {
                    alert('Cette fonctionnalité n\'est pas encore implémentée.');
                    // TODO: Implement document deletion
                }
            });
            
            // Open right panel
            handlers.ui.toggleRightPanel(true);
        },
        
        processDocument: async (doc) => {
            if (!confirm(`Êtes-vous sûr de vouloir retraiter ce document: ${doc.filename}?`)) {
                return;
            }
            
            try {
                utils.showLoading('Retraitement du document...');
                
                const formData = new FormData();
                formData.append('force_reprocess', 'true');
                
                const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.processDoc}/${doc.filename}`, {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    const errorData = await response.json().catch(() => ({}));
                    throw new Error(errorData.detail || `Erreur ${response.status}: ${response.statusText}`);
                }
                
                const data = await response.json();
                
                utils.hideLoading();
                
                alert(`Document "${doc.filename}" retraité avec succès.`);
                
                // Reload documents
                await handlers.documents.loadDocuments();
                
            } catch (error) {
                utils.hideLoading();
                alert(`Erreur lors du retraitement: ${error.message}`);
            }
        }
    },
    
    // Section Search
    search: {
        init: () => {
            // Event listeners
            DOM.searchButton.addEventListener('click', handlers.search.performSearch);
            DOM.searchInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    handlers.search.performSearch();
                }
            });
        },
        
        performSearch: async () => {
            const query = DOM.searchInput.value.trim();
            if (!query) return;
            
            try {
                utils.showLoading('Recherche en cours...');
                
                const useRerank = DOM.useRerank.checked;
                const topK = parseInt(DOM.topK.value);
                
                // Send search request
                const response = await utils.fetchAPI(API_ENDPOINTS.search, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        query,
                        top_k: topK,
                        use_rerank: useRerank,
                        use_llm_rerank: false
                    })
                });
                
                utils.hideLoading();
                
                // Store results
                appState.searchResults = response.results || [];
                
                // Render results
                handlers.search.renderResults(query, response);
                
            } catch (error) {
                utils.hideLoading();
                DOM.searchResults.innerHTML = `
                    <div class="error-message">
                        <i class="fas fa-exclamation-circle"></i> 
                        Erreur lors de la recherche: ${error.message}
                    </div>
                `;
            }
        },
        
        renderResults: (query, response) => {
            const results = response.results || [];
            const totalResults = response.total_results || 0;
            const searchTime = response.search_time || 0;
            
            if (!results.length) {
                DOM.searchResults.innerHTML = `
                    <div class="loading-indicator">
                        <i class="fas fa-search"></i> 
                        Aucun résultat trouvé pour "${query}".
                    </div>
                `;
                return;
            }
            
            // Results header
            let header = `
                <div style="margin-bottom: 20px;">
                    <h3>${totalResults} résultat(s) pour "${query}"</h3>
                    <p style="color: var(--secondary-color);">
                        Recherche effectuée en ${searchTime.toFixed(2)} secondes
                    </p>
                </div>
            `;
            
            // Build results
            let resultsHTML = '';
            
            results.forEach((result, index) => {
                const metadata = result.metadata || {};
                const score = result.score * 100; // Convert to percentage
                
                resultsHTML += `
                    <div class="result-card">
                        <div class="result-header">
                            <div class="result-source">
                                <i class="fas fa-file-alt"></i>
                                ${metadata.filename || 'Document inconnu'} (p.${metadata.page_number || '?'})
                            </div>
                            <div class="result-score">${score.toFixed(1)}%</div>
                        </div>
                        <div class="result-content">
                            <p>${result.text}</p>
                            <div class="result-meta">
                                <span><i class="fas fa-file-alt"></i> ${metadata.document_id ? metadata.document_id.substring(0, 8) + '...' : 'ID inconnu'}</span>
                                <span><i class="fas fa-puzzle-piece"></i> ${metadata.chunk_id ? metadata.chunk_id.substring(0, 8) + '...' : 'Chunk inconnu'}</span>
                                ${metadata.section_type ? `<span><i class="fas fa-bookmark"></i> ${metadata.section_type} ${metadata.section_number || ''}</span>` : ''}
                            </div>
                        </div>
                    </div>
                `;
            });
            
            DOM.searchResults.innerHTML = header + resultsHTML;
        }
    },
    
    // Section History
    history: {
        init: async () => {
            // Si nous utilisons l'agent, charger les sessions au lieu de l'historique standard
            if (appState.useAgent) {
                await handlers.history.loadSessions();
            } else {
                await handlers.history.loadHistory();
            }
        },
        
        loadHistory: async () => {
            try {
                DOM.historyList.innerHTML = '<div class="loading-indicator">Chargement de l\'historique...</div>';
                
                const data = await utils.fetchAPI(API_ENDPOINTS.history);
                appState.history = data || [];
                
                handlers.history.renderHistory();
                
            } catch (error) {
                DOM.historyList.innerHTML = `
                    <div class="error-message">
                        <i class="fas fa-exclamation-circle"></i> 
                        Erreur lors du chargement de l'historique: ${error.message}
                    </div>
                `;
            }
        },
        
        renderHistory: () => {
            if (!appState.history.length) {
                DOM.historyList.innerHTML = `
                    <div class="loading-indicator">
                        <i class="fas fa-history"></i> 
                        Aucun historique disponible. Posez une question pour commencer!
                    </div>
                `;
                return;
            }
            
            // Render history items
            DOM.historyList.innerHTML = '';
            
            appState.history.forEach(item => {
                const card = document.createElement('div');
                card.className = 'history-item';
                
                const time = item.timestamp ? utils.formatDate(item.timestamp) : 'Date inconnue';
                
                card.innerHTML = `
                    <div class="history-header">
                        <div>Question</div>
                        <div class="history-time">${time}</div>
                    </div>
                    <div class="history-query">${item.query}</div>
                    <div class="history-answer">${item.answer}</div>
                `;
                
                // Add click event to replay question
                card.addEventListener('click', () => {
                    // Switch to chat
                    navigation.activateSection('chat');
                    
                    // Set question text
                    DOM.questionInput.value = item.query;
                    
                    // Focus input
                    DOM.questionInput.focus();
                });
                
                DOM.historyList.appendChild(card);
            });
        },
        
        loadSessions: async () => {
            try {
                DOM.historyList.innerHTML = '<div class="loading-indicator">Chargement des sessions...</div>';
                
                const data = await utils.fetchAPI(API_ENDPOINTS.agentSessions);
                appState.sessions = data.sessions || [];
                
                handlers.history.renderSessions();
                
            } catch (error) {
                DOM.historyList.innerHTML = `
                    <div class="error-message">
                        <i class="fas fa-exclamation-circle"></i> 
                        Erreur lors du chargement des sessions: ${error.message}
                    </div>
                `;
            }
        },
        
        renderSessions: () => {
            if (!appState.sessions.length) {
                DOM.historyList.innerHTML = `
                    <div class="loading-indicator">
                        <i class="fas fa-history"></i> 
                        Aucune session disponible. Posez une question pour commencer!
                    </div>
                `;
                return;
            }
            
            // Render sessions
            DOM.historyList.innerHTML = '';
            
            // Add a header with explanation
            const header = document.createElement('div');
            header.className = 'history-header-info';
            header.innerHTML = `
                <div style="padding: 10px; margin-bottom: 15px; background-color: var(--message-system-bg); border-radius: 8px;">
                    <p><strong>Sessions de conversation</strong></p>
                    <p style="font-size: 0.9rem;">Cliquez sur une session pour continuer la conversation.</p>
                </div>
            `;
            DOM.historyList.appendChild(header);
            
            appState.sessions.forEach(session => {
                const sessionItem = document.createElement('div');
                sessionItem.className = 'history-item';
                
                // Mettre en évidence la session active
                if (appState.currentSessionId === session.session_id) {
                    sessionItem.classList.add('active');
                }
                
                const time = new Date(session.last_time * 1000).toLocaleString();
                
                sessionItem.innerHTML = `
                    <div class="history-header">
                        <div>Session #${session.session_id}</div>
                        <div class="history-time">${time}</div>
                    </div>
                    <div class="history-query">${session.first_query}</div>
                    <div class="history-interactions">${session.interactions} interactions</div>
                `;
                
                // Add click event to load this session
                sessionItem.addEventListener('click', () => {
                    appState.currentSessionId = session.session_id;
                    handlers.chat.loadSession(session.session_id);
                    navigation.activateSection('chat');
                });
                
                DOM.historyList.appendChild(sessionItem);
            });
        }
    },
    
    // UI handlers
    ui: {
        toggleRightPanel: (show) => {
            if (show) {
                DOM.rightPanel.classList.add('active');
                appState.rightPanelOpen = true;
            } else {
                DOM.rightPanel.classList.remove('active');
                appState.rightPanelOpen = false;
            }
        }
    }
};

// Navigation system
const navigation = {
    init: () => {
        // Add click event to nav items
        DOM.navItems.forEach(item => {
            item.addEventListener('click', () => {
                const section = item.dataset.section;
                navigation.activateSection(section);
            });
        });
        
        // Close panel handler
        DOM.closePanel.addEventListener('click', () => {
            handlers.ui.toggleRightPanel(false);
        });
    },
    
    activateSection: (section) => {
        // Update nav items
        DOM.navItems.forEach(item => {
            if (item.dataset.section === section) {
                item.classList.add('active');
            } else {
                item.classList.remove('active');
            }
        });
        
        // Update sections
        DOM.sections.forEach(s => {
            if (s.id === section) {
                s.classList.add('active');
            } else {
                s.classList.remove('active');
            }
        });
        
        // Update app state
        appState.activeSection = section;
    }
};

// Initialize application
(async function initApp() {
    // Initialize navigation
    navigation.init();
    
    // Initialize all section handlers
    await handlers.chat.init();
    await handlers.documents.init();
    await handlers.search.init();
    await handlers.history.init();
    
    // Ajouter un message de bienvenue adapté à l'agent
    if (appState.useAgent) {
        handlers.chat.addMessage(
            "Bienvenue sur LexCam, votre assistant juridique intelligent camerounais. Je peux vous aider à trouver des informations précises sur la législation camerounaise, répondre à vos questions juridiques et vous fournir des références aux textes de loi pertinents. Comment puis-je vous assister aujourd'hui?",
            'system'
        );
    }
})();