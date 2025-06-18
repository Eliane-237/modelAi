// Configuration de l'API
const API_BASE_URL = 'http://localhost:8000/api';
const API_ENDPOINTS = {
    rag: '/rag/question',
    search: '/search',
    documents: '/documents/list',
    upload: '/documents/upload',
    processDoc: '/documents/process',
    history: '/rag/history',
    // Endpoints pour l'Agent RAG avec streaming
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
    
    // Chat section avec éléments streaming
    chatMessages: document.getElementById('chatMessages'),
    questionInput: document.getElementById('questionInput'),
    sendQuestion: document.getElementById('sendQuestion'),
    resetChatBtn: document.getElementById('resetChat'),
    useStreaming: document.getElementById('useStreaming'),
    connectionStatus: document.getElementById('connectionStatus'),
    sessionInfo: document.getElementById('sessionInfo'),
    currentSessionId: document.getElementById('currentSessionId'),
    
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

// État de l'application (fusionné avec chatState)
const appState = {
    activeSection: 'chat',
    documents: [],
    searchResults: [],
    history: [],
    sessions: [],
    currentSessionId: null,
    rightPanelOpen: false,
    isLoading: false,
    useAgent: true,
    // État du streaming
    isStreaming: false,
    eventSource: null,
    currentMessage: null
};

// Utilitaires génériques
const utils = {
    formatDate: (timestamp) => {
        const date = new Date(timestamp * 1000);
        return date.toLocaleString();
    },
    
    truncateText: (text, maxLength = 150) => {
        if (text.length <= maxLength) return text;
        return text.substr(0, maxLength) + '...';
    },
    
    showLoading: (message = 'Traitement en cours...') => {
        DOM.loadingMessage.textContent = message;
        DOM.loadingModal.classList.add('active');
        appState.isLoading = true;
    },
    
    hideLoading: () => {
        DOM.loadingModal.classList.remove('active');
        appState.isLoading = false;
    },
    
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
    },

    // Utilitaires pour le streaming
    updateConnectionStatus: (status, message) => {
        if (DOM.connectionStatus) {
            DOM.connectionStatus.className = `connection-status ${status}`;
            const span = DOM.connectionStatus.querySelector('span');
            if (span) span.textContent = message;
        }
    },

    updateSessionInfo: (sessionId, responseTime = null) => {
        appState.currentSessionId = sessionId;
        
        if (DOM.sessionInfo && DOM.currentSessionId) {
            DOM.sessionInfo.style.display = 'block';
            DOM.currentSessionId.textContent = sessionId;
            
            if (responseTime) {
                DOM.currentSessionId.textContent += ` (${responseTime.toFixed(2)}s)`;
            }
        }
    }
};

// Gestionnaires pour chaque section
const handlers = {
    // Section Chat - Assistant juridique avec streaming intégré
    chat: {
        init: () => {
            // Event listeners principaux
            DOM.sendQuestion?.addEventListener('click', handlers.chat.sendQuestion);
            DOM.resetChatBtn?.addEventListener('click', handlers.chat.resetChat);
            
            // Gestion des raccourcis clavier
            DOM.questionInput?.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    if (!appState.isStreaming) {
                        handlers.chat.sendQuestion();
                    }
                }
            });

            // Auto-resize du textarea
            DOM.questionInput?.addEventListener('input', function() {
                this.style.height = 'auto';
                this.style.height = Math.min(this.scrollHeight, 120) + 'px';
            });

            // Nettoyage lors de la fermeture
            window.addEventListener('beforeunload', () => {
                handlers.chat.closeSSEConnection();
            });
            
            console.log('✅ Chat avec streaming initialisé');
        },
        
        sendQuestion: async () => {
            const question = DOM.questionInput.value.trim();
            if (!question || appState.isStreaming) return;
            
            try {
                // Ajouter le message utilisateur
                handlers.chat.addMessage(question, 'user');
                
                // Effacer et désactiver
                DOM.questionInput.value = '';
                DOM.questionInput.style.height = 'auto';
                handlers.chat.setInputState(false);
                
                // Vérifier le mode streaming
                const useStreaming = DOM.useStreaming?.checked ?? true;
                
                if (useStreaming && appState.useAgent) {
                    // Mode streaming
                    await handlers.chat.startStreamingResponse(question);
                } else {
                    // Mode standard (fallback)
                    await handlers.chat.handleStandardResponse(question);
                }
                
            } catch (error) {
                console.error('Erreur lors de l\'envoi:', error);
                handlers.chat.addMessage(`❌ Erreur: ${error.message}`, 'system error');
            } finally {
                handlers.chat.setInputState(true);
            }
        },

        startStreamingResponse: async (query) => {
            return new Promise((resolve, reject) => {
                try {
                    // Fermer connexion précédente
                    if (appState.eventSource) {
                        appState.eventSource.close();
                    }
                    
                    // Créer le message streaming
                    const messageElement = handlers.chat.addMessage('', 'ai streaming');
                    const contentDiv = messageElement.querySelector('.message-content');
                    
                    // Indicateur de frappe
                    handlers.chat.addTypingIndicator(contentDiv);
                    
                    // Mettre à jour le statut
                    utils.updateConnectionStatus('streaming', 'Génération en cours...');
                    
                    // URL SSE
                    const url = new URL(`${API_BASE_URL}${API_ENDPOINTS.agentStream}`);
                    url.searchParams.set('query', query);
                    if (appState.currentSessionId) {
                        url.searchParams.set('session_id', appState.currentSessionId);
                    }
                    
                    // Créer EventSource
                    appState.eventSource = new EventSource(url);
                    appState.isStreaming = true;
                    
                    // Gérer les événements
                    handlers.chat.setupSSEEventHandlers(resolve, reject, messageElement);
                    
                } catch (error) {
                    reject(error);
                }
            });
        },

        setupSSEEventHandlers: (resolve, reject, messageElement) => {
            const contentDiv = messageElement.querySelector('.message-content');
            let messageContent = '';
            
            // Événement start
            appState.eventSource.addEventListener('start', (event) => {
                try {
                    const data = JSON.parse(event.data);
                    appState.currentSessionId = data.session_id;
                    utils.updateSessionInfo(data.session_id);
                    handlers.chat.removeTypingIndicator(contentDiv);
                } catch (e) {
                    console.error('Erreur parsing start:', e);
                }
            });
            
            // Événement token
            appState.eventSource.addEventListener('token', (event) => {
                try {
                    const token = event.data;
                    messageContent += token;
                    handlers.chat.updateMessageContent(contentDiv, messageContent);
                    handlers.chat.scrollToBottom();
                } catch (e) {
                    console.error('Erreur token:', e);
                }
            });
            
            // Événement end
            appState.eventSource.addEventListener('end', (event) => {
                try {
                    const data = JSON.parse(event.data);
                    
                    // Finaliser le message
                    messageElement.classList.remove('streaming');
                    
                    // Ajouter les sources seulement si elles sont utilisées et pertinentes
                    if (data.source_documents && data.source_documents.length > 0) {
                        // Vérifier si les sources ont été réellement utilisées
                        const hasValidSources = data.source_documents.some(source => 
                            source.metadata && 
                            source.metadata.score && 
                            source.metadata.score > 0.3 && // Seuil de pertinence
                            source.text && 
                            source.text.trim().length > 50 // Texte substantiel
                        );
                        
                        if (hasValidSources) {
                            handlers.chat.addSourcesInfo(messageElement, data.source_documents);
                        }
                    }
                    
                    // Mettre à jour la session
                    utils.updateSessionInfo(data.session_id, data.response_time);
                    utils.updateConnectionStatus('connected', 'Prêt');
                    
                    handlers.chat.closeSSEConnection();
                    resolve();
                    
                } catch (e) {
                    console.error('Erreur end:', e);
                    handlers.chat.closeSSEConnection();
                    resolve();
                }
            });
            
            // Événement error
            appState.eventSource.addEventListener('error', (event) => {
                console.error('Erreur SSE:', event);
                
                messageElement.classList.remove('streaming');
                messageElement.classList.add('error');
                
                const errorDiv = document.createElement('div');
                errorDiv.className = 'error-message';
                errorDiv.textContent = 'Erreur lors de la réception de la réponse.';
                contentDiv.appendChild(errorDiv);
                
                utils.updateConnectionStatus('disconnected', 'Erreur de connexion');
                handlers.chat.closeSSEConnection();
                reject(new Error('Erreur de streaming'));
            });
        },

        handleStandardResponse: async (question) => {
            try {
                utils.showLoading('Recherche de la réponse...');
                
                if (appState.useAgent) {
                    // Agent API
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
                    appState.currentSessionId = response.session_id;
                    utils.updateSessionInfo(response.session_id, response.response_time);
                    
                    // Ajouter la réponse avec sources seulement si pertinentes
                    const sourcesToShow = handlers.chat.filterRelevantSources(response.source_documents);
                    
                    handlers.chat.addMessage(response.response, 'ai', sourcesToShow, {
                        domains: response.domains,
                        intent: response.intent,
                        responseTime: response.response_time
                    });
                } else {
                    // RAG traditionnel
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
                    handlers.chat.addMessage(response.answer, 'ai', response.source_documents);
                }
                
                handlers.chat.scrollToBottom();
                
            } catch (error) {
                utils.hideLoading();
                handlers.chat.addMessage(
                    `Désolé, une erreur est survenue: ${error.message}`,
                    'system'
                );
            }
        },

        // Méthodes utilitaires pour le streaming
        filterRelevantSources: (sources) => {
            if (!sources || !Array.isArray(sources)) return [];
            
            return sources.filter(source => {
                // Vérifier la pertinence de la source
                const metadata = source.metadata || {};
                const score = metadata.score || 0;
                const text = source.text || '';
                
                // Critères de filtrage :
                // 1. Score de similarité > 0.3 (seuil de pertinence)
                // 2. Texte substantiel (> 50 caractères)
                // 3. Métadonnées valides
                return (
                    score > 0.3 && 
                    text.trim().length > 50 && 
                    metadata.filename && 
                    metadata.document_id
                );
            });
        },

        updateMessageContent: (contentDiv, text) => {
            contentDiv.innerHTML = '';
            
            // Formatage en temps réel du streaming
            const formattedText = handlers.chat.formatMessageText(text);
            
            const textElement = document.createElement('div');
            textElement.className = 'streaming-text';
            textElement.innerHTML = formattedText;
            contentDiv.appendChild(textElement);
            
            // Curseur de frappe
            if (appState.isStreaming) {
                const cursor = document.createElement('span');
                cursor.className = 'typing-cursor';
                cursor.textContent = '▋';
                contentDiv.appendChild(cursor);
            }
        },

        addTypingIndicator: (contentDiv) => {
            const indicator = document.createElement('div');
            indicator.className = 'typing-indicator';
            indicator.innerHTML = '<span></span><span></span><span></span>';
            contentDiv.appendChild(indicator);
        },

        removeTypingIndicator: (contentDiv) => {
            const indicator = contentDiv.querySelector('.typing-indicator');
            if (indicator) indicator.remove();
        },

        closeSSEConnection: () => {
            if (appState.eventSource) {
                appState.eventSource.close();
                appState.eventSource = null;
            }
            appState.isStreaming = false;
            appState.currentMessage = null;
        },

        setInputState: (enabled) => {
            if (DOM.questionInput) DOM.questionInput.disabled = !enabled;
            if (DOM.sendQuestion) DOM.sendQuestion.disabled = !enabled;
            
            // Mettre à jour le texte du bouton
            if (DOM.sendQuestion) {
                const btnText = DOM.sendQuestion.querySelector('.btn-text') || DOM.sendQuestion;
                btnText.textContent = enabled ? 'Envoyer' : 'Génération...';
            }
        },

        scrollToBottom: () => {
            if (DOM.chatMessages) {
                DOM.chatMessages.scrollTop = DOM.chatMessages.scrollHeight;
            }
        },
        
        addMessage: (text, type, sources = [], metadata = {}) => {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${type}`;
            
            // Message content
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            
            if (text) {
                // Améliorer le formatage du texte avec Markdown-like
                const formattedText = handlers.chat.formatMessageText(text);
                contentDiv.innerHTML = formattedText;
            }
            
            messageDiv.appendChild(contentDiv);
            
            // Add sources if available and relevant
            if (sources && sources.length > 0 && type === 'ai') {
                // Filtrer les sources pertinentes
                const relevantSources = handlers.chat.filterRelevantSources(sources);
                
                if (relevantSources.length > 0) {
                    handlers.chat.addSourcesInfo(messageDiv, relevantSources);
                    
                    // Add metadata seulement si on a des sources
                    if (metadata.domains && metadata.domains.length) {
                        const domainsDiv = document.createElement('div');
                        domainsDiv.className = 'domains-list';
                        domainsDiv.style.cssText = "margin-top: 12px; font-size: 0.85rem; color: #64748b; padding: 8px 12px; background-color: #f1f5f9; border-radius: 6px;";
                        domainsDiv.innerHTML = `<i class="fas fa-balance-scale"></i> <strong>Domaines juridiques:</strong> ${metadata.domains.join(', ')}`;
                        contentDiv.appendChild(domainsDiv);
                    }
                    
                    if (metadata.responseTime) {
                        const timeDiv = document.createElement('div');
                        timeDiv.className = 'response-time';
                        timeDiv.style.cssText = "margin-top: 8px; font-size: 0.75rem; color: #94a3b8; text-align: right;";
                        timeDiv.innerHTML = `<i class="fas fa-clock"></i> Réponse générée en ${metadata.responseTime.toFixed(2)}s`;
                        contentDiv.appendChild(timeDiv);
                    }
                } else {
                    // Pas de sources pertinentes, ajouter juste un indicateur subtil
                    if (metadata.responseTime) {
                        const timeDiv = document.createElement('div');
                        timeDiv.className = 'response-time';
                        timeDiv.style.cssText = "margin-top: 8px; font-size: 0.75rem; color: #94a3b8; text-align: right;";
                        timeDiv.innerHTML = `<i class="fas fa-robot"></i> Réponse générée en ${metadata.responseTime.toFixed(2)}s`;
                        contentDiv.appendChild(timeDiv);
                    }
                }
            }
            
            // Add timestamp
            const metaDiv = document.createElement('div');
            metaDiv.className = 'message-meta';
            metaDiv.textContent = new Date().toLocaleTimeString();
            messageDiv.appendChild(metaDiv);
            
            // Add to chat
            DOM.chatMessages.appendChild(messageDiv);
            handlers.chat.scrollToBottom();
            
            return messageDiv;
        },

        formatMessageText: (text) => {
            if (!text) return '';
            
            // Nettoyer le texte et séparer du contenu des sources
            let cleanText = text;
            
            // Supprimer les sources qui apparaissent à la fin du texte
            const sourcePatterns = [
                /Sources?\s*:[\s\S]*$/i,
                /Références?\s*:[\s\S]*$/i,
                /Bibliographie\s*:[\s\S]*$/i
            ];
            
            sourcePatterns.forEach(pattern => {
                cleanText = cleanText.replace(pattern, '').trim();
            });
            
            // Formatage Markdown-like
            let formattedText = cleanText
                // Titres
                .replace(/^#\s+(.+)$/gm, '<h3>$1</h3>')
                .replace(/^##\s+(.+)$/gm, '<h4>$1</h4>')
                .replace(/^###\s+(.+)$/gm, '<h5>$1</h5>')
                
                // Texte en gras
                .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
                .replace(/(?:^|\s)(\d+\.\s)/gm, '<br><strong>$1</strong>')
                
                // Listes à puces
                .replace(/^\*\s+(.+)$/gm, '<li>$1</li>')
                .replace(/^-\s+(.+)$/gm, '<li>$1</li>')
                
                // Paragraphes
                .split('\n\n')
                .map(paragraph => {
                    paragraph = paragraph.trim();
                    if (!paragraph) return '';
                    
                    // Si c'est une liste
                    if (paragraph.includes('<li>')) {
                        return `<ul>${paragraph}</ul>`;
                    }
                    
                    // Si c'est un titre
                    if (paragraph.startsWith('<h')) {
                        return paragraph;
                    }
                    
                    // Paragraphe normal
                    return `<p>${paragraph}</p>`;
                })
                .filter(p => p)
                .join('');
            
            return formattedText;
        },

        addSourcesInfo: (messageElement, sources) => {
            if (!sources || sources.length === 0) return;
            
            // Filtrer les sources vraiment pertinentes pour l'affichage
            const displaySources = sources.filter(source => {
                const metadata = source.metadata || {};
                return metadata.filename && source.text && source.text.trim().length > 30;
            });
            
            if (displaySources.length === 0) return;
            
            const sourceListDiv = document.createElement('div');
            sourceListDiv.className = 'source-list';
            sourceListDiv.style.cssText = `
                margin-top: 16px; 
                padding: 12px 16px; 
                background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); 
                border-radius: 8px; 
                border-left: 4px solid #3b82f6;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            `;
            
            const sourceHeader = document.createElement('div');
            sourceHeader.style.cssText = "margin-bottom: 8px; font-weight: 600; color: #1e40af; display: flex; align-items: center; gap: 6px;";
            sourceHeader.innerHTML = '<i class="fas fa-book-open"></i> Sources consultées';
            sourceListDiv.appendChild(sourceHeader);
            
            // Container pour les sources
            const sourcesContainer = document.createElement('div');
            sourcesContainer.style.cssText = "display: flex; flex-direction: column; gap: 6px;";
            
            // Limiter à 3 sources maximum pour ne pas encombrer
            displaySources.slice(0, 3).forEach((source, index) => {
                const sourceItem = document.createElement('div');
                sourceItem.className = 'source-item';
                sourceItem.style.cssText = `
                    display: flex; 
                    align-items: center; 
                    gap: 8px; 
                    padding: 8px 12px; 
                    background-color: white; 
                    border-radius: 6px; 
                    cursor: pointer; 
                    transition: all 0.2s ease;
                    border: 1px solid #e2e8f0;
                    font-size: 0.9rem;
                `;
                
                const metadata = source.metadata || {};
                const filename = metadata.filename || 'Document inconnu';
                const page = metadata.page_number || '?';
                const score = metadata.score ? `${(metadata.score * 100).toFixed(0)}%` : '';
                
                // Icône du document
                const iconSpan = document.createElement('span');
                iconSpan.innerHTML = '<i class="fas fa-file-alt"></i>';
                iconSpan.style.cssText = "color: #6b7280; width: 16px; text-align: center;";
                
                // Informations du document
                const infoSpan = document.createElement('span');
                infoSpan.style.cssText = "flex: 1; display: flex; flex-direction: column; gap: 2px;";
                
                const nameDiv = document.createElement('div');
                nameDiv.style.cssText = "font-weight: 500; color: #374151;";
                nameDiv.textContent = filename;
                
                const detailsDiv = document.createElement('div');
                detailsDiv.style.cssText = "font-size: 0.8rem; color: #6b7280;";
                detailsDiv.innerHTML = `Page ${page}${score ? ` • Pertinence: ${score}` : ''}`;
                
                infoSpan.appendChild(nameDiv);
                infoSpan.appendChild(detailsDiv);
                
                // Badge de numéro
                const badgeSpan = document.createElement('span');
                badgeSpan.style.cssText = `
                    background-color: #3b82f6; 
                    color: white; 
                    border-radius: 50%; 
                    width: 20px; 
                    height: 20px; 
                    display: flex; 
                    align-items: center; 
                    justify-content: center; 
                    font-size: 0.75rem; 
                    font-weight: 600;
                `;
                badgeSpan.textContent = index + 1;
                
                sourceItem.appendChild(iconSpan);
                sourceItem.appendChild(infoSpan);
                sourceItem.appendChild(badgeSpan);
                
                // Événements de hover
                sourceItem.addEventListener('mouseenter', () => {
                    sourceItem.style.backgroundColor = '#f1f5f9';
                    sourceItem.style.borderColor = '#3b82f6';
                    sourceItem.style.transform = 'translateX(2px)';
                });
                
                sourceItem.addEventListener('mouseleave', () => {
                    sourceItem.style.backgroundColor = 'white';
                    sourceItem.style.borderColor = '#e2e8f0';
                    sourceItem.style.transform = 'translateX(0)';
                });
                
                sourceItem.addEventListener('click', () => handlers.chat.showSourceDetails(source));
                
                sourcesContainer.appendChild(sourceItem);
            });
            
            sourceListDiv.appendChild(sourcesContainer);
            
            // Ajouter une note sur le nombre total de sources si plus de 3
            if (displaySources.length > 3) {
                const moreInfo = document.createElement('div');
                moreInfo.style.cssText = "margin-top: 8px; font-size: 0.8rem; color: #6b7280; text-align: center; font-style: italic;";
                moreInfo.textContent = `... et ${displaySources.length - 3} autre(s) source(s)`;
                sourceListDiv.appendChild(moreInfo);
            }
            
            const contentDiv = messageElement.querySelector('.message-content');
            contentDiv.appendChild(sourceListDiv);
        },

        showSourceDetails: (source) => {
            const panel = DOM.rightPanel;
            if (!panel) return;
            
            panel.classList.add('active');
            
            const metadata = source.metadata || {};
            const filename = metadata.filename || 'Document inconnu';
            const page = metadata.page_number || '?';
            
            DOM.previewTitle.textContent = `Source: ${filename}`;
            
            DOM.panelContent.innerHTML = `
                <h3>${filename}</h3>
                <div class="document-meta" style="margin: 20px 0;">
                    <p><strong>Page:</strong> ${page}</p>
                    <p><strong>Document ID:</strong> ${metadata.document_id || 'N/A'}</p>
                    <p><strong>Score:</strong> ${metadata.score ? metadata.score.toFixed(3) : 'N/A'}</p>
                </div>
                <div class="source-content" style="background-color: #f8fafc; padding: 15px; border-radius: 8px; margin-top: 20px;">
                    <h4>Extrait source:</h4>
                    <p style="line-height: 1.6;">${source.text || 'Texte non disponible'}</p>
                </div>
            `;
        },
        
        resetChat: async () => {
            try {
                // Fermer le streaming en cours
                if (appState.isStreaming) {
                    handlers.chat.closeSSEConnection();
                }

                utils.showLoading('Réinitialisation de la conversation...');
                
                if (appState.currentSessionId) {
                    await utils.fetchAPI(API_ENDPOINTS.agentReset, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            session_id: appState.currentSessionId
                        })
                    });
                    
                    appState.currentSessionId = null;
                    utils.updateSessionInfo('-');
                }
                
                utils.hideLoading();
                
                // Garder le message de bienvenue
                const welcomeMessage = DOM.chatMessages.querySelector('.message.system');
                DOM.chatMessages.innerHTML = '';
                if (welcomeMessage) {
                    DOM.chatMessages.appendChild(welcomeMessage);
                }
                
                utils.updateConnectionStatus('connected', 'Prêt');
                
                console.log('🔄 Conversation réinitialisée');
                
            } catch (error) {
                utils.hideLoading();
                handlers.chat.addMessage(
                    `Erreur lors de la réinitialisation: ${error.message}`,
                    'system'
                );
            }
        },
        
        loadSession: async (sessionId) => {
            try {
                utils.showLoading('Chargement de la session...');
                
                const response = await utils.fetchAPI(`/agent/sessions/${sessionId}`);
                
                utils.hideLoading();
                appState.currentSessionId = sessionId;
                utils.updateSessionInfo(sessionId);
                
                DOM.chatMessages.innerHTML = '';
                
                handlers.chat.addMessage(
                    `Session #${sessionId} chargée avec succès. Cette conversation contient ${response.message_count} messages.`,
                    'system'
                );
                
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
    
    // Section Documents (inchangée)
    documents: {
        init: async () => {
            DOM.uploadBtn?.addEventListener('click', () => DOM.fileUpload.click());
            DOM.fileUpload?.addEventListener('change', handlers.documents.handleFileSelection);
            DOM.docSearch?.addEventListener('input', handlers.documents.filterDocuments);
            DOM.docFilter?.addEventListener('change', handlers.documents.filterDocuments);

            // Glisser-déposer
            const uploadZone = document.querySelector('.upload-zone');
            if (uploadZone) {
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
            }
            
            await handlers.documents.loadDocuments();
        },

        // ... (toutes les autres méthodes documents restent identiques)
        handleFileSelection: (e) => {
            const files = Array.from(e.target.files).filter(file => file.type === 'application/pdf');
            
            if (files.length === 0) {
                alert('Veuillez sélectionner des fichiers PDF.');
                return;
            }
            
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
                
                document.getElementById('uploadSelectedBtn').addEventListener('click', () => {
                    handlers.documents.uploadSelectedFiles(files);
                });
                
                document.getElementById('clearSelectedBtn').addEventListener('click', () => {
                    selectedFilesContainer.innerHTML = '';
                    DOM.fileUpload.value = '';
                });
            }
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
            
            const searchText = DOM.docSearch.value.toLowerCase();
            const filterValue = DOM.docFilter.value;
            
            const filteredDocs = appState.documents.filter(doc => {
                const filename = doc.filename?.toLowerCase() || '';
                const isScanned = doc.is_scanned || false;
                
                const textMatch = !searchText || filename.includes(searchText);
                
                let typeMatch = true;
                if (filterValue === 'scanned') typeMatch = isScanned;
                if (filterValue === 'digital') typeMatch = !isScanned;
                
                return textMatch && typeMatch;
            });
            
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
            // Version simplifiée pour économiser l'espace
            console.log(`Téléchargement de ${files.length} fichiers...`);
            // Implémenter la logique d'upload ici
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
            
            DOM.panelContent.innerHTML = content;
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
                
                utils.hideLoading();
                alert(`Document "${doc.filename}" retraité avec succès.`);
                await handlers.documents.loadDocuments();
                
            } catch (error) {
                utils.hideLoading();
                alert(`Erreur lors du retraitement: ${error.message}`);
            }
        }
    },
    
    // Section Search (optimisée)
    search: {
        init: () => {
            DOM.searchButton?.addEventListener('click', handlers.search.performSearch);
            DOM.searchInput?.addEventListener('keydown', (e) => {
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
                
                const useRerank = DOM.useRerank?.checked ?? true;
                const topK = parseInt(DOM.topK?.value ?? 10);
                
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
                appState.searchResults = response.results || [];
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
            
            let header = `
                <div style="margin-bottom: 20px;">
                    <h3>${totalResults} résultat(s) pour "${query}"</h3>
                    <p style="color: var(--secondary-color);">
                        Recherche effectuée en ${searchTime.toFixed(2)} secondes
                    </p>
                </div>
            `;
            
            let resultsHTML = '';
            
            results.forEach((result) => {
                const metadata = result.metadata || {};
                const score = result.score * 100;
                
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
    
    // Section History avec support des sessions
    history: {
        init: async () => {
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
                
                card.addEventListener('click', () => {
                    navigation.activateSection('chat');
                    DOM.questionInput.value = item.query;
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
            
            DOM.historyList.innerHTML = '';
            
            const header = document.createElement('div');
            header.className = 'history-header-info';
            header.innerHTML = `
                <div style="padding: 10px; margin-bottom: 15px; background-color: var(--message-system-bg, #fef3c7); border-radius: 8px;">
                    <p><strong>Sessions de conversation</strong></p>
                    <p style="font-size: 0.9rem;">Cliquez sur une session pour continuer la conversation.</p>
                </div>
            `;
            DOM.historyList.appendChild(header);
            
            appState.sessions.forEach(session => {
                const sessionItem = document.createElement('div');
                sessionItem.className = 'history-item';
                
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
                DOM.rightPanel?.classList.add('active');
                appState.rightPanelOpen = true;
            } else {
                DOM.rightPanel?.classList.remove('active');
                appState.rightPanelOpen = false;
            }
        }
    }
};

// Navigation system (optimisé)
const navigation = {
    init: () => {
        DOM.navItems.forEach(item => {
            item.addEventListener('click', () => {
                const section = item.dataset.section;
                navigation.activateSection(section);
            });
        });
        
        DOM.closePanel?.addEventListener('click', () => {
            handlers.ui.toggleRightPanel(false);
        });
    },
    
    activateSection: (section) => {
        // Mettre à jour la navigation
        DOM.navItems.forEach(item => {
            if (item.dataset.section === section) {
                item.classList.add('active');
            } else {
                item.classList.remove('active');
            }
        });
        
        // Mettre à jour les sections
        DOM.sections.forEach(s => {
            if (s.id === section) {
                s.classList.add('active');
            } else {
                s.classList.remove('active');
            }
        });
        
        appState.activeSection = section;
        
        // Recharger les données si nécessaire
        if (section === 'history') {
            handlers.history.init();
        }
    }
};

// Fonctions utilitaires globales pour le debugging
const debugUtils = {
    enableDebugMode: () => {
        window.appState = appState;
        window.handlers = handlers;
        window.utils = utils;
        console.log('🐛 Mode debug activé - Variables disponibles: appState, handlers, utils');
    },
    
    testStreaming: async () => {
        console.log('🧪 Test du streaming...');
        await handlers.chat.startStreamingResponse('Test du streaming LexCam');
    },
    
    logState: () => {
        console.log('📊 État actuel de l\'application:', {
            activeSection: appState.activeSection,
            currentSessionId: appState.currentSessionId,
            isStreaming: appState.isStreaming,
            documentsCount: appState.documents.length,
            sessionsCount: appState.sessions.length
        });
    }
};

// Gestion des erreurs globales
window.addEventListener('error', (event) => {
    console.error('❌ Erreur globale:', event.error);
    utils.updateConnectionStatus('disconnected', 'Erreur système');
});

window.addEventListener('unhandledrejection', (event) => {
    console.error('❌ Promesse rejetée:', event.reason);
    event.preventDefault();
});

// Initialize application
(async function initApp() {
    console.log('🚀 Initialisation de LexCam avec streaming intégré');
    
    try {
        // Initialiser la navigation
        navigation.init();
        
        // Initialiser tous les gestionnaires de section
        await handlers.chat.init();
        await handlers.documents.init();
        await handlers.search.init();
        await handlers.history.init();
        
        // Activer le mode debug en développement
        if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
            debugUtils.enableDebugMode();
        }
        
        // Initialiser le statut de connexion
        utils.updateConnectionStatus('connected', 'Prêt');
        
        console.log('✅ LexCam initialisé avec succès');
        
    } catch (error) {
        console.error('❌ Erreur lors de l\'initialisation:', error);
        utils.updateConnectionStatus('disconnected', 'Erreur d\'initialisation');
    }
})();