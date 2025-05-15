/**
 * Module JavaScript pour l'intégration de l'interface utilisateur avec le système RAG juridique camerounais
 * basé sur LangChain. Ce module gère l'interactivité du chat, le streaming des réponses et la gestion des sessions.
 */

// Configuration de l'API
const API_BASE_URL = 'http://localhost:8000/api';
const API_ENDPOINTS = {
    chat: '/langchain/chat',
    chatStream: '/langchain/chat/stream',
    resetChat: '/langchain/chat/reset',
    sessions: '/langchain/chat/sessions',
    sessionInfo: '/langchain/chat/sessions'
};

// État de l'application
const chatState = {
    currentSessionId: null,
    isStreaming: false,
    messageQueue: [],
    isProcessing: false,
    eventSource: null
};

/**
 * Initialise le module de chat LangChain
 */
function initLangChainChat() {
    console.log('Initialisation du module de chat LangChain');
    
    // Éléments DOM
    const chatContainer = document.getElementById('chatMessages');
    const questionInput = document.getElementById('questionInput');
    const sendButton = document.getElementById('sendQuestion');
    const resetButton = document.getElementById('resetChat');
    const sessionSelector = document.getElementById('sessionSelector');
    
    // Vérifier que les éléments DOM nécessaires sont présents
    if (!chatContainer || !questionInput || !sendButton) {
        console.error("Éléments DOM du chat non trouvés");
        return;
    }
    
    // Événements
    sendButton.addEventListener('click', handleSendQuestion);
    
    // Ajouter un gestionnaire pour la touche Entrée dans la zone de texte
    questionInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendQuestion();
        }
    });
    
    // Ajouter un gestionnaire pour le bouton de réinitialisation si présent
    if (resetButton) {
        resetButton.addEventListener('click', resetConversation);
    }
    
    // Ajouter un gestionnaire pour le sélecteur de session si présent
    if (sessionSelector) {
        sessionSelector.addEventListener('change', loadSelectedSession);
        // Charger les sessions disponibles
        loadAvailableSessions();
    }
    
    console.log('Module de chat LangChain initialisé');
}

/**
 * Gère l'envoi d'une question au système RAG
 */
async function handleSendQuestion() {
    const questionInput = document.getElementById('questionInput');
    const query = questionInput.value.trim();
    
    if (!query) return;
    
    try {
        // Ajouter le message de l'utilisateur à l'interface
        addMessage(query, 'user');
        
        // Effacer l'entrée
        questionInput.value = '';
        
        // Désactiver l'entrée pendant le traitement
        questionInput.disabled = true;
        document.getElementById('sendQuestion').disabled = true;
        
        // Afficher l'indicateur de chargement
        const loadingMessage = addMessage('...', 'ai loading');
        
        // Mode streaming ou classique selon les préférences utilisateur
        const useStreaming = document.getElementById('useStreaming')?.checked || true;
        
        if (useStreaming) {
            // Utiliser l'API de streaming
            await handleStreamingResponse(query, loadingMessage);
        } else {
            // Utiliser l'API classique
            await handleStandardResponse(query, loadingMessage);
        }
    } catch (error) {
        console.error('Erreur lors de l\'envoi de la question:', error);
        
        // Supprimer le message de chargement s'il existe encore
        const loadingElement = document.querySelector('.message.loading');
        if (loadingElement) {
            loadingElement.remove();
        }
        
        // Afficher l'erreur
        addMessage(`Désolé, une erreur est survenue: ${error.message}`, 'system');
    } finally {
        // Réactiver l'entrée
        questionInput.disabled = false;
        document.getElementById('sendQuestion').disabled = false;
    }
}

/**
 * Gère la réponse standard (non streaming)
 */
async function handleStandardResponse(query, loadingElement) {
    try {
        const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.chatStream}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query: query,
                session_id: sessionId
            })
        })
        
        if (!response.ok) {
            throw new Error(`Erreur serveur: ${response.status} ${response.statusText}`);
        }
        
        const data = await response.json();
        
        // Mettre à jour l'ID de session
        chatState.currentSessionId = data.session_id;
        updateSessionInfo(data.session_id);
        
        // Supprimer le message de chargement
        if (loadingElement) {
            loadingElement.remove();
        }
        
        // Ajouter la réponse
        const messageElement = addMessage(data.response, 'ai');
        
        // Ajouter les sources si présentes
        if (data.source_documents && data.source_documents.length > 0) {
            addSourcesInfo(messageElement, data.source_documents);
        }
        
        // Faire défiler jusqu'au bas du chat
        scrollToBottom();
    } catch (error) {
        console.error('Erreur lors de la récupération de la réponse:', error);
        throw error;
    }
}

/**
 * Gère la réponse en streaming
 */
async function handleStreamingResponse(query, loadingElement) {
    try {
        // Annuler tout streaming précédent
        if (chatState.eventSource) {
            chatState.eventSource.close();
        }
        
        // Créer un nouvel élément pour la réponse streaming
        let messageContent = '';
        
        // Supprimer le message de chargement
        if (loadingElement) {
            loadingElement.remove();
        }
        
        // Créer un élément pour la réponse
        const messageElement = addMessage('', 'ai streaming');
        
        // Configurer les Server-Sent Events
        const url = new URL(`${API_BASE_URL}${API_ENDPOINTS.chatStream}`);
        const params = new URLSearchParams({
            query: query,
            session_id: chatState.currentSessionId || '',
            streaming: true
        });
        
        chatState.eventSource = new EventSource(`${url}?${params}`);
        
        // Gestion des événements SSE
        chatState.eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                messageContent += data;
                messageElement.querySelector('.message-content').textContent = messageContent;
                scrollToBottom();
            } catch (e) {
                console.error('Erreur lors du parsing des données streaming:', e);
            }
        };
        
        chatState.eventSource.addEventListener('start', (event) => {
            try {
                const data = JSON.parse(event.data);
                chatState.currentSessionId = data.session_id;
                updateSessionInfo(data.session_id);
            } catch (e) {
                console.error('Erreur lors du parsing des données de début de streaming:', e);
            }
        });
        
        // REMPLACER CE BLOC PAR LE CODE SUGGÉRÉ
        chatState.eventSource.addEventListener('token', (event) => {
            try {
                // Le token est déjà une chaîne, pas besoin de parser du JSON
                messageContent += event.data;
                messageElement.querySelector('.message-content').textContent = messageContent;
                scrollToBottom();
            } catch (e) {
                console.error('Erreur lors du traitement du token streaming:', e);
            }
        });
        
        chatState.eventSource.addEventListener('end', (event) => {
            try {
                const data = JSON.parse(event.data);
                
                // Mettre à jour l'ID de session
                chatState.currentSessionId = data.session_id;
                updateSessionInfo(data.session_id);
                
                // Finaliser le message
                messageElement.classList.remove('streaming');
                
                // Fermer la connexion streaming
                chatState.eventSource.close();
                chatState.eventSource = null;
            } catch (e) {
                console.error('Erreur lors du parsing des données de fin de streaming:', e);
            }
        });
        
        chatState.eventSource.addEventListener('error', (event) => {
            console.error('Erreur SSE:', event);
            messageElement.classList.remove('streaming');
            messageElement.classList.add('error');
            
            // Ajouter un message d'erreur
            const errorDiv = document.createElement('div');
            errorDiv.className = 'error-message';
            errorDiv.textContent = 'Erreur lors du streaming de la réponse.';
            messageElement.appendChild(errorDiv);
            
            // Fermer la connexion streaming
            chatState.eventSource.close();
            chatState.eventSource = null;
        });
    } catch (error) {
        console.error('Erreur lors de l\'initialisation du streaming:', error);
        throw error;
    }
}

/**
 * Ajoute un message dans la fenêtre de chat
 */
function addMessage(text, type) {
    const chatContainer = document.getElementById('chatMessages');
    
    // Créer l'élément de message
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    
    // Créer le contenu du message
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = text;
    
    // Ajouter le contenu au message
    messageDiv.appendChild(contentDiv);
    
    // Ajouter les métadonnées (heure, etc.)
    const metaDiv = document.createElement('div');
    metaDiv.className = 'message-meta';
    metaDiv.textContent = new Date().toLocaleTimeString();
    messageDiv.appendChild(metaDiv);
    
    // Ajouter le message au conteneur
    chatContainer.appendChild(messageDiv);
    
    // Faire défiler vers le bas
    scrollToBottom();
    
    // Retourner l'élément du message pour d'éventuelles modifications ultérieures
    return messageDiv;
}

/**
 * Ajoute des informations sur les sources à un message
 */
function addSourcesInfo(messageElement, sources) {
    if (!sources || sources.length === 0) return;
    
    // Créer la liste des sources
    const sourceListDiv = document.createElement('div');
    sourceListDiv.className = 'source-list';
    
    // Ajouter l'en-tête
    const sourceHeader = document.createElement('p');
    sourceHeader.innerHTML = '<strong>Sources:</strong>';
    sourceListDiv.appendChild(sourceHeader);
    
    // Ajouter chaque source
    sources.slice(0, 3).forEach(source => {
        const sourceItem = document.createElement('div');
        sourceItem.className = 'source-item';
        
        const metadata = source.metadata || {};
        const filename = metadata.filename || 'Document inconnu';
        const page = metadata.page_number || '?';
        
        sourceItem.innerHTML = `<i class="fas fa-file-alt"></i> ${filename} (page ${page})`;
        
        // Rendre la source cliquable pour afficher le texte complet
        sourceItem.addEventListener('click', () => {
            showSourceDetails(source);
        });
        
        sourceListDiv.appendChild(sourceItem);
    });
    
    // Ajouter à l'élément message
    const contentDiv = messageElement.querySelector('.message-content');
    contentDiv.appendChild(sourceListDiv);
}

/**
 * Affiche les détails d'une source dans un panneau latéral
 */
function showSourceDetails(source) {
    // Obtenir ou créer le panneau latéral
    let panel = document.getElementById('rightPanel');
    if (!panel) {
        console.warn('Panneau latéral non trouvé, impossible d\'afficher les détails de la source');
        return;
    }
    
    // Activer le panneau
    panel.classList.add('active');
    
    // Remplir le contenu du panneau
    const metadata = source.metadata || {};
    const filename = metadata.filename || 'Document inconnu';
    const page = metadata.page_number || '?';
    
    // Titre du panneau
    document.getElementById('previewTitle').textContent = `Source: ${filename}`;
    
    // Contenu du panneau
    document.getElementById('panelContent').innerHTML = `
        <h3>${filename}</h3>
        <div class="document-meta" style="margin: 20px 0;">
            <p><strong>Page:</strong> ${page}</p>
            <p><strong>Document ID:</strong> ${metadata.document_id || 'N/A'}</p>
            <p><strong>Chunk ID:</strong> ${metadata.chunk_id || 'N/A'}</p>
        </div>
        <div class="source-content" style="background-color: #f8fafc; padding: 15px; border-radius: 8px; margin-top: 20px;">
            <h4>Extrait source:</h4>
            <p>${source.text || 'Texte non disponible'}</p>
        </div>
    `;
}

/**
 * Fait défiler le conteneur de chat vers le bas
 */
function scrollToBottom() {
    const chatContainer = document.getElementById('chatMessages');
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

/**
 * Charge la liste des sessions disponibles
 */
async function loadAvailableSessions() {
    try {
        const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.sessions}`);
        
        if (!response.ok) {
            throw new Error(`Erreur serveur: ${response.status} ${response.statusText}`);
        }
        
        const data = await response.json();
        
        // Mettre à jour le sélecteur de session
        const sessionSelector = document.getElementById('sessionSelector');
        if (!sessionSelector) return;
        
        // Vider le sélecteur sauf l'option "Nouvelle conversation"
        while (sessionSelector.options.length > 1) {
            sessionSelector.remove(1);
        }
        
        // Ajouter les sessions disponibles
        data.sessions.forEach(session => {
            const option = document.createElement('option');
            option.value = session.session_id;
            
            // Formater la date
            const date = new Date(session.last_time * 1000);
            const formattedDate = date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
            
            option.textContent = `${session.first_query.substring(0, 30)}... (${formattedDate})`;
            sessionSelector.appendChild(option);
        });
    } catch (error) {
        console.error('Erreur lors du chargement des sessions:', error);
    }
}

/**
 * Charge une session sélectionnée
 */
async function loadSelectedSession() {
    const sessionSelector = document.getElementById('sessionSelector');
    if (!sessionSelector) return;
    
    const sessionId = sessionSelector.value;
    
    // Si "Nouvelle conversation" est sélectionné, réinitialiser
    if (sessionId === 'new') {
        resetConversation();
        return;
    }
    
    try {
        // Afficher un indicateur de chargement
        const chatContainer = document.getElementById('chatMessages');
        chatContainer.innerHTML = '<div class="loading-indicator">Chargement de la conversation...</div>';
        
        // Charger les informations de la session
        const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.sessionInfo}/${sessionId}`);
        
        if (!response.ok) {
            throw new Error(`Erreur serveur: ${response.status} ${response.statusText}`);
        }
        
        const data = await response.json();
        
        // Mettre à jour l'ID de session courant
        chatState.currentSessionId = parseInt(sessionId);
        
        // Effacer le chat
        chatContainer.innerHTML = '';
        
        // Ajouter les messages de la conversation
        data.messages.forEach(msg => {
            const type = msg.type === 'human' ? 'user' : (msg.type === 'ai' ? 'ai' : 'system');
            addMessage(msg.content, type);
        });
        
        // Mettre à jour l'affichage de la session
        updateSessionInfo(sessionId);
        
    } catch (error) {
        console.error('Erreur lors du chargement de la session:', error);
        addMessage(`Erreur lors du chargement de la session: ${error.message}`, 'system');
    }
}

/**
 * Réinitialise la conversation
 */
async function resetConversation() {
    try {
        // Appeler l'API pour réinitialiser
        const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.resetChat}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                session_id: chatState.currentSessionId
            })
        });
        
        if (!response.ok) {
            throw new Error(`Erreur serveur: ${response.status} ${response.statusText}`);
        }
        
        const data = await response.json();
        
        // Mettre à jour l'ID de session
        chatState.currentSessionId = data.new_session_id;
        
        // Effacer le chat
        const chatContainer = document.getElementById('chatMessages');
        chatContainer.innerHTML = '';
        
        // Ajouter un message système
        addMessage('Conversation réinitialisée. Posez une nouvelle question!', 'system');
        
        // Mettre à jour l'affichage de la session
        updateSessionInfo(data.new_session_id);
        
        // Mettre à jour le sélecteur de session
        const sessionSelector = document.getElementById('sessionSelector');
        if (sessionSelector) {
            sessionSelector.value = 'new';
        }
        
    } catch (error) {
        console.error('Erreur lors de la réinitialisation de la conversation:', error);
        addMessage(`Erreur lors de la réinitialisation: ${error.message}`, 'system');
    }
}

/**
 * Met à jour l'affichage des informations de session
 */
function updateSessionInfo(sessionId) {
    const sessionInfo = document.getElementById('sessionInfo');
    if (!sessionInfo) return;
    
    sessionInfo.textContent = `Session active: ${sessionId}`;
}

/**
 * Gère le feedback utilisateur sur les réponses
 */
function handleFeedback(messageId, rating) {
    // Récupérer l'élément de message
    const messageElement = document.getElementById(messageId);
    if (!messageElement) return;
    
    // Mettre à jour l'apparence des boutons de feedback
    const feedbackButtons = messageElement.querySelectorAll('.feedback-button');
    feedbackButtons.forEach(btn => {
        btn.classList.remove('active');
        if (btn.getAttribute('data-rating') === rating.toString()) {
            btn.classList.add('active');
        }
    });
    
    // Envoyer le feedback au serveur
    submitFeedback({
        message_id: messageId,
        session_id: chatState.currentSessionId,
        rating: rating,
        timestamp: Date.now() / 1000
    });
}

/**
 * Envoie le feedback à l'API
 */
async function submitFeedback(feedbackData) {
    try {
        const response = await fetch(`${API_BASE_URL}/langchain/feedback`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(feedbackData)
        });
        
        if (!response.ok) {
            throw new Error(`Erreur lors de l'envoi du feedback: ${response.status}`);
        }
        
        console.log('Feedback envoyé avec succès');
    } catch (error) {
        console.error('Erreur lors de l\'envoi du feedback:', error);
    }
}

// Exporter les fonctions pour une utilisation externe
window.LangChainChat = {
    init: initLangChainChat,
    send: handleSendQuestion,
    reset: resetConversation,
    loadSessions: loadAvailableSessions,
    feedback: handleFeedback
};

// Initialiser le module quand le DOM est chargé
document.addEventListener('DOMContentLoaded', initLangChainChat);