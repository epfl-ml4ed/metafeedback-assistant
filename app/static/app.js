/* ============================================================================
   GLOBAL STATE & CONFIG
   ============================================================================ */

let appState = {
    participantID: null,
    currentStage: null,
    contextData: {
        problem: '',
        solution: ''
    },
    stageData: {
        1: {
            feedback: '',
            classifications: []
        },
        2: {
            feedback: '',
            suggestions: null,
            selectedStyle: null
        },
        3: {
            feedback: '',
            chatHistory: [],
            selectedStyle: null
        }
    },
    allLogs: []
};

const API_BASE = 'http://localhost:8000/api';

/* ============================================================================
   PARTICIPANT ID GENERATION
   ============================================================================ */

function generateParticipantID() {
    const date = new Date();
    const dateStr = date.toISOString().slice(0, 10).replace(/-/g, '');
    const randomNum = Math.floor(Math.random() * 90000) + 10000; // 5-digit random
    return `PID-${dateStr}-${randomNum}`;
}

function initializeParticipant() {
    appState.participantID = generateParticipantID();
    document.getElementById('participantID').textContent = appState.participantID;
    logEvent('participant_initialized', {
        participant_id: appState.participantID,
        timestamp: new Date().toISOString()
    });
}

/* ============================================================================
   CONTEXT LOADING (Problem & Solution)
   ============================================================================ */

async function loadContextData() {
    try {
        const response = await fetch('/static/context.json');
        const data = await response.json();
        appState.contextData = {
            problem: data.problem || '',
            solution: data.solution || ''
        };
        renderContextAllPages();
    } catch (error) {
        console.error('Error loading context:', error);
        appState.contextData = {
            problem: 'Error loading problem',
            solution: 'Error loading solution'
        };
    }
}

function renderContextAllPages() {
    const stages = [1, 2, 3];
    stages.forEach(stage => {
        const problemEl = document.getElementById(`stage${stage}-problem`);
        const solutionEl = document.getElementById(`stage${stage}-solution`);
        if (problemEl) problemEl.textContent = appState.contextData.problem;
        if (solutionEl) solutionEl.textContent = appState.contextData.solution;
    });
}

/* ============================================================================
   PAGE NAVIGATION
   ============================================================================ */

function showPage(pageId) {
    // Hide all pages
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    // Show target page
    const targetPage = document.getElementById(pageId);
    if (targetPage) {
        targetPage.classList.add('active');
    }
}

function goToStage(stage) {
    appState.currentStage = stage;
    updateProgressIndicator();

    switch (stage) {
        case 1:
            showPage('page-stage1');
            // Load last feedback if exists
            const feedbackEl = document.getElementById('stage1-feedback');
            if (appState.stageData[1].feedback) {
                feedbackEl.value = appState.stageData[1].feedback;
            }
            break;

        case 2:
            showPage('page-stage2');
            // Carry forward feedback from stage 1
            const stage2FeedbackEl = document.getElementById('stage2-feedback');
            stage2FeedbackEl.value = appState.stageData[1].feedback;
            appState.stageData[2].feedback = appState.stageData[1].feedback;
            logEvent('stage_2_entered', {
                stage: 2,
                initial_feedback: appState.stageData[1].feedback
            });
            break;

        case 3:
        showPage('page-stage3');
        // Carry forward feedback
        const stage3FeedbackEl = document.getElementById('stage3-feedback');
        stage3FeedbackEl.value = appState.stageData[2].feedback;
        appState.stageData[3].feedback = appState.stageData[2].feedback;
        
        // Initialize chat with context
        document.getElementById('stage3-chat-messages').innerHTML = '';
        appState.stageData[3].chatHistory = [];
        
        // Display the selected suggestion as reference 
        displaySelectedSuggestion();
        
        logEvent('stage_3_entered', {
            stage: 3,
            initial_feedback: appState.stageData[2].feedback,
            selected_style: appState.stageData[2].selectedStyle
        });
        break;
    }
}

function updateProgressIndicator() {
    const stageElements = document.querySelectorAll('.stage');
    stageElements.forEach(el => {
        const stageNum = parseInt(el.dataset.stage);
        el.classList.remove('active', 'completed');

        if (stageNum === appState.currentStage) {
            el.classList.add('active');
        } else if (stageNum < appState.currentStage) {
            el.classList.add('completed');
        }
    });
}

/* ============================================================================
   HELPER: Find sentences in text and their classifications
   ============================================================================ */

function findSentenceMatches(feedback, classifications) {
    /**
     * Creates HTML spans with colored underlines for each classified sentence.
     * Matches sentences from feedback text with their classifications.
     * Only highlights sentences that match one of the three rubric categories.
     * Skips "none" classifications (no highlight).
     */
    let processedText = feedback;

    // Sort classifications by sentence length (longest first) to avoid substring conflicts
    const sorted = [...classifications].sort((a, b) => b.sentence.length - a.sentence.length);

    sorted.forEach(item => {
        if (item.sentence && item.sentence.trim()) {
            // Only highlight if it matches one of the three categories
            let className = null;
            if (item.current_state === 1) {
                className = 'current-state';
            } else if (item.next_steps === 1) {
                className = 'next-steps';
            } else if (item.strategy === 1) {
                className = 'strategy';
            }
            
            // Only create span if classified (skip if "none")
            if (className) {
                const sentence = item.sentence.trim();
                const regex = new RegExp(`(${sentence.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
                processedText = processedText.replace(regex, `<span class="classified-span ${className}">$1</span>`);
            }
        }
    });

    return processedText;
}

/* ============================================================================
   STAGE 1: CLASSIFICATION
   ============================================================================ */

async function stage1Analyze() {
    const feedbackEl = document.getElementById('stage1-feedback');
    const feedback = feedbackEl.value.trim();

    if (!feedback) {
        alert('Please enter some feedback before analyzing.');
        return;
    }

    appState.stageData[1].feedback = feedback;

    try {
        const response = await fetch(`${API_BASE}/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                statement: appState.contextData.problem,
                student_solution: appState.contextData.solution,
                feedback: feedback
            })
        });

        const data = await response.json();
        appState.stageData[1].classifications = data.classified_sentences;

        renderStage1ClassificationPreview(feedback, data.classified_sentences);

        logEvent('stage_1_analyze', {
            stage: 1,
            feedback: feedback,
            classifications: data.classified_sentences,
            timestamp: new Date().toISOString()
        });

    } catch (error) {
        console.error('Error analyzing feedback:', error);
        alert('Error analyzing feedback. Please try again.');
    }
}

function renderStage1ClassificationPreview(feedback, classifications) {
    const container = document.getElementById('stage1-classifications');
    container.innerHTML = '';

    if (!feedback.trim()) {
        container.innerHTML = '<p style="color: var(--text-tertiary); font-size: 13px;">No feedback to preview.</p>';
        return;
    }

    // Create preview box
    const previewBox = document.createElement('div');
    previewBox.className = 'classification-preview-box';
    
    const previewHeader = document.createElement('div');
    previewHeader.className = 'preview-header';
    previewHeader.textContent = 'Classification Preview';
    previewBox.appendChild(previewHeader);

    const previewContent = document.createElement('div');
    previewContent.className = 'preview-content';
    
    // Generate HTML with colored spans
    const htmlContent = findSentenceMatches(feedback, classifications);
    previewContent.innerHTML = htmlContent;
    
    previewBox.appendChild(previewContent);
    container.appendChild(previewBox);
}

/* ============================================================================
   STAGE 2: LLM SUGGESTIONS & REFINEMENT
   ============================================================================ */

async function stage2Analyze() {
    const feedbackEl = document.getElementById('stage2-feedback');
    const feedback = feedbackEl.value.trim();

    if (!feedback) {
        alert('Please enter some feedback before getting suggestions.');
        return;
    }

    appState.stageData[2].feedback = feedback;

    try {
        // First classify the feedback
        const classifyResponse = await fetch(`${API_BASE}/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                statement: appState.contextData.problem,
                student_solution: appState.contextData.solution,
                feedback: feedback
            })
        });

        const classifyData = await classifyResponse.json();

        // Store suggestions (which come from the analyze endpoint)
        appState.stageData[2].suggestions = classifyData.llm_suggestions;

        renderStage2Suggestions(classifyData.llm_suggestions);

        logEvent('stage_2_analyze', {
            stage: 2,
            feedback: feedback,
            classifications: classifyData.classified_sentences,
            suggestions: classifyData.llm_suggestions,
            timestamp: new Date().toISOString()
        });

    } catch (error) {
        console.error('Error getting suggestions:', error);
        alert('Error getting suggestions. Please try again.');
    }
}

function renderStage2Suggestions(suggestions) {
    const container = document.getElementById('stage2-suggestions');
    container.innerHTML = '';

    if (!suggestions || !suggestions.micro || !suggestions.coaching) {
        container.innerHTML = '<p style="color: var(--text-tertiary);">No suggestions available.</p>';
        return;
    }

    // Micro suggestions
    const microBox = document.createElement('div');
    microBox.className = 'suggestion-box micro';
    microBox.innerHTML = `
        <div class="suggestion-title">📎 Micro Suggestions</div>
        <div class="suggestion-content">${suggestions.micro}</div>
        <button class="btn btn-secondary" onclick="selectSuggestionStyle('micro')" style="margin-top: var(--spacing-md); width: 100%;">
            Choose Micro Style
        </button>
    `;
    container.appendChild(microBox);

    // Coaching suggestions
    const coachingBox = document.createElement('div');
    coachingBox.className = 'suggestion-box coaching';
    coachingBox.innerHTML = `
        <div class="suggestion-title">💭 Coaching Suggestions</div>
        <div class="suggestion-content">${suggestions.coaching}</div>
        <button class="btn btn-secondary" onclick="selectSuggestionStyle('coaching')" style="margin-top: var(--spacing-md); width: 100%;">
            Choose Coaching Style
        </button>
    `;
    container.appendChild(coachingBox);
}

function selectSuggestionStyle(style) {
    appState.stageData[2].selectedStyle = style;
    alert(`You selected: ${style.charAt(0).toUpperCase() + style.slice(1)} Style`);
    logEvent('stage_2_style_selected', {
        stage: 2,
        selected_style: style,
        timestamp: new Date().toISOString()
    });
}

/* ============================================================================
   STAGE 3: CHAT REFINEMENT
   ============================================================================ */

async function stage3SendMessage() {
    const chatInputEl = document.getElementById('stage3-chat-input');
    const feedbackEl = document.getElementById('stage3-feedback');
    
    const message = chatInputEl.value.trim();
    const currentFeedback = feedbackEl.value.trim();

    if (!message) {
        alert('Please enter a message.');
        return;
    }

    // Update feedback from textarea
    appState.stageData[3].feedback = currentFeedback;

    // Add user message to chat
    appendChatMessage('user', message);
    appState.stageData[3].chatHistory.push({
        role: 'user',
        content: message
    });

    chatInputEl.value = '';
    chatInputEl.focus();

    try {
        // Get classifications for current feedback
        // alert('classified_sentences' + JSON.stringify(appState.stageData[1].classifications) + ' \nsuggestions: ' + JSON.stringify( appState.stageData[2].suggestions));

        const classifyResponse = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                style: appState.stageData[2].selectedStyle || 'coaching',
                history: appState.stageData[3].chatHistory,
                context: {
                    problem: appState.contextData.problem,
                    solution: appState.contextData.solution,
                    feedback: currentFeedback,
                    classified_sentences: appState.stageData[1].classifications,
                    suggestions: appState.stageData[2].suggestions
                }
            })
        });

        const classifyData = await classifyResponse.json();

        // Build context for chat
        const context = {
            problem: appState.contextData.problem,
            solution: appState.contextData.solution,
            feedback: currentFeedback,
            classified_sentences: classifyData.classified_sentences,
            suggestions: classifyData.llm_suggestions
        };

        // Send to chat endpoint
        const chatResponse = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                style: appState.stageData[2].selectedStyle || 'coaching',
                history: appState.stageData[3].chatHistory,
                context: context
            })
        });

        const chatData = await chatResponse.json();
        const assistantReply = chatData.reply;

        appendChatMessage('assistant', assistantReply);
        appState.stageData[3].chatHistory.push({
            role: 'assistant',
            content: assistantReply
        });

        logEvent('stage_3_chat_message', {
            stage: 3,
            user_message: message,
            assistant_reply: assistantReply,
            feedback: currentFeedback,
            timestamp: new Date().toISOString()
        });

    } catch (error) {
        console.error('Error sending chat message:', error);
        appendChatMessage('assistant', 'Error: Could not get response. Please try again.');
    }
}

function appendChatMessage(role, text) {
    const container = document.getElementById('stage3-chat-messages');
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${role}`;
    
    const bubble = document.createElement('div');
    bubble.className = 'chat-bubble';
    bubble.textContent = text;
    
    messageDiv.appendChild(bubble);
    container.appendChild(messageDiv);
    
    // Auto-scroll to bottom
    container.scrollTop = container.scrollHeight;
}

function stage3EndChat() {
    console.log('End Chat button clicked');
    const feedbackEl = document.getElementById('stage3-feedback');
    const finalFeedback = feedbackEl.value.trim();

    if (!finalFeedback) {
        alert('Please ensure your feedback is complete before submitting.');
        return;
    }

    appState.stageData[3].feedback = finalFeedback;

    logEvent('stage_3_completed', {
        stage: 3,
        final_feedback: finalFeedback,
        chat_history: appState.stageData[3].chatHistory,
        timestamp: new Date().toISOString()
    });

    // Save final state to backend
    saveFinalState();

    // Show completion page
    document.getElementById('completion-participant-id').textContent = appState.participantID;
    showPage('page-completion');
}

function displaySelectedSuggestion() {
    const container = document.getElementById('stage3-selected-suggestion');
    const titleEl = document.getElementById('suggestion-title');
    const contentEl = document.getElementById('suggestion-content');
    
    if (!container || !titleEl || !contentEl) {
        console.log('Container elements not found');
        return;
    }
    
    const selectedStyle = appState.stageData[2].selectedStyle || 'coaching';
    const suggestions = appState.stageData[2].suggestions;

    if (!suggestions) {
        container.style.display = 'none';
        return;
    }

    const suggestionText = selectedStyle === 'micro' ? suggestions.micro : suggestions.coaching;
    const styleTitle = selectedStyle === 'micro' ? '📎 Micro Suggestions' : '💭 Coaching Suggestions';

    // Just update text content, don't create HTML
    titleEl.textContent = styleTitle + ' (Reference)';
    contentEl.textContent = suggestionText;
    
    // Show the container
    container.style.display = 'block';
}

/* ============================================================================
   LOGGING SYSTEM
   ============================================================================ */

function logEvent(eventType, data) {
    const logEntry = {
        participant_id: appState.participantID,
        event_type: eventType,
        data: data,
        timestamp: new Date().toISOString()
    };

    appState.allLogs.push(logEntry);

    // Send to backend
    fetch(`${API_BASE}/log_revision`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(logEntry)
    }).catch(err => console.error('Logging error:', err));
}

async function saveFinalState() {
    const finalLog = {
        participant_id: appState.participantID,
        event_type: 'study_completed',
        data: {
            stage_1_feedback: appState.stageData[1].feedback,
            stage_1_classifications: appState.stageData[1].classifications,
            stage_2_feedback: appState.stageData[2].feedback,
            stage_2_suggestions: appState.stageData[2].suggestions,
            stage_3_feedback: appState.stageData[3].feedback,
            stage_3_chat_history: appState.stageData[3].chatHistory,
            all_logs: appState.allLogs
        },
        timestamp: new Date().toISOString()
    };

    fetch(`${API_BASE}/log_revision`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(finalLog)
    }).catch(err => console.error('Error saving final state:', err));
}

/* ============================================================================
   INITIALIZATION
   ============================================================================ */

window.addEventListener('DOMContentLoaded', async () => {
    // Generate participant ID
    initializeParticipant();

    // Load context data
    await loadContextData();

    // Show welcome page
    showPage('page-welcome');

    // Handle Enter key in textareas
    document.getElementById('stage1-feedback')?.addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key === 'Enter') {
            stage1Analyze();
        }
    });

    document.getElementById('stage2-feedback')?.addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key === 'Enter') {
            stage2Analyze();
        }
    });

    document.getElementById('stage3-chat-input')?.addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key === 'Enter') {
            stage3SendMessage();
        }
    });
});