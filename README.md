# Pedagogical Feedback Assistant 

AI-assisted feedback coaching for TAs. This web-application guides users through three stages of feedback refinement, with AI-powered classification and suggestions.
Three stages:
Stage 1: Classifier
Stage 2: Micro + Coaching Guided LLM generated suggestions
Stage 3: Chat

User written feedback persists from one stage to the next.

## Architecture Overview

```
┌─────────────────┐
│   Frontend      │  (HTML/CSS/JS)
│  - Multi-page   │  - Stage 1: Classification
│  - Dark theme   │  - Stage 2: LLM Suggestions
│  - Responsive   │  - Stage 3: Chat Refinement
└────────┬────────┘
         │ API calls
         ▼
┌─────────────────┐
│   Backend       │  (FastAPI)
│  - Classifiers  │  - Sentence classification
│  - LLM          │  - OpenAI GPT-4o LLM
│  - Logging      │  - Event logging
└─────────────────┘
```

## Key Features

###  Stage 1: Classification
- Teachers write feedback and see real-time classification
- Three rubric categories: Current State, Next Steps, Strategy
- Color-coded visual feedback (green, yellow, red)
- Iterative refinement with multiple analyze attempts

###  Stage 2: Refinement with Suggestions
- Dual coaching modes: Micro & Coaching styles
- LLM-powered suggestions for improving feedback
- Feedback persists across stages
- Multiple suggestion requests as feedback evolves

###  Stage 3: Chat Refinement
- Natural language conversation with AI assistant
- Full context awareness (classifications, previous feedback versions)
- Dynamic feedback modification with auto-updated context
- Chat history logging

###  Participant Tracking
- Unique participant IDs: `PID-YYYYMMDD-5DIGITS`
- Persistent sidebar with progress indicator
- Event logging (JSONL format)

## Setup Instructions

### Prerequisites
- Python 3.9+
- Node.js (optional, for development)
- OpenAI API key
- PyTorch (for classifiers)
- HuggingFace transformers library

### Installation

1. **Clone/Extract the project**
```bash
cd metafeedback-assistant
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install fastapi uvicorn pydantic openai torch transformers nltk
```
Or more simply install the requirements.txt file into your virtual environment

4. **Configure environment**
Configure your API key in your environment

### Running the Application

# Terminal: Backend API
python -m uvicorn app.main:app

#  Open browser
Application can be found at
# http://localhost:8000/static/index.html
```

## File Structure

```
metafeedback-assistant
|   README.md
|   requirements.txt
|
+---app
|   |   main.py
|   |
|   +---classifiers
|   |   |   loader.py
|   |   |   predict.py
|   |
|   +---llm
|   |   |   client.py
|   |   |   prompts.py
|   |
|   +---routers
|   |   |   router.py
|   |
|   +---schemas
|   |   |   feedback.py
|   |
|   +---static
|   |   |   app.js
|   |   |   context.json
|   |   |   index.html
|   |   |   styles.css
|   |
    +---utils
        |   logger.py
        |   text.py


```

## Configuration

### Editing Problem Statement & Solution
Edit `app/static/context.json`:
```json
{
  "problem": "Your problem statement here",
  "solution": "The student solution here"
}
```

### Customizing Prompts
Edit `app/llm/prompts.py` to modify:
- System prompts for different coaching styles
- Rubric definitions
- LLM instruction details

### Classifier Thresholds
Adjust in `app/classifiers/predict.py`:
```python
threshold = 0.6  # Confidence threshold for sentence classification
```

## Data & Logging

All interactions are logged in `activity_log.jsonl` with:
- Event type (initialization, analysis, chat, completion)
- Participant ID
- Full context (feedback, classifications, suggestions)
- Timestamps (client & server)

Example log entry:
```json
{
  "participant_id": "PID-20250210-54321",
  "event_type": "stage_1_analyze",
  "data": {
    "feedback": "Your feedback text...",
    "classifications": [...],
    "timestamp": "2025-02-10T14:30:00Z"
  },
  "server_timestamp": "2025-02-10T14:30:01Z"
}
```

## Customization Guide

### Modifying Rubric Categories
1. Edit `RUBRIC_DEFINITIONS` in `app/llm/prompts.py`
2. Update color variables in `styles.css`
3. Modify classifier output handling in `app/classifiers/predict.py`

### Changing LLM Provider
Replace OpenAI client in `app/llm/client.py`:
```python
# Instead of OpenAI, use Anthropic, Gemini, etc.
from anthropic import Anthropic
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
```

## API Endpoints

### POST `/api/analyze`
Classify feedback and get suggestions.
```json
{
  "statement": "Problem text",
  "student_solution": "Student's work",
  "feedback": "Teacher's feedback"
}
```

### POST `/api/chat`
Continue chat for feedback refinement.
```json
{
  "style": "coaching",
  "history": [{"role": "user", "content": "..."}],
  "context": {...}
}
```

### POST `/api/log_revision`
Log any event to the activity log.
```json
{
  "participant_id": "PID-...",
  "event_type": "stage_1_analyze",
  "data": {...},
  "timestamp": "2025-02-10T14:30:00Z"
}
```

## Future Enhancements

Backend Improvements

- [ ] Enhance prompts for less vague and verbose answers. Put more rules in place
- [ ] Micro and Coaching should both produce answers that are close in character number. 
- [ ] Better context managements for the chat stage. Make sure the whole conetx from previous stages persists.
- [ ] Implement post-test stage to evaluate the improvement in feedback quality (e.g. increase in rubric coverage)
- [ ] Sentiment analysis (is the feedback too pesimistic? - this is also a big challenge for TAs when giving grading feedback)
- [ ] Connect system to the ML4ED Laboratory server 
- [ ] (Optional) Base LLM suggestions in pedagogical literature

Frontend Improvements
- [] Circular progress indicator while the system is waiting for the answers form the LLM/Classifiers
- [] Change Stage 1 highlight colores for the three rubrics - red underline is misleading, change with other color
- [] Stage 2, Micro and Coaching suggestions should be shown side by side to avoid any primacy effect
- [] Stage 3, the chat should start with a suggestive message from the LLM of what types  of questions can be asked by the user

System Design Improvements

- [] Modular Format - Classifier + Suggestions or Classifier + Chat 

---
