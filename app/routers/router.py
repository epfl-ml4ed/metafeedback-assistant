from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from app.schemas.feedback import FeedbackRequest
from app.utils.text import split_sentences
from app.classifiers.predict import classify_feedback_sentence
from app.llm.client import get_llm_suggestions
from app.utils.logger import save_to_log
from app.llm.prompts import SYSTEM_PROMPTS
from app.llm.client import client
import json
from datetime import datetime


router = APIRouter(prefix="/api")


class LogEntry(BaseModel):
    participant_id: str
    event_type: str
    data: Dict[str, Any]
    timestamp: str


class ChatRequest(BaseModel):
    style: str
    history: List[Dict[str, str]]
    context: Optional[Dict[str, Any]] = None


@router.post("/analyze")
def analyze(req: FeedbackRequest):
    sentences = [s for s in split_sentences(req.feedback) if s.strip()]
    classified = [classify_feedback_sentence(s) for s in sentences]

    try:
        suggestions = get_llm_suggestions(
            req.statement,
            req.student_solution,
            classified
        )
    except Exception as e:
        # still return classifications so UI can underline
        suggestions = {"micro": "", "coaching": ""}
        return {
            "classified_sentences": classified,
            "llm_suggestions": suggestions,
            "llm_error": str(e),
        }

    return {
        "classified_sentences": classified,
        "llm_suggestions": suggestions
    }

@router.post("/chat")
def chat_interaction(data: ChatRequest):
    """
    Handles multi-turn chat for feedback refinement.
    
    STRATEGY:
    - First message: Include full context + question
    - Subsequent messages: Just conversation history (more efficient)
    
    This avoids repeating context on every turn while keeping it available
    through the conversation history.
    """
    try:
        style = data.style
        history = data.history
        context = data.context or {}

        # Get base system prompt for the style
        base_prompt = SYSTEM_PROMPTS.get(style, SYSTEM_PROMPTS["coaching"])

        # Check if this is the first message (history will have exactly 1 user message)
        is_first_message = len(history) == 1
        
        if is_first_message:
            # First message: Include full context in system prompt
            system_prompt = _build_context_aware_system_prompt(style, context)
        else:
            # Subsequent messages: Just use base prompt
            # LLM has conversation history to reference
            system_prompt = base_prompt

        # Build message list: system prompt + conversation history
        messages = [
            {"role": "system", "content": system_prompt}
        ]

        # Add conversation history AS IS
        messages.extend(history)

        # Get response from Claude
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7
        )

        reply = response.choices[0].message.content

        return {"reply": reply}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _build_context_aware_system_prompt(style: str, context: Dict[str, Any]) -> str:
    """
    Builds a system prompt that includes context inline.
    This keeps the system prompt focused while incorporating necessary context.
    Context is NOT sent as separate messages—it's baked into the system prompt.
    
    Only called on the first chat message to avoid repeating context unnecessarily.
    """
    
    # Get base system prompt for the style
    base_prompt = SYSTEM_PROMPTS.get(style, SYSTEM_PROMPTS["coaching"])
    
    # Extract context
    problem = context.get("problem", "")
    solution = context.get("solution", "")
    feedback = context.get("feedback", "")
    classified = context.get("classified_sentences", [])
    suggestions = context.get("suggestions", {})

    # Format classified sentences
    classification_lines = []
    for item in classified:
        if item.get("current_state"):
            label = "current_state"
        elif item.get("next_steps"):
            label = "next_steps"
        elif item.get("strategy"):
            label = "strategy"
        else:
            label = "none"
        
        classification_lines.append(f"- \"{item['sentence']}\" → {label}")
    
    classification_text = "\n".join(classification_lines) if classification_lines else "(No classifications yet)"

    # Format previous suggestions
    micro_text = suggestions.get('micro', 'None') if suggestions.get('micro') else 'None'
    coaching_text = suggestions.get('coaching', 'None') if suggestions.get('coaching') else 'None'

    # Embed context into system prompt
    enhanced_prompt = f"""{base_prompt}

=== ACTIVE CONTEXT (Reference Only) ===

Problem:
{problem}

Student Solution:
{solution}

Teacher's Current Feedback:
{feedback}

Classified Sentences:
{classification_text}

Previous Suggestions (for reference):
- Micro: {micro_text}
- Coaching: {coaching_text}

=== END CONTEXT ===

IMPORTANT: 
- Respond ONLY to the teacher's current question.
- Do NOT repeat or summarize the context unless asked.
- Do NOT give suggestions the teacher has already incorporated.
- Answer their questions on what they didn't understand in the suggestion that your {style} previously gave them.
"""

    return enhanced_prompt

@router.post("/log_revision")
def log_revision(log_entry: LogEntry):
    """
    Comprehensive logging endpoint.
    Logs all events: initialization, analysis, chat messages, completions.
    Stores as JSONL for easy analysis.
    """
    try:
        # Convert pydantic model to dict
        log_dict = log_entry.dict()

        # Add server-side timestamp for verification
        log_dict["server_timestamp"] = datetime.utcnow().isoformat()

        # Append to JSONL file
        save_to_log("activity_log.jsonl", log_dict)

        return {"status": "logged", "timestamp": log_dict["server_timestamp"]}

    except Exception as e:
        print(f"Logging error: {e}")
        # Don't fail the user's request if logging fails
        return {"status": "error", "message": str(e)}


@router.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}