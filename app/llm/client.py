from openai import OpenAI
import os
from app.llm.prompts import SYSTEM_PROMPTS

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
PROMPT_STYLE = "socratic" # "micro", "coaching"


def get_llm_suggestions(problem, solution, classified_sentences):
    """
    We show the LLM what's ALREADY there, 
    so it only suggests what's GENUINELY missing.
    """
    
    # Count what's already in the feedback
    coverage = {
        "current_state": sum(1 for s in classified_sentences if s.get("current_state")),
        "next_steps": sum(1 for s in classified_sentences if s.get("next_steps")),
        "strategy": sum(1 for s in classified_sentences if s.get("strategy"))
    }
    
    # Extract actual content from each category
    coverage_details = {
        "current_state": [s["sentence"] for s in classified_sentences if s.get("current_state")],
        "next_steps": [s["sentence"] for s in classified_sentences if s.get("next_steps")],
        "strategy": [s["sentence"] for s in classified_sentences if s.get("strategy")]
    }

    # Explicit "what's already there" section
    coverage_text = f"""
    WHAT THE FEEDBACK ALREADY COVERS:

    Current State ({coverage['current_state']} sentences):
    {chr(10).join('- ' + s for s in coverage_details['current_state']) if coverage_details['current_state'] else '(None)'}

    Next Steps ({coverage['next_steps']} sentences):
    {chr(10).join('- ' + s for s in coverage_details['next_steps']) if coverage_details['next_steps'] else '(None)'}

    Strategy ({coverage['strategy']} sentences):
    {chr(10).join('- ' + s for s in coverage_details['strategy']) if coverage_details['strategy'] else '(None)'}
    """

    user_prompt = f"""
    You are evaluating the quality of a TEACHER'S feedback on a student's solution.

    CONTEXT (for reference only):
    Problem: {problem}
    Student Solution: {solution}

    === WHAT'S ALREADY COVERED (by category) ===
    {coverage_text}

    YOUR TASK:
    Analyze the feedback against the rubric. Identify ONLY what is genuinely missing.

    CRITICAL RULES:
    1. Do NOT suggest anything the teacher has already said (check the "WHAT'S ALREADY COVERED" section)
    2. Do NOT rewrite the feedback
    3. Do NOT solve the problem
    4. Only suggest gaps that matter for student learning
    5. If feedback is already strong, say so explicitly

    Based on the rubric:
    - CURRENT STATE: Does the teacher identify what the student did right/wrong?
    - NEXT STEPS: Does the teacher give concrete actions to improve?
    - STRATEGY: Does the teacher help the student approach similar problems in the future?

    What is GENUINELY missing that would improve student learning?
    """

    # TWO SEPARATE CALLS FOR THE TWO STYLES
    micro_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPTS["micro"]},
            {"role": "user", "content": user_prompt}
        ],
         max_tokens=250
    )

    coaching_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPTS["coaching"]},
            {"role": "user", "content": user_prompt}
        ],
         max_tokens=250
    )

    return {
        "micro": micro_response.choices[0].message.content,
        "coaching": coaching_response.choices[0].message.content
    }