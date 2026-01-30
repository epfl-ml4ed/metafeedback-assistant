import torch
import torch.nn.functional as F

# Import the three loaded model bundles
# They must each have: model, tokenizer
from app.classifiers.loader import (
    current_state,
    next_steps,
    strategy
)

# # Predict a single sentence for one rubric
# def predict_sentence(model_bundle, text, threshold=0.6):
#     """
#     Returns:
#         0 → negative (or low confidence → treat as NONE)
#         1 → positive (confident)
#     """
#     inputs = model_bundle.tokenizer(
#         text,
#         return_tensors="pt",
#         truncation=True
#     )

#     with torch.no_grad():
#         logits = model_bundle.model(**inputs).logits

#     # Convert logits to probabilities
#     probs = F.softmax(logits, dim=1)

#     predicted_class = torch.argmax(probs, dim=1).item()
#     confidence = probs[0][predicted_class].item()

#     # If not confident enough it is 0
#     if confidence < threshold:
#         return 0

#     # If confident we return raw prediction
#     return predicted_class


# # Predict all three labels for a sentence
# def classify_feedback_sentence(sentence):
#     """
#     Applies all three binary classifiers (current state,
#     next steps, strategy) with thresholding + ambiguity rule.

#     Ambiguity rule:
#         If more than one classifier fires (1), set all to NONE (0).
#     """

#     cs = predict_sentence(current_state, sentence)
#     ns = predict_sentence(next_steps, sentence)
#     st = predict_sentence(strategy, sentence)

#     # If two or more classifiers fire we treat as NONE
#     if cs + ns + st > 1:
#         cs, ns, st = 0, 0, 0

#     return {
#         "sentence": sentence,
#         "current_state": cs,
#         "next_steps": ns,
#         "strategy": st
#     }

# import torch
# import torch.nn.functional as F

# # Import the three loaded model bundles
# # They must each have: model, tokenizer
# from app.classifiers.loader import (
#     current_state,
#     next_steps,
#     strategy
# )

# Predict a single sentence for one rubric
def predict_sentence(model_bundle, text, threshold=0.6):
    """
    Returns:
        (label, confidence)
        label: 0 or 1
        confidence: probability of predicted class
    """
    inputs = model_bundle.tokenizer(
        text,
        return_tensors="pt",
        truncation=True
    )

    with torch.no_grad():
        logits = model_bundle.model(**inputs).logits

    probs = F.softmax(logits, dim=1)

    predicted_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0][predicted_class].item()

    if confidence < threshold:
        return 0, confidence

    return predicted_class, confidence



def classify_feedback_sentence(sentence):
    cs, cs_conf = predict_sentence(current_state, sentence)
    ns, ns_conf = predict_sentence(next_steps, sentence)
    st, st_conf = predict_sentence(strategy, sentence)

    # collect only fired classifiers
    fired = [
        ("current_state", cs, cs_conf),
        ("next_steps", ns, ns_conf),
        ("strategy", st, st_conf),
    ]
    fired = [f for f in fired if f[1] == 1]

    # if more than one fired, keep only the highest confidence
    if len(fired) > 1:
        winner = max(fired, key=lambda x: x[2])[0]
        cs = 1 if winner == "current_state" else 0
        ns = 1 if winner == "next_steps" else 0
        st = 1 if winner == "strategy" else 0

    return {
        "sentence": sentence,
        "current_state": cs,
        "next_steps": ns,
        "strategy": st
    }
