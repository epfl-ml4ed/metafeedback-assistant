import re

def split_sentences(text):
    raw = re.split(r"[.!?]\s+", text.strip())
    return [s for s in raw if s]
