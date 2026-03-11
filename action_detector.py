

import re
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessor import get_sentences, extract_utterances




FUTURE_PATTERNS = [
    r'\b(i|we|they|he|she|\w+)\s+will\s+\w+',
    r'\b(i|we)\s+\'ll\s+\w+',
    r'\bgoing to\s+\w+',
    r'\bplan to\s+\w+',
    r'\bintend to\s+\w+',
]


NECESSITY_PATTERNS = [
    r'\b(we|i|team)\s+need\s+to\s+\w+',
    r'\bmust\s+\w+',
    r'\bshould\s+\w+',
    r'\bhave\s+to\s+\w+',
    r'\brequired\s+to\s+\w+',
    r'\bensure\s+\w+',
]


ACTION_VERB_PATTERNS = [
    r'^(send|schedule|review|update|create|prepare|contact|follow|check|submit|complete|finalize|implement|discuss|share|provide|arrange|coordinate)\b',
]


DEADLINE_PATTERNS = [
    r'\bby\s+(friday|monday|tuesday|wednesday|thursday|saturday|sunday)\b',
    r'\bby\s+(next\s+week|end\s+of\s+week|eod|eow|tomorrow)\b',
    r'\bby\s+\w+\s+\d+\b',
    r'\bbefore\s+the\s+\w+\b',
    r'\bdeadline\b',
    r'\bdue\s+(date|by|on)\b',
]


ASSIGNMENT_PATTERNS = [
    r'\bassigned\s+to\s+\w+',
    r'\baction\s+(for|item)\b',
    r'\bresponsible\s+for\b',
    r'\btake\s+ownership\b',
    r'\blead\s+on\b',
]


FOLLOWUP_PATTERNS = [
    r'\bfollow\s*up\b',
    r'\bget\s+back\s+to\b',
    r'\bcircle\s+back\b',
    r'\btouch\s+base\b',
    r'\bsend\s+(an?\s+)?(email|report|update|summary)\b',
    r'\bschedule\s+a\s+(meeting|call|review)\b',
]

ALL_PATTERNS = (
    FUTURE_PATTERNS +
    NECESSITY_PATTERNS +
    ACTION_VERB_PATTERNS +
    DEADLINE_PATTERNS +
    ASSIGNMENT_PATTERNS +
    FOLLOWUP_PATTERNS
)


def score_sentence_for_action(sentence: str) -> dict:
    
    sentence_lower = sentence.lower().strip()
    matched_patterns = []
    score = 0

    for pattern in ALL_PATTERNS:
        if re.search(pattern, sentence_lower):
            matched_patterns.append(pattern)
            score += 1

    
    for pattern in DEADLINE_PATTERNS:
        if re.search(pattern, sentence_lower):
            score += 1  

    return {
        "sentence": sentence,
        "score": score,
        "matched_patterns": len(matched_patterns),
        "is_action_item": score >= 1
    }


def extract_action_items(text: str, min_score: int = 1) -> list:
    
    sentences = get_sentences(text)
    action_items = []

    for sentence in sentences:
        result = score_sentence_for_action(sentence)
        if result['score'] >= min_score:
            action_items.append({
                "text": sentence,
                "confidence_score": result['score'],
                "pattern_matches": result['matched_patterns']
            })


    action_items.sort(key=lambda x: x['confidence_score'], reverse=True)

    return action_items


def extract_action_items_with_speakers(utterances: list, min_score: int = 1) -> list:
    
    action_items = []

    for utterance in utterances:
        speaker = utterance.get('speaker', 'Unknown')
        text = utterance.get('text', '')

        result = score_sentence_for_action(text)
        if result['score'] >= min_score:
            action_items.append({
                "speaker": speaker,
                "text": text,
                "confidence_score": result['score']
            })

    action_items.sort(key=lambda x: x['confidence_score'], reverse=True)
    return action_items


def get_action_items(text: str, top_n: int = 10) -> list:
    
    items = extract_action_items(text, min_score=1)

    
    return [item['text'] for item in items[:top_n]]



if __name__ == "__main__":
    from src.preprocessor import load_transcript, split_into_meetings

    DATA_PATH = "data/raw/Consolidated_meeting_transcript.txt"
    raw = load_transcript(DATA_PATH)
    meetings = split_into_meetings(raw)

    first_title = list(meetings.keys())[0]
    first_text = list(meetings.values())[0]

    print(f"Testing on: {first_title}\n")

    print("=== ACTION ITEMS ===")
    items = extract_action_items(first_text)
    for i, item in enumerate(items[:10], 1):
        print(f"\n{i}. [{item['confidence_score']} pts] {item['text']}")