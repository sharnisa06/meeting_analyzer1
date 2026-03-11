import re
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessor import extract_utterances


def analyze_speakers(utterances: list) -> dict:
    speaker_stats = {}

    for utterance in utterances:
        speaker = utterance.get('speaker', 'Unknown')
        text = utterance.get('text', '')

        if speaker not in speaker_stats:
            speaker_stats[speaker] = {
                "utterance_count": 0,
                "word_count": 0,
                "questions_asked": 0,
            }

        stats = speaker_stats[speaker]
        stats["utterance_count"] += 1
        stats["word_count"] += len(text.split())
        if '?' in text:
            stats["questions_asked"] += 1

    total_words = sum(s["word_count"] for s in speaker_stats.values())

    for speaker, stats in speaker_stats.items():
        word_share = stats["word_count"] / total_words if total_words > 0 else 0
        question_score = min(stats["questions_asked"] / 10, 1.0)
        stats["importance_score"] = round(0.7 * word_share + 0.3 * question_score, 4)
        stats["word_share_percent"] = round(word_share * 100, 1)

    return dict(sorted(
        speaker_stats.items(),
        key=lambda x: x[1]["importance_score"],
        reverse=True
    ))


def get_speaker_summary(utterances: list) -> list:
    stats = analyze_speakers(utterances)
    summary = []
    for speaker, data in stats.items():
        summary.append({
            "speaker": speaker,
            "utterance_count": data["utterance_count"],
            "word_count": data["word_count"],
            "word_share_percent": data["word_share_percent"],
            "questions_asked": data["questions_asked"],
            "importance_score": data["importance_score"]
        })
    return summary


if __name__ == "__main__":
    from src.preprocessor import load_transcript, split_into_meetings

    DATA_PATH = "data/raw/Consolidated_meeting_transcript.txt"
    raw = load_transcript(DATA_PATH)
    meetings = split_into_meetings(raw)
    first_text = list(meetings.values())[0]
    utterances = extract_utterances(first_text)

    print("=== SPEAKER ANALYSIS ===")
    for s in get_speaker_summary(utterances):
        print(f"\n{s['speaker']}")
        print(f"  Words: {s['word_count']} ({s['word_share_percent']}%)")
        print(f"  Importance: {s['importance_score']}")