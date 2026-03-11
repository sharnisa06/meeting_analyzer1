
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessor import get_sentences


print("Loading SBERT model...")
MODEL = SentenceTransformer('all-MiniLM-L6-v2')
print("SBERT model loaded!")


def encode_sentences(sentences: list) -> np.ndarray:
   
    if not sentences:
        return np.array([])

    embeddings = MODEL.encode(sentences, show_progress_bar=False)
    return embeddings


def rank_sentences_by_importance(text: str, top_n: int = 5) -> list:
    
    sentences = get_sentences(text)

    if len(sentences) == 0:
        return []

    if len(sentences) == 1:
        return [(sentences[0], 1.0)]

    
    embeddings = encode_sentences(sentences)

    centroid = np.mean(embeddings, axis=0, keepdims=True)

    
    scores = cosine_similarity(embeddings, centroid).flatten()

    sentence_scores = list(zip(sentences, scores))

    sentence_scores.sort(key=lambda x: x[1], reverse=True)

    return sentence_scores[:top_n]


def get_important_sentences(text: str, top_n: int = 5) -> list:

    ranked = rank_sentences_by_importance(text, top_n=top_n)
    return [sentence for sentence, score in ranked]


def compute_semantic_similarity(text1: str, text2: str) -> float:
    
    emb1 = MODEL.encode([text1])
    emb2 = MODEL.encode([text2])
    similarity = cosine_similarity(emb1, emb2)[0][0]
    return float(similarity)


if __name__ == "__main__":
    from src.preprocessor import load_transcript, split_into_meetings

    DATA_PATH = "data/raw/Consolidated_meeting_transcript.txt"
    raw = load_transcript(DATA_PATH)
    meetings = split_into_meetings(raw)

    first_title = list(meetings.keys())[0]
    first_text = list(meetings.values())[0]

    print(f"\nTesting on: {first_title}\n")

    print("=== TOP IMPORTANT SENTENCES (SBERT) ===")
    important = rank_sentences_by_importance(first_text, top_n=5)
    for i, (sentence, score) in enumerate(important, 1):
        print(f"\n{i}. [score: {score:.4f}]")
        print(f"   {sentence}")