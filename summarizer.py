
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessor import clean_text, get_sentences


def build_similarity_matrix(sentences: list) -> np.ndarray:
    
    if len(sentences) < 2:
        return np.ones((len(sentences), len(sentences)))

    try:
       
        
        vectorizer = TfidfVectorizer(stop_words='english', max_features=200)
        tfidf_matrix = vectorizer.fit_transform(sentences)

        
        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        return similarity_matrix

    except Exception as e:
        print(f"Similarity matrix error: {e}")
        
        return np.eye(len(sentences))


def textrank_summarize(text: str, num_sentences: int = 5) -> list:
   
    
    sentences = get_sentences(text)

    if len(sentences) == 0:
        return ["No content to summarize."]

    if len(sentences) <= num_sentences:
        
        return sentences

    
    sim_matrix = build_similarity_matrix(sentences)

   
    graph = nx.from_numpy_array(sim_matrix)

   
    try:
        scores = nx.pagerank(graph, alpha=0.85, max_iter=200)
    except nx.PowerIterationFailedConvergence:
        
        print("PageRank failed to converge, using TF-IDF fallback")
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf = vectorizer.fit_transform(sentences)
        scores = {i: float(tfidf[i].sum()) for i in range(len(sentences))}

    
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)


    top_indices = sorted([idx for idx, score in ranked[:num_sentences]])

    return [sentences[i] for i in top_indices]


def tfidf_summarize(text: str, num_sentences: int = 5) -> list:

    sentences = get_sentences(text)

    if len(sentences) == 0:
        return ["No content to summarize."]

    if len(sentences) <= num_sentences:
        return sentences

    
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)


    sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()

    
    top_indices = sentence_scores.argsort()[-num_sentences:][::-1]
    top_indices = sorted(top_indices)  

    return [sentences[i] for i in top_indices]


def generate_summary(text: str, num_sentences: int = 5, method: str = "textrank") -> dict:
   
    if method == "tfidf":
        summary_sentences = tfidf_summarize(text, num_sentences)
    else:
        summary_sentences = textrank_summarize(text, num_sentences)

    
    summary_text = ' '.join(summary_sentences)

    return {
        "summary": summary_text,
        "summary_sentences": summary_sentences,
        "method_used": method,
        "sentence_count": len(summary_sentences)
    }


if __name__ == "__main__":
    from src.preprocessor import load_transcript, split_into_meetings

    DATA_PATH = "data/raw/Consolidated_meeting_transcript.txt"
    raw = load_transcript(DATA_PATH)
    meetings = split_into_meetings(raw)

    # Test on first meeting
    first_title = list(meetings.keys())[0]
    first_text = list(meetings.values())[0]

    print(f"Testing on: {first_title}")
    print(f"Original length: {len(first_text.split())} words\n")

    
    result = generate_summary(first_text, num_sentences=5, method="textrank")
    print("=== TEXTRANK SUMMARY ===")
    print(result['summary'])

    print("\n=== TFIDF SUMMARY ===")
    result2 = generate_summary(first_text, num_sentences=5, method="tfidf")
    print(result2['summary'])