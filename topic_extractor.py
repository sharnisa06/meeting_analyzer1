

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessor import clean_text_for_keywords, get_sentences


def extract_tfidf_keywords(text: str, top_n: int = 10) -> list:
    
    sentences = get_sentences(text)

    if len(sentences) < 2:
        return []

    
    CUSTOM_STOPS = list(TfidfVectorizer(stop_words='english').get_stop_words()) + [
        'thats', 'think', 'know', 'just', 'like', 'really', 'going',
        'meeting', 'mark', 'jennifer', 'jack', 'tom', 'alice', 'john',
        'donald', 'peterson', 'robbins', 'miles', 'linnes', 'peters', 'ruting'
    ]
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=500,
        ngram_range=(1, 2),     # unigrams AND bigrams
        min_df=1,               # word must appear at least once
        max_df=0.95             # ignore words in more than 95% of sentences
    )

    tfidf_matrix = vectorizer.fit_transform(sentences)
    feature_names = vectorizer.get_feature_names_out()

   
    word_scores = np.array(tfidf_matrix.sum(axis=0)).flatten()

    
    word_score_pairs = list(zip(feature_names, word_scores))
    word_score_pairs.sort(key=lambda x: x[1], reverse=True)

    return word_score_pairs[:top_n]


def extract_lda_topics(text: str, num_topics: int = 3, words_per_topic: int = 5) -> list:
    
    sentences = get_sentences(text)

    if len(sentences) < 3:
        return []

    # CountVectorizer for LDA (LDA needs raw counts, not TF-IDF)
    vectorizer = CountVectorizer(
        stop_words='english',
        max_features=300,
        min_df=1
    )

    try:
        count_matrix = vectorizer.fit_transform(sentences)
        feature_names = vectorizer.get_feature_names_out()

        lda = LatentDirichletAllocation(
            n_components=num_topics,
            random_state=42,
            max_iter=10,
            learning_method='batch'
        )
        lda.fit(count_matrix)

        
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            
            top_word_indices = topic.argsort()[-words_per_topic:][::-1]
            top_words = [feature_names[i] for i in top_word_indices]
            topics.append({
                "topic_id": topic_idx + 1,
                "words": top_words,
                "label": f"Topic {topic_idx + 1}: {', '.join(top_words[:3])}"
            })

        return topics

    except Exception as e:
        print(f"LDA error: {e}")
        return []


def get_key_topics(text: str, top_n: int = 8) -> list:
    
    
    keywords = extract_tfidf_keywords(text, top_n=5)
    keyword_list = [word for word, score in keywords]

    
    lda_topics = extract_lda_topics(text, num_topics=3)
    
    
    lda_words = []
    for topic in lda_topics:
        lda_words.extend(topic['words'][:2])  # top 2 words per topic


    all_topics = keyword_list + lda_words
    seen = set()
    unique_topics = []
    for t in all_topics:
        if t not in seen and len(t) > 2:
            seen.add(t)
            unique_topics.append(t)

    return unique_topics[:top_n]



if __name__ == "__main__":
    from src.preprocessor import load_transcript, split_into_meetings

    DATA_PATH = "data/raw/Consolidated_meeting_transcript.txt"
    raw = load_transcript(DATA_PATH)
    meetings = split_into_meetings(raw)

    first_title = list(meetings.keys())[0]
    first_text = list(meetings.values())[0]

    print(f"Testing on: {first_title}\n")

    print("=== TF-IDF KEYWORDS ===")
    keywords = extract_tfidf_keywords(first_text, top_n=10)
    for word, score in keywords:
        print(f"  {word:30s} → {score:.4f}")

    print("\n=== LDA TOPICS ===")
    topics = extract_lda_topics(first_text, num_topics=3)
    for t in topics:
        print(f"  {t['label']}")

    print("\n=== FINAL KEY TOPICS (for API) ===")
    final = get_key_topics(first_text)
    print(final)