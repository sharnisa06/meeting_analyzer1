

import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import Counter
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessor import clean_text_for_keywords, get_sentences
from src.embeddings import rank_sentences_by_importance
from src.topic_extractor import extract_lda_topics
from src.speaker_analyzer import get_speaker_summary

# Output directory for saved charts
CHARTS_DIR = "sample_outputs"
os.makedirs(CHARTS_DIR, exist_ok=True)


def plot_word_frequency(text: str, top_n: int = 15, save: bool = True) -> str:
    
    
    clean = clean_text_for_keywords(text)
    words = clean.split()
    word_counts = Counter(words).most_common(top_n)

    if not word_counts:
        print("No words to plot")
        return ""

    words_list = [w[0] for w in word_counts]
    counts = [w[1] for w in word_counts]

    
    fig, ax = plt.subplots(figsize=(12, 6))

    
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(words_list)))
    bars = ax.barh(words_list[::-1], counts[::-1], color=colors[::-1])

    
    for bar, count in zip(bars, counts[::-1]):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                str(count), va='center', fontsize=9)

    ax.set_xlabel('Frequency', fontsize=12)
    ax.set_title(f'Top {top_n} Most Frequent Words in Meeting', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    if save:
        path = os.path.join(CHARTS_DIR, "word_frequency.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
        plt.close()
        return path

    plt.show()
    return ""


def plot_sentence_importance(text: str, top_n: int = 8, save: bool = True) -> str:
    
    ranked = rank_sentences_by_importance(text, top_n=top_n)

    if not ranked:
        print("No sentences to plot")
        return ""

    sentences, scores = zip(*ranked)

    
    short_labels = [s[:60] + "..." if len(s) > 60 else s for s in sentences]

    fig, ax = plt.subplots(figsize=(13, 7))

    # Color by score
    colors = plt.cm.RdYlGn(np.array(scores))
    bars = ax.barh(range(len(short_labels)), scores, color=colors)

    # Labels
    ax.set_yticks(range(len(short_labels)))
    ax.set_yticklabels(short_labels, fontsize=8)
    ax.set_xlabel('Semantic Importance Score (SBERT)', fontsize=11)
    ax.set_title('Sentence Importance Ranking', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.0)

    # Score annotations
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{score:.3f}', va='center', fontsize=8)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    if save:
        path = os.path.join(CHARTS_DIR, "sentence_importance.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
        plt.close()
        return path

    plt.show()
    return ""


def plot_topic_distribution(text: str, num_topics: int = 3, save: bool = True) -> str:
    
    topics = extract_lda_topics(text, num_topics=num_topics)

    if not topics:
        print("No topics to plot")
        return ""

    labels = [t['label'] for t in topics]
   
    sizes = [1/len(topics)] * len(topics)

    colors = plt.cm.Set3(np.linspace(0, 1, len(topics)))

    fig, ax = plt.subplots(figsize=(10, 7))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=None,
        autopct='%1.0f%%',
        colors=colors,
        startangle=90,
        pctdistance=0.75
    )

    # Legend with topic words
    ax.legend(
        wedges,
        labels,
        title="Topics",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
        fontsize=9
    )

    ax.set_title('Meeting Topic Distribution (LDA)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save:
        path = os.path.join(CHARTS_DIR, "topic_distribution.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
        plt.close()
        return path

    plt.show()
    return ""


def plot_speaker_participation(utterances: list, save: bool = True) -> str:
    
    speaker_data = get_speaker_summary(utterances)

    if not speaker_data:
        print("No speaker data to plot")
        return ""

    speakers = [s['speaker'] for s in speaker_data]
    word_counts = [s['word_count'] for s in speaker_data]
    importance = [s['importance_score'] for s in speaker_data]

    x = np.arange(len(speakers))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Bar chart for word count
    color1 = '#4C72B0'
    bars = ax1.bar(x - width/2, word_counts, width, label='Word Count', color=color1, alpha=0.8)
    ax1.set_xlabel('Speaker', fontsize=12)
    ax1.set_ylabel('Word Count', color=color1, fontsize=11)
    ax1.tick_params(axis='y', labelcolor=color1)

    # Second y-axis for importance score
    ax2 = ax1.twinx()
    color2 = '#DD8452'
    ax2.bar(x + width/2, importance, width, label='Importance Score', color=color2, alpha=0.8)
    ax2.set_ylabel('Importance Score', color=color2, fontsize=11)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, 1.0)

    # X labels
    ax1.set_xticks(x)
    ax1.set_xticklabels(speakers, rotation=30, ha='right', fontsize=9)
    ax1.set_title('Speaker Participation Analysis', fontsize=14, fontweight='bold')

    # Legend
    patch1 = mpatches.Patch(color=color1, alpha=0.8, label='Word Count')
    patch2 = mpatches.Patch(color=color2, alpha=0.8, label='Importance Score')
    ax1.legend(handles=[patch1, patch2], loc='upper right')

    ax1.spines['top'].set_visible(False)
    plt.tight_layout()

    if save:
        path = os.path.join(CHARTS_DIR, "speaker_participation.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
        plt.close()
        return path

    plt.show()
    return ""


def generate_all_charts(text: str, utterances: list) -> dict:
    
    print("Generating all charts...")
    paths = {}
    paths['word_frequency'] = plot_word_frequency(text)
    paths['sentence_importance'] = plot_sentence_importance(text)
    paths['topic_distribution'] = plot_topic_distribution(text)
    paths['speaker_participation'] = plot_speaker_participation(utterances)
    print("All charts generated!")
    return paths


if __name__ == "__main__":
    from src.preprocessor import (
        load_transcript, split_into_meetings, extract_utterances
    )

    DATA_PATH = "data/raw/Consolidated_meeting_transcript.txt"
    raw = load_transcript(DATA_PATH)
    meetings = split_into_meetings(raw)

    first_title = list(meetings.keys())[0]
    first_text = list(meetings.values())[0]
    utterances = extract_utterances(first_text)

    print(f"Generating charts for: {first_title}\n")
    paths = generate_all_charts(first_text, utterances)

    print("\nCharts saved to:")
    for name, path in paths.items():
        print(f"  {name}: {path}")