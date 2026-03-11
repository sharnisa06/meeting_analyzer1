import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessor import clean_text, get_sentences, extract_utterances
from src.summarizer import generate_summary
from src.topic_extractor import get_key_topics
from src.action_detector import get_action_items

SAMPLE = """
John Smith: We need to fix the login bug by Friday.
Sarah Jones: I will handle that and send you a full update by Thursday.
Mark Davis: Great. We should also schedule a review meeting next week.
John Smith: Agreed. I will send the calendar invite today.
Sarah Jones: Can we also discuss the budget concerns that came up last week?
Mark Davis: Yes, we need to allocate more resources to the testing team.
"""

def test_preprocessor():
    print("Testing preprocessor...")
    sentences = get_sentences(SAMPLE)
    assert len(sentences) > 0
    utterances = extract_utterances(SAMPLE)
    assert len(utterances) > 0
    print(f"  Sentences: {len(sentences)} ✅")
    print(f"  Utterances: {len(utterances)} ✅")

def test_summarizer():
    print("Testing summarizer...")
    result = generate_summary(SAMPLE, num_sentences=2)
    assert len(result["summary"]) > 0
    print(f"  Summary: {result['summary'][:80]}... ✅")

def test_topic_extractor():
    print("Testing topic extractor...")
    topics = get_key_topics(SAMPLE, top_n=5)
    assert len(topics) > 0
    print(f"  Topics: {topics} ✅")

def test_action_detector():
    print("Testing action detector...")
    items = get_action_items(SAMPLE, top_n=5)
    assert len(items) > 0
    print(f"  Action items: {len(items)} found ✅")

if __name__ == "__main__":
    print("Running all tests...\n")
    test_preprocessor()
    test_summarizer()
    test_topic_extractor()
    test_action_detector()
    print("\nAll tests passed! ✅")