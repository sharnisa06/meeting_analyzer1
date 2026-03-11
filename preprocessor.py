

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize


nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)


STOP_WORDS = set(stopwords.words('english'))


def load_transcript(filepath: str) -> str:
    """
    Load raw text file from disk.
    WHY: Single entry point for file loading — easy to swap source later.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def split_into_meetings(raw_text: str) -> dict:
   
    parts = re.split(r'(\[.+?\])', raw_text)
    
    meetings = {}
    current_title = "Unknown"
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        
        if re.match(r'^\[.+?\]$', part):
            
            current_title = part.strip('[]').strip()
        else:
           
            if len(part) > 100:  
                meetings[current_title] = part
    
    return meetings


def extract_utterances(meeting_text: str) -> list:
    
    utterances = []
    
    
    lines = meeting_text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        
        match = re.match(r'^([A-Za-z\s]+)\(([^)]+)\):\s*(.+)$', line)
        if match:
            utterances.append({
                "speaker": match.group(1).strip(),
                "role": match.group(2).strip(),
                "text": match.group(3).strip()
            })
            continue
        
        
        match = re.match(r'^([A-Za-z](?:[A-Za-z\s]{2,40})):\s*(.+)$', line)
        if match:
            utterances.append({
                "speaker": match.group(1).strip(),
                "role": None,
                "text": match.group(2).strip()
            })
            continue
        
        if utterances:
            utterances[-1]["text"] += " " + line
    
    return utterances


def clean_text(text: str) -> str:
   
    # Step 1: Lowercase everything
    text = text.lower()
    
    # Step 2: Remove stage directions like [laughter], [pause]
    text = re.sub(r'\[.*?\]', '', text)
    
    # Step 3: Remove special characters but KEEP periods, commas, apostrophes
    text = re.sub(r'[^\w\s\.\,\?\!\'\-]', '', text)
    
    # Step 4: Fix multiple spaces/newlines
    text = re.sub(r'\s+', ' ', text)
    
    # Step 5: Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def clean_text_for_keywords(text: str) -> str:
    
    text = clean_text(text)
    
    
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    
    words = word_tokenize(text)
    
    
    words = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    
    return ' '.join(words)


def get_sentences(text: str) -> list:
   
    clean = clean_text(text)
    sentences = sent_tokenize(clean)
    
   
    
    sentences = [s for s in sentences if len(s.split()) >= 5]
    
    return sentences


def preprocess_meeting(meeting_text: str) -> dict:
    
    utterances = extract_utterances(meeting_text)
    
    # Combine all text (ignoring speaker tags) for NLP
    full_text = ' '.join([u['text'] for u in utterances])
    
    
    speaker_count = {}
    for u in utterances:
        spk = u['speaker']
        speaker_count[spk] = speaker_count.get(spk, 0) + 1
    
    return {
        "utterances": utterances,
        "clean_text": clean_text(full_text),
        "keyword_text": clean_text_for_keywords(full_text),
        "sentences": get_sentences(full_text),
        "speaker_count": speaker_count,
        "raw_text": meeting_text
    }


def preprocess_all_meetings(filepath: str) -> dict:
    
    raw_text = load_transcript(filepath)
    meetings = split_into_meetings(raw_text)
    
    result = {}
    for title, text in meetings.items():
        print(f"Processing: {title}")
        result[title] = preprocess_meeting(text)
    
    return result


if __name__ == "__main__":
    DATA_PATH = "data/raw/Consolidated_meeting_transcript.txt"
    
    all_meetings = preprocess_all_meetings(DATA_PATH)
    
    print(f"\n✅ Total meetings processed: {len(all_meetings)}")
    
    
    first = list(all_meetings.values())[0]
    
    print(f"\n📊 Utterances found: {len(first['utterances'])}")
    print(f"📝 Sentences found: {len(first['sentences'])}")
    print(f"👥 Speakers: {first['speaker_count']}")
    print(f"\n🔤 Clean text preview:\n{first['clean_text'][:300]}")
    print(f"\n📌 First 3 sentences:")
    for s in first['sentences'][:3]:
        print(f"  - {s}")