# Meeting Conversation Analyzer

An AI-powered system that reads meeting transcripts and automatically generates structured insights.

## Dataset
- **Source**: [Kaggle - Meeting Conversation Dataset](https://www.kaggle.com/datasets/bumbiedumb/meeting-conversation-dataset)
- **File**: `Consolidated_meeting_transcript.txt` (197,739 characters, 925 lines)
- **Structure**: 12 meetings separated by `[Meeting Title]` headers
- **Speaker format**: `Name (Role): text` or `Name: text`

## Project Structure
```
meeting-analyzer/
├── data/raw/          ← Kaggle dataset
├── src/
│   ├── preprocessor.py      ← Text cleaning + speaker extraction
│   ├── summarizer.py        ← TextRank + TF-IDF summarization
│   ├── topic_extractor.py   ← TF-IDF keywords + LDA topics
│   ├── action_detector.py   ← Rule-based action item detection
│   ├── embeddings.py        ← Sentence-BERT embeddings
│   └── speaker_analyzer.py ← Speaker importance weighting
├── api/main.py        ← FastAPI endpoints
├── visualizations/    ← Chart generation
└── sample_outputs/    ← Generated charts + sample JSON
```

## Approach

### 1. Preprocessing
- Split transcript into individual meetings using `[Title]` headers
- Extract speaker tags using regex patterns
- Clean text: lowercase, remove brackets, fix whitespace
- Separate cleaning pipelines for NLP vs keyword extraction

### 2. NLP Features Implemented

| Feature | Method | Why |
|---------|--------|-----|
| Extractive Summarization | TextRank (graph-based PageRank) | Finds most central sentences |
| Keyword Extraction | TF-IDF with bigrams | Finds meeting-specific terms |
| Topic Modeling | LDA (Latent Dirichlet Allocation) | Finds thematic clusters |
| Action Item Detection | Rule-based regex patterns | Fast, transparent, no training needed |
| Sentence Importance | Sentence-BERT embeddings | Semantic understanding beyond keywords |
| Speaker Analysis | Word count + question detection | Measures participation and influence |

### 3. API Design
- FastAPI with auto-generated docs at `/docs`
- POST `/analyze-meeting` — main analysis endpoint
- GET `/meetings` — list all dataset meetings
- GET `/meetings/{title}` — analyze specific meeting

## Installation
```bash
git clone <your-repo-url>
cd meeting-analyzer


python -m venv venv
venv\Scripts\activate  


pip install -r requirements.txt


python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## Running the API
```bash
uvicorn api.main:app --reload
```

Visit `http://127.0.0.1:8000/docs` for interactive API documentation.

## API Usage

### Analyze a meeting
```bash
curl -X POST "http://127.0.0.1:8000/analyze-meeting" \
  -H "Content-Type: application/json" \
  -d '{"text": "your meeting transcript here..."}'
```

### Expected Output
```json
{
  "summary": "...",
  "key_topics": ["topic1", "topic2"],
  "important_sentences": ["...", "..."],
  "action_items": ["...", "..."],
  "speaker_analysis": [...],
  "metadata": {...}
}
```

## Visualizations

Four charts are generated automatically:
1. **Word Frequency** — most common meaningful words
2. **Sentence Importance** — SBERT semantic scores
3. **Topic Distribution** — LDA topic pie chart
4. **Speaker Participation** — word count + importance per speaker

## Preprocessing Decisions

- **Kept punctuation** during sentence tokenization (needed for boundary detection)
- **Removed punctuation** only for keyword extraction
- **Filtered short sentences** (< 5 words) as they add no summarization value
- **Custom stopwords** added for meeting-specific noise (speaker names, filler words)
- **Minimum score threshold** of 1 for action item detection to reduce false positives

## Limitations

- Speaker name extraction may fail on unusual name formats
- LDA topic quality depends on meeting length (short meetings = weaker topics)
- Action item detection is rule-based — may miss implicit commitments
- SBERT model requires internet connection on first run for download
- Abstractive summarization not implemented (would require GPT/T5)

## Sample Outputs

See `sample_outputs/` folder for:
- `word_frequency.png`
- `sentence_importance.png`
- `topic_distribution.png`
- `speaker_participation.png`
- `sample_response.json`