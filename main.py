from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessor import preprocess_meeting, load_transcript, split_into_meetings
from src.summarizer import generate_summary
from src.topic_extractor import get_key_topics
from src.action_detector import get_action_items
from src.embeddings import get_important_sentences
from src.speaker_analyzer import get_speaker_summary

app = FastAPI(
    title="Meeting Conversation Analyzer",
    description="AI-powered meeting analysis system.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

DATA_PATH = "data/raw/Consolidated_meeting_transcript.txt"

class MeetingRequest(BaseModel):
    text: str
    num_summary_sentences: Optional[int] = 5
    num_topics: Optional[int] = 8
    num_action_items: Optional[int] = 10
    num_important_sentences: Optional[int] = 5
    summary_method: Optional[str] = "textrank"

class MeetingResponse(BaseModel):
    summary: str
    key_topics: list
    important_sentences: list
    action_items: list
    speaker_analysis: list
    metadata: dict

@app.get("/")
def root():
    return {
        "message": "Meeting Analyzer API is running!",
        "docs": "Visit /docs for interactive API documentation"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "version": "1.0.0"}

@app.post("/analyze-meeting", response_model=MeetingResponse)
def analyze_meeting(request: MeetingRequest):
    start_time = time.time()

    if not request.text or len(request.text.strip()) < 50:
        raise HTTPException(
            status_code=400,
            detail="Text too short. Please provide at least 50 characters."
        )

    try:
        preprocessed = preprocess_meeting(request.text)
        summary_result = generate_summary(
            request.text,
            num_sentences=request.num_summary_sentences,
            method=request.summary_method
        )
        topics = get_key_topics(request.text, top_n=request.num_topics)
        action_items = get_action_items(request.text, top_n=request.num_action_items)
        important_sentences = get_important_sentences(
            request.text,
            top_n=request.num_important_sentences
        )
        speaker_analysis = get_speaker_summary(preprocessed["utterances"])
        processing_time = round(time.time() - start_time, 2)

        return MeetingResponse(
            summary=summary_result["summary"],
            key_topics=topics,
            important_sentences=important_sentences,
            action_items=action_items,
            speaker_analysis=speaker_analysis,
            metadata={
                "processing_time_seconds": processing_time,
                "total_sentences": len(preprocessed["sentences"]),
                "total_utterances": len(preprocessed["utterances"]),
                "total_speakers": len(preprocessed["speaker_count"]),
                "summary_method": request.summary_method,
                "word_count": len(request.text.split())
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/meetings")
def list_meetings():
    try:
        raw = load_transcript(DATA_PATH)
        meetings = split_into_meetings(raw)
        return {
            "total_meetings": len(meetings),
            "meetings": [
                {
                    "title": title,
                    "word_count": len(text.split()),
                    "character_count": len(text)
                }
                for title, text in meetings.items()
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/meetings/{meeting_title}")
def analyze_specific_meeting(meeting_title: str, num_sentences: int = 5):
    try:
        raw = load_transcript(DATA_PATH)
        meetings = split_into_meetings(raw)

        matched_title = None
        for title in meetings.keys():
            if title.lower() == meeting_title.lower():
                matched_title = title
                break

        if not matched_title:
            raise HTTPException(
                status_code=404,
                detail=f"Meeting '{meeting_title}' not found. Available: {list(meetings.keys())}"
            )

        request = MeetingRequest(
            text=meetings[matched_title],
            num_summary_sentences=num_sentences
        )
        return analyze_meeting(request)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))