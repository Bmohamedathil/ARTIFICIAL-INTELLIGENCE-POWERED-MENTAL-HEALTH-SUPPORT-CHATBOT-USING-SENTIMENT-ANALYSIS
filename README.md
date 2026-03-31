# 🧠 MINDFULAI – AI POWERED MENTAL HEALTH SUPPORT CHATBOT USING SENTIMENT ANALYSIS

MindfulAI is an intelligent AI-powered mental health support chatbot designed to provide real-time emotional assistance using Sentiment Analysis, Natural Language Processing (NLP), and voice-based interaction. The system detects user emotions and generates empathetic counselling responses to support mental wellness.


## 📌 About

MindfulAI is a computer vision–independent, NLP-driven conversational AI system that provides accessible, 24/7 emotional support for users experiencing stress, anxiety, sadness, or other emotional challenges.

Mental health services are often limited due to:

* High therapy costs
* Social stigma
* Limited availability
* Geographic barriers

This project addresses these challenges by using:

* Real-time Sentiment Analysis
* Emotion Detection Algorithms
* Context-aware Response Generation
* Voice-based Conversational Interface

MindfulAI automatically analyzes user input, detects emotional states, and responds with empathetic and supportive messages.


## 🚀 Features

* ✅ Real-time sentiment analysis
* ✅ Multi-emotion detection (Anxiety, Sadness, Anger, Stress, Loneliness, Neutral)
* ✅ Emotional intensity scoring (0.0 – 1.0 scale)
* ✅ Crisis keyword detection system
* ✅ Empathetic counselling response generation
* ✅ Voice-first conversational interface
* ✅ Text-to-Speech (TTS) and Speech-to-Text (STT)
* ✅ Conversation history tracking
* ✅ 24/7 availability
* ✅ Privacy-focused design


## 🛠️ Requirements

### 🖥️ Operating System

* Windows 10 / 11 (64-bit)
* Ubuntu (64-bit recommended)


### 🧑‍💻 Development Environment

* Python 3.9+
* Node.js 18+
* VS Code / PyCharm

### 🤖 Backend Framework

* FastAPI
* Uvicorn
* Pydantic
* httpx
* python-dotenv


### 🌐 Frontend Framework

* React.js
* Vite
* React Router DOM
* CSS3 (Glassmorphism UI)


### 🧠 AI & NLP Technologies

* Custom Sentiment Analysis Engine
* Emotion Classification Module
* Intensity Scoring Algorithm
* Crisis Detection Module
* Template-based Response Generator



### 🔧 Additional Dependencies

* NumPy
* Matplotlib
* WebSocket
* SpeechRecognition
* pyttsx3 (or external TTS API)
* Git (Version Control)

## 🏗️ System Architecture

The system architecture consists of the following components:

1. 🎤 Voice / Text Input
2. 🔄 Text Preprocessing Module
3. 🧠 Sentiment Analysis Engine
4. 📊 Emotion Detection & Intensity Scoring
5. 💬 Counselling Response Generator
6. 🔊 Text-to-Speech Output
7. 📂 Session Management & Logging
8. 🖥️ Monitoring Dashboard

## Program
agent.py
```
"""
AI Agent Module
Core AI agent implementation for mental health support
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
import asyncio

from sentiment_analyzer import SentimentAnalyzer, SentimentResult
from response_generator import ResponseGenerator, CounsellingResponse
from session_manager import SessionManager, Session
from utils import generate_id, log_info, log_error


@dataclass
class AgentState:
    """Current state of the AI agent"""
    is_listening: bool = False
    is_speaking: bool = False
    is_processing: bool = False
    current_emotion: Optional[str] = None
    conversation_turn: int = 0


@dataclass
class ConversationContext:
    """Context for ongoing conversation"""
    session_id: str
    user_name: Optional[str] = None
    topics_discussed: List[str] = field(default_factory=list)
    sentiment_history: List[str] = field(default_factory=list)
    primary_concerns: List[str] = field(default_factory=list)
    rapport_level: float = 0.0


class MentalHealthAgent:
    """
    AI Mental Health Support Agent
    
    Provides empathetic, sentiment-aware conversation support
    for mental health and emotional wellbeing.
    """
    
    def __init__(
        self, 
        agent_name: str = "MindfulAI",
        enable_sentiment: bool = True,
        enable_crisis_detection: bool = True
    ):
        """
        Initialize the mental health agent
        
        Args:
            agent_name: Display name for the agent
            enable_sentiment: Enable sentiment analysis
            enable_crisis_detection: Enable crisis keyword detection
        """
        self.agent_name = agent_name
        self.agent_id = generate_id("agent")
        self.enable_sentiment = enable_sentiment
        self.enable_crisis_detection = enable_crisis_detection
        
        # Initialize components
        self.sentiment_analyzer = SentimentAnalyzer()
        self.response_generator = ResponseGenerator()
        self.session_manager = SessionManager()
        
        # Agent state
        self.state = AgentState()
        self.contexts: Dict[str, ConversationContext] = {}
        
        # Statistics
        self.total_conversations = 0
        self.total_messages_processed = 0
        self.started_at = datetime.now()
        
        log_info(f"Agent initialized", agent_id=self.agent_id, name=self.agent_name)
    
    async def start_conversation(
        self, 
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Start a new conversation session
        
        Args:
            user_id: Optional user identifier
            
        Returns:
            Session information dict
        """
        # Create session
        session = self.session_manager.create_session(user_id)
        
        # Initialize conversation context
        self.contexts[session.session_id] = ConversationContext(
            session_id=session.session_id
        )
        
        self.total_conversations += 1
        self.state.conversation_turn = 0
        
        log_info(
            "Conversation started",
            session_id=session.session_id,
            total_conversations=self.total_conversations
        )
        
        # Generate greeting
        greeting = self._generate_greeting()
        
        return {
            "session_id": session.session_id,
            "status": "active",
            "greeting": greeting,
            "agent_name": self.agent_name,
            "created_at": session.created_at.isoformat()
        }
    
    async def process_message(
        self, 
        session_id: str, 
        message: str
    ) -> Dict[str, Any]:
        """
        Process a user message and generate response
        
        Args:
            session_id: Active session ID
            message: User's message content
            
        Returns:
            Response dict with sentiment analysis and response
        """
        self.state.is_processing = True
        self.total_messages_processed += 1
        
        try:
            # Get session
            session = self.session_manager.get_session(session_id)
            if not session:
                return {"error": "Session not found or expired"}
            
            # Get context
            context = self.contexts.get(session_id)
            if not context:
                context = ConversationContext(session_id=session_id)
                self.contexts[session_id] = context
            
            # Analyze sentiment
            sentiment = None
            if self.enable_sentiment:
                sentiment = self.sentiment_analyzer.analyze(message)
                
                # Update context
                context.sentiment_history.append(sentiment.sentiment.value)
                if sentiment.keywords:
                    context.topics_discussed.extend(sentiment.keywords)
            
            # Check for crisis
            crisis_detected = False
            if self.enable_crisis_detection and sentiment:
                crisis_detected = sentiment.requires_attention
            
            # Generate response
            response = self._generate_response(message, sentiment, context)
            
            # Store messages
            self.session_manager.add_message(
                session_id, "user", message, 
                sentiment.sentiment.value if sentiment else None
            )
            self.session_manager.add_message(
                session_id, "agent", response.message
            )
            
            # Update state
            self.state.conversation_turn += 1
            if sentiment:
                self.state.current_emotion = sentiment.emotions[0].value if sentiment.emotions else None
            
            return {
                "session_id": session_id,
                "response": response.message,
                "sentiment": {
                    "category": sentiment.sentiment.value if sentiment else None,
                    "confidence": sentiment.confidence if sentiment else None,
                    "emotions": [e.value for e in sentiment.emotions] if sentiment else [],
                    "intensity": sentiment.intensity if sentiment else None,
                } if sentiment else None,
                "strategy": response.strategy.value,
                "follow_up_questions": response.follow_up_questions,
                "suggested_techniques": response.suggested_techniques,
                "crisis_detected": crisis_detected,
                "turn": self.state.conversation_turn
            }
            
        except Exception as e:
            log_error("Error processing message", error=e, session_id=session_id)
            return {"error": str(e)}
        finally:
            self.state.is_processing = False
    
    async def end_conversation(
        self, 
        session_id: str,
        reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        End a conversation session
        
        Args:
            session_id: Session to end
            reason: Optional reason for ending
            
        Returns:
            Session summary dict
        """
        # Get summary before ending
        summary = self.session_manager.get_session_summary(session_id)
        
        # End session
        self.session_manager.end_session(session_id, reason)
        
        # Clean up context
        if session_id in self.contexts:
            del self.contexts[session_id]
        
        # Generate closing message
        closing = self._generate_closing()
        
        log_info(
            "Conversation ended",
            session_id=session_id,
            reason=reason,
            messages=summary.get("total_messages") if summary else 0
        )
        
        return {
            "session_id": session_id,
            "closing_message": closing,
            "summary": summary
        }
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        uptime = (datetime.now() - self.started_at).total_seconds()
        active_sessions = self.session_manager.get_active_session_count()
        
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "status": "online",
            "uptime_seconds": uptime,
            "total_conversations": self.total_conversations,
            "total_messages": self.total_messages_processed,
            "active_sessions": active_sessions,
            "state": {
                "is_listening": self.state.is_listening,
                "is_speaking": self.state.is_speaking,
                "is_processing": self.state.is_processing,
                "current_emotion": self.state.current_emotion
            }
        }
    
    def _generate_greeting(self) -> str:
        """Generate initial greeting"""
        greetings = [
            f"Hello! I'm {self.agent_name}, your mental wellness companion. How are you feeling today?",
            f"Hi there! Welcome. I'm here to listen and support you. What's on your mind?",
            f"Hello! I'm glad you're here. This is a safe space to share. How can I support you today?",
        ]
        import random
        return random.choice(greetings)
    
    def _generate_closing(self) -> str:
        """Generate closing message"""
        closings = [
            "Thank you for sharing with me today. Take care of yourself, and remember, it's okay to seek help.",
            "I appreciate you opening up. Remember, you're not alone. Take care until next time.",
            "Thank you for this conversation. Please be kind to yourself, and don't hesitate to reach out again.",
        ]
        import random
        return random.choice(closings)
    
    def _generate_response(
        self, 
        user_message: str,
        sentiment: Optional[SentimentResult],
        context: ConversationContext
    ) -> CounsellingResponse:
        """Generate counselling response"""
        if sentiment:
            return self.response_generator.generate_response(user_message, sentiment)
        
        # Fallback response without sentiment
        from response_generator import CounsellingResponse, ResponseStrategy
        return CounsellingResponse(
            message="I hear you. Can you tell me more about what you're experiencing?",
            strategy=ResponseStrategy.ACTIVE_LISTENING,
            follow_up_questions=["How does that make you feel?"],
            suggested_techniques=["Deep breathing"],
            empathy_score=0.7
        )


# Global agent instance
agent = MentalHealthAgent()


def get_agent() -> MentalHealthAgent:
    """Get the global agent instance"""
    return agent
```
config.py
```
"""
Configuration Settings
Application configuration using environment variables
"""

import os
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class AIConfig:
    """AI Service Configuration"""
    api_key: str
    agent_id: str
    model_name: str = "conversational-ai"
    voice_id: Optional[str] = None
    language: str = "en"
    
    @classmethod
    def from_env(cls) -> "AIConfig":
        """Create config from environment variables"""
        return cls(
            api_key=os.getenv("ELEVENLABS_API_KEY", ""),
            agent_id=os.getenv("ELEVENLABS_AGENT_ID", ""),
            voice_id=os.getenv("VOICE_ID"),
            language=os.getenv("LANGUAGE", "en")
        )


@dataclass
class ServerConfig:
    """Server Configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    reload: bool = True
    workers: int = 1
    
    # CORS settings
    cors_origins: list = None
    cors_methods: list = None
    cors_headers: list = None
    
    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = [
                "http://localhost:5173",
                "http://localhost:3000",
                "http://127.0.0.1:5173",
            ]
        if self.cors_methods is None:
            self.cors_methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        if self.cors_headers is None:
            self.cors_headers = ["*"]
    
    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Create config from environment variables"""
        return cls(
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8000")),
            debug=os.getenv("DEBUG", "true").lower() == "true",
        )


@dataclass
class SessionConfig:
    """Session Management Configuration"""
    timeout_minutes: int = 30
    max_messages: int = 100
    enable_history: bool = True
    save_transcripts: bool = False  # Privacy-focused default
    
    @classmethod
    def from_env(cls) -> "SessionConfig":
        """Create config from environment variables"""
        return cls(
            timeout_minutes=int(os.getenv("SESSION_TIMEOUT_MINUTES", "30")),
            max_messages=int(os.getenv("MAX_MESSAGES_PER_SESSION", "100")),
            save_transcripts=os.getenv("SAVE_TRANSCRIPTS", "false").lower() == "true"
        )


@dataclass
class SentimentConfig:
    """Sentiment Analysis Configuration"""
    enable_sentiment: bool = True
    enable_crisis_detection: bool = True
    intensity_threshold: float = 0.7
    min_confidence: float = 0.5
    
    @classmethod
    def from_env(cls) -> "SentimentConfig":
        """Create config from environment variables"""
        return cls(
            enable_sentiment=os.getenv("ENABLE_SENTIMENT", "true").lower() == "true",
            enable_crisis_detection=os.getenv("ENABLE_CRISIS_DETECTION", "true").lower() == "true",
            intensity_threshold=float(os.getenv("INTENSITY_THRESHOLD", "0.7"))
        )


@dataclass
class AppConfig:
    """Main Application Configuration"""
    app_name: str = "MindfulAI"
    version: str = "1.0.0"
    description: str = "AI-Powered Mental Health Support Chatbot"
    
    ai: AIConfig = None
    server: ServerConfig = None
    session: SessionConfig = None
    sentiment: SentimentConfig = None
    
    def __post_init__(self):
        if self.ai is None:
            self.ai = AIConfig.from_env()
        if self.server is None:
            self.server = ServerConfig.from_env()
        if self.session is None:
            self.session = SessionConfig.from_env()
        if self.sentiment is None:
            self.sentiment = SentimentConfig.from_env()
    
    @classmethod
    def load(cls) -> "AppConfig":
        """Load application configuration"""
        return cls()
    
    def validate(self) -> tuple:
        """
        Validate configuration
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        if not self.ai.api_key:
            errors.append("AI API key is not configured")
        
        if not self.ai.agent_id:
            errors.append("AI Agent ID is not configured")
        
        return len(errors) == 0, errors


# Global configuration instance
config = AppConfig.load()


def get_config() -> AppConfig:
    """Get application configuration"""
    return config


def reload_config() -> AppConfig:
    """Reload configuration from environment"""
    global config
    load_dotenv(override=True)
    config = AppConfig.load()
    return config
```
main.py
```
"""
AI Mental Health Support Chatbot - Backend API
FastAPI server with sentiment analysis and counselling response endpoints
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from datetime import datetime
import httpx

# Load environment variables
load_dotenv()

# Import modules
from models import (
    AnalyzeTextRequest, 
    SentimentResponse,
    HealthCheckResponse,
    ErrorResponse
)
from sentiment_analyzer import analyze_sentiment, SentimentType
from config import get_config

# Initialize FastAPI app
app = FastAPI(
    title="MindfulAI - Mental Health Support API",
    description="AI-powered mental health support chatbot using sentiment analysis and counselling response generation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# Health Check Endpoints
# ============================================

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - API information"""
    return {
        "name": "MindfulAI Mental Health Support API",
        "version": "1.0.0",
        "description": "AI-powered mental health support using sentiment analysis",
        "status": "online",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    config = get_config()
    
    return HealthCheckResponse(
        status="healthy",
        version="1.0.0",
        ai_services_available=bool(config.ai.api_key and config.ai.agent_id)
    )


# ============================================
# Voice Session Endpoints
# ============================================

@app.get("/api/signed-url", tags=["Voice Session"])
async def get_signed_url():
    """
    Get a signed URL for establishing a voice conversation session.
    This endpoint communicates with the AI service to get an authenticated
    session URL for real-time voice communication.
    """
    api_key = os.getenv("ELEVENLABS_API_KEY")
    agent_id = os.getenv("ELEVENLABS_AGENT_ID")
    
    if not api_key or not agent_id:
        raise HTTPException(
            status_code=500,
            detail="AI service not configured. Please add API_KEY and AGENT_ID to .env file."
        )
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://api.elevenlabs.io/v1/convai/conversation/get_signed_url?agent_id={agent_id}",
                headers={"xi-api-key": api_key},
                timeout=30.0
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"AI service error: {response.text}"
                )
            
            data = response.json()
            return {"signedUrl": data.get("signed_url")}
            
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="AI service timeout. Please try again."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to connect to AI service: {str(e)}"
        )


# ============================================
# Sentiment Analysis Endpoints
# ============================================

@app.post("/api/analyze", tags=["Sentiment Analysis"])
async def analyze_text(request: AnalyzeTextRequest):
    """
    Analyze text for sentiment and emotions.
    
    This endpoint performs sentiment analysis on the provided text,
    detecting emotional states and providing confidence scores.
    """
    try:
        result = analyze_sentiment(request.text)
        
        response = {
            "text": request.text,
            "sentiment": result.sentiment.value,
            "confidence": result.confidence,
            "intensity": result.intensity,
            "requires_attention": result.requires_attention,
            "timestamp": datetime.now().isoformat()
        }
        
        if request.include_emotions:
            response["emotions"] = [e.value for e in result.emotions]
        
        if request.include_keywords:
            response["keywords"] = result.keywords
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@app.get("/api/sentiment/categories", tags=["Sentiment Analysis"])
async def get_sentiment_categories():
    """Get available sentiment categories and emotions"""
    from sentiment_analyzer import SentimentType, EmotionType
    
    return {
        "sentiment_categories": [s.value for s in SentimentType],
        "emotion_types": [e.value for e in EmotionType],
        "description": {
            "positive": "User expressing happiness, gratitude, or hope",
            "negative": "User expressing sadness, anxiety, or frustration",
            "neutral": "No strong emotional indicators",
            "mixed": "Combination of positive and negative sentiments"
        }
    }


# ============================================
# Agent Status Endpoints
# ============================================

@app.get("/api/agent/status", tags=["Agent"])
async def get_agent_status():
    """Get the current status of the AI agent"""
    try:
        from agent import get_agent
        agent = get_agent()
        return agent.get_agent_status()
    except Exception as e:
        return {
            "agent_name": "MindfulAI",
            "status": "online",
            "message": "Agent ready for conversations"
        }


@app.get("/api/agent/info", tags=["Agent"])
async def get_agent_info():
    """Get information about the AI agent capabilities"""
    return {
        "name": "MindfulAI",
        "type": "Mental Health Support Agent",
        "capabilities": [
            "Real-time voice conversation",
            "Sentiment analysis",
            "Emotion detection",
            "Counselling response generation",
            "Crisis keyword detection",
            "Conversation transcription"
        ],
        "supported_languages": ["en"],
        "version": "1.0.0"
    }


# ============================================
# Configuration Endpoints
# ============================================

@app.get("/api/config/status", tags=["Configuration"])
async def get_config_status():
    """Check configuration status"""
    api_key = os.getenv("ELEVENLABS_API_KEY")
    agent_id = os.getenv("ELEVENLABS_AGENT_ID")
    
    return {
        "api_key_configured": bool(api_key),
        "agent_id_configured": bool(agent_id),
        "ready": bool(api_key and agent_id),
        "message": "All services configured" if (api_key and agent_id) 
                   else "Please configure API_KEY and AGENT_ID in .env file"
    }


# ============================================
# Error Handlers
# ============================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


# ============================================
# Startup / Shutdown Events
# ============================================

@app.on_event("startup")
async def startup_event():
    """Application startup"""
    print("=" * 50)
    print("🧠 MindfulAI Mental Health Support API")
    print("=" * 50)
    print("✅ Server started successfully")
    print("📊 Sentiment Analysis: Enabled")
    print("🎙️ Voice Support: Enabled")
    print("=" * 50)


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown"""
    print("🔌 Shutting down MindfulAI server...")
```
models.py
```
"""
Data Models for Mental Health Chatbot
Pydantic models for API request/response validation
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from enum import Enum


# ============================================
# Enums
# ============================================

class SessionStatus(str, Enum):
    """Session status enumeration"""
    ACTIVE = "active"
    PAUSED = "paused"
    ENDED = "ended"
    TIMEOUT = "timeout"


class MessageRole(str, Enum):
    """Message sender role"""
    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"


class SentimentCategory(str, Enum):
    """Sentiment categories"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class EmotionCategory(str, Enum):
    """Detected emotion categories"""
    HAPPY = "happy"
    SAD = "sad"
    ANXIOUS = "anxious"
    ANGRY = "angry"
    FEARFUL = "fearful"
    HOPEFUL = "hopeful"
    CONFUSED = "confused"
    LONELY = "lonely"
    STRESSED = "stressed"
    CALM = "calm"
    NEUTRAL = "neutral"


# ============================================
# Request Models
# ============================================

class StartSessionRequest(BaseModel):
    """Request to start a new conversation session"""
    user_id: Optional[str] = Field(
        None, 
        description="Optional user identifier for session tracking"
    )
    session_type: str = Field(
        default="voice",
        description="Type of session: 'voice' or 'text'"
    )
    language: str = Field(
        default="en",
        description="Preferred language code"
    )


class MessageRequest(BaseModel):
    """Request containing user message for analysis"""
    session_id: str = Field(..., description="Active session identifier")
    content: str = Field(..., description="Message content (text or transcript)")
    timestamp: Optional[datetime] = Field(
        default_factory=datetime.now,
        description="Message timestamp"
    )
    is_voice: bool = Field(
        default=True,
        description="Whether message was from voice input"
    )


class AnalyzeTextRequest(BaseModel):
    """Request to analyze text for sentiment"""
    text: str = Field(..., min_length=1, description="Text to analyze")
    include_emotions: bool = Field(
        default=True, 
        description="Include detailed emotion analysis"
    )
    include_keywords: bool = Field(
        default=True, 
        description="Include extracted keywords"
    )


class EndSessionRequest(BaseModel):
    """Request to end a conversation session"""
    session_id: str = Field(..., description="Session to end")
    reason: Optional[str] = Field(
        None, 
        description="Optional reason for ending session"
    )


# ============================================
# Response Models
# ============================================

class SentimentResponse(BaseModel):
    """Sentiment analysis response"""
    sentiment: SentimentCategory
    confidence: float = Field(..., ge=0.0, le=1.0)
    emotions: List[EmotionCategory]
    intensity: float = Field(..., ge=0.0, le=1.0)
    keywords: List[str]
    requires_attention: bool
    analysis_timestamp: datetime = Field(default_factory=datetime.now)


class CounsellingResponseModel(BaseModel):
    """Generated counselling response"""
    message: str
    strategy: str
    follow_up_questions: List[str]
    suggested_techniques: List[str]
    empathy_score: float = Field(..., ge=0.0, le=1.0)


class SessionResponse(BaseModel):
    """Session creation response"""
    session_id: str
    status: SessionStatus
    created_at: datetime
    signed_url: Optional[str] = None


class MessageResponse(BaseModel):
    """Response to a processed message"""
    message_id: str
    session_id: str
    role: MessageRole
    content: str
    sentiment: Optional[SentimentResponse] = None
    counselling_response: Optional[CounsellingResponseModel] = None
    timestamp: datetime


class ConversationMessage(BaseModel):
    """Single message in conversation history"""
    message_id: str
    role: MessageRole
    content: str
    sentiment: Optional[SentimentCategory] = None
    timestamp: datetime


class ConversationHistoryResponse(BaseModel):
    """Conversation history response"""
    session_id: str
    messages: List[ConversationMessage]
    message_count: int
    session_duration_seconds: int
    overall_sentiment: Optional[SentimentCategory] = None


class SessionSummaryResponse(BaseModel):
    """Session summary with analytics"""
    session_id: str
    status: SessionStatus
    start_time: datetime
    end_time: Optional[datetime]
    message_count: int
    user_message_count: int
    agent_message_count: int
    average_sentiment: Optional[SentimentCategory]
    primary_emotions: List[EmotionCategory]
    topics_discussed: List[str]


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    ai_services_available: bool
    timestamp: datetime = Field(default_factory=datetime.now)


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: Optional[str] = None
    error_code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


# ============================================
# Configuration Models
# ============================================

class AgentConfig(BaseModel):
    """AI Agent configuration"""
    agent_name: str = "MindfulAI"
    voice_id: Optional[str] = None
    language: str = "en"
    response_style: str = "empathetic"
    max_session_duration_minutes: int = 60
    enable_sentiment_analysis: bool = True
    enable_crisis_detection: bool = True


class SessionConfig(BaseModel):
    """Session configuration"""
    session_timeout_minutes: int = 30
    max_messages_per_session: int = 100
    enable_transcription: bool = True
    save_conversation_history: bool = False  # Privacy focused


# ============================================
# Analytics Models
# ============================================

class SentimentTrend(BaseModel):
    """Sentiment trend over time"""
    timestamp: datetime
    sentiment: SentimentCategory
    intensity: float


class SessionAnalytics(BaseModel):
    """Analytics for a completed session"""
    session_id: str
    duration_minutes: float
    total_messages: int
    sentiment_distribution: dict
    emotion_frequency: dict
    sentiment_trend: List[SentimentTrend]
    engagement_score: float
    resolution_status: Optional[str]
```
response_generator.py
```
"""
Counselling Response Generator
Generates empathetic, context-aware counselling responses
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Dict
import random

from sentiment_analyzer import SentimentType, EmotionType, SentimentResult


class ResponseStrategy(Enum):
    """Types of counselling response strategies"""
    ACTIVE_LISTENING = "active_listening"
    VALIDATION = "validation"
    EXPLORATION = "exploration"
    COPING_SUGGESTION = "coping_suggestion"
    REFRAMING = "reframing"
    ENCOURAGEMENT = "encouragement"
    GROUNDING = "grounding"
    RESOURCE_REFERRAL = "resource_referral"


@dataclass
class CounsellingResponse:
    """Generated counselling response"""
    message: str
    strategy: ResponseStrategy
    follow_up_questions: List[str]
    suggested_techniques: List[str]
    empathy_score: float


class ResponseGenerator:
    """
    AI Counselling Response Generator
    
    Generates contextually appropriate, empathetic responses
    based on sentiment analysis results.
    """
    
    # Response templates organized by strategy
    ACTIVE_LISTENING_TEMPLATES = [
        "I hear you saying that {summary}. That sounds really {emotion_adj}.",
        "It sounds like you're experiencing {emotion}. Thank you for sharing that with me.",
        "What I'm understanding is that {summary}. Is that right?",
        "So you're feeling {emotion} because of {topic}. That makes a lot of sense.",
    ]
    
    VALIDATION_TEMPLATES = [
        "It's completely understandable to feel {emotion} in this situation.",
        "Your feelings are valid. Many people would feel the same way.",
        "It makes sense that you're feeling this way. {emotion} is a natural response.",
        "What you're going through sounds really difficult, and it's okay to feel {emotion}.",
    ]
    
    EXPLORATION_TEMPLATES = [
        "Can you tell me more about what's making you feel this way?",
        "What do you think might be contributing to these feelings?",
        "How long have you been feeling like this?",
        "What was happening when you first started feeling {emotion}?",
    ]
    
    COPING_TEMPLATES = [
        "One thing that might help is taking a few deep breaths. Would you like to try that together?",
        "Sometimes it helps to focus on what we can control. What's one small thing you could do right now?",
        "Have you tried any relaxation techniques before? They can be helpful in moments like this.",
        "It might help to take a short break and do something calming. What activities usually help you feel better?",
    ]
    
    ENCOURAGEMENT_TEMPLATES = [
        "I want you to know that you're not alone in this. I'm here to support you.",
        "It takes courage to talk about these feelings. You're doing great by opening up.",
        "Remember that difficult times are temporary. You've gotten through hard things before.",
        "You're stronger than you know. Taking this step to talk about your feelings shows that.",
    ]
    
    GROUNDING_TEMPLATES = [
        "Let's try a grounding exercise. Can you name 5 things you can see around you right now?",
        "Take a moment to feel your feet on the ground. You are safe in this moment.",
        "Let's focus on the present. What's one thing you can hear right now?",
        "Try taking a slow, deep breath with me. Inhale for 4 counts, hold for 4, exhale for 4.",
    ]
    
    # Emotion-specific responses
    EMOTION_RESPONSES: Dict[EmotionType, List[str]] = {
        EmotionType.ANXIOUS: [
            "Anxiety can feel overwhelming, but remember that this feeling will pass.",
            "When anxiety is high, focusing on your breath can help bring you back to the present.",
        ],
        EmotionType.SAD: [
            "Sadness is a natural emotion, and it's okay to feel it fully.",
            "I'm sorry you're feeling down. Would you like to talk about what's on your mind?",
        ],
        EmotionType.ANGRY: [
            "It's okay to feel angry. Let's explore what's behind that anger.",
            "Anger often signals that something important to us has been hurt or threatened.",
        ],
        EmotionType.LONELY: [
            "Loneliness can be really painful. I'm glad you're reaching out.",
            "Even though you might feel alone, know that support is available.",
        ],
        EmotionType.STRESSED: [
            "Stress can build up over time. Let's think about ways to lighten the load.",
            "It sounds like you have a lot on your plate. What feels most pressing right now?",
        ],
    }
    
    # Follow-up questions by emotion
    FOLLOW_UP_QUESTIONS: Dict[EmotionType, List[str]] = {
        EmotionType.ANXIOUS: [
            "What thoughts come up when you feel anxious?",
            "Are there specific situations that trigger your anxiety?",
            "How does anxiety show up in your body?",
        ],
        EmotionType.SAD: [
            "How long have you been feeling this way?",
            "Is there a particular event that started these feelings?",
            "What usually helps you feel even a little bit better?",
        ],
        EmotionType.STRESSED: [
            "What are the main sources of stress in your life right now?",
            "How has this stress been affecting your daily life?",
            "What does your self-care routine look like?",
        ],
    }
    
    # Coping techniques
    COPING_TECHNIQUES = {
        EmotionType.ANXIOUS: [
            "4-7-8 Breathing technique",
            "Progressive muscle relaxation",
            "5-4-3-2-1 grounding exercise",
            "Mindful observation",
        ],
        EmotionType.SAD: [
            "Journaling your thoughts",
            "Gentle physical activity",
            "Connecting with a friend",
            "Engaging in a small, enjoyable activity",
        ],
        EmotionType.STRESSED: [
            "Breaking tasks into smaller steps",
            "Time-boxing and prioritization",
            "Taking short breaks",
            "Delegating when possible",
        ],
    }
    
    def __init__(self):
        """Initialize the response generator"""
        self.conversation_context = []
        self.response_count = 0
    
    def generate_response(
        self, 
        user_input: str, 
        sentiment: SentimentResult
    ) -> CounsellingResponse:
        """
        Generate an appropriate counselling response
        
        Args:
            user_input: The user's message
            sentiment: Sentiment analysis result
            
        Returns:
            CounsellingResponse with appropriate message and strategies
        """
        self.response_count += 1
        
        # Determine appropriate strategy
        strategy = self._select_strategy(sentiment)
        
        # Get primary emotion
        primary_emotion = sentiment.emotions[0] if sentiment.emotions else EmotionType.NEUTRAL
        
        # Generate main response
        message = self._generate_message(strategy, primary_emotion, user_input)
        
        # Get follow-up questions
        follow_ups = self._get_follow_up_questions(primary_emotion)
        
        # Get coping techniques
        techniques = self._get_coping_techniques(primary_emotion)
        
        # Calculate empathy score
        empathy_score = self._calculate_empathy_score(sentiment)
        
        # Update conversation context
        self.conversation_context.append({
            "user_input": user_input,
            "sentiment": sentiment.sentiment.value,
            "emotion": primary_emotion.value,
            "strategy": strategy.value
        })
        
        return CounsellingResponse(
            message=message,
            strategy=strategy,
            follow_up_questions=follow_ups,
            suggested_techniques=techniques,
            empathy_score=empathy_score
        )
    
    def _select_strategy(self, sentiment: SentimentResult) -> ResponseStrategy:
        """Select appropriate response strategy based on sentiment"""
        if sentiment.requires_attention:
            return ResponseStrategy.RESOURCE_REFERRAL
        
        if sentiment.intensity > 0.7:
            return ResponseStrategy.GROUNDING
        
        if EmotionType.ANXIOUS in sentiment.emotions:
            return ResponseStrategy.COPING_SUGGESTION
        
        if sentiment.sentiment == SentimentType.NEGATIVE:
            return ResponseStrategy.VALIDATION
        
        if len(self.conversation_context) < 2:
            return ResponseStrategy.ACTIVE_LISTENING
        
        return ResponseStrategy.EXPLORATION
    
    def _generate_message(
        self, 
        strategy: ResponseStrategy, 
        emotion: EmotionType,
        user_input: str
    ) -> str:
        """Generate response message based on strategy"""
        templates = {
            ResponseStrategy.ACTIVE_LISTENING: self.ACTIVE_LISTENING_TEMPLATES,
            ResponseStrategy.VALIDATION: self.VALIDATION_TEMPLATES,
            ResponseStrategy.EXPLORATION: self.EXPLORATION_TEMPLATES,
            ResponseStrategy.COPING_SUGGESTION: self.COPING_TEMPLATES,
            ResponseStrategy.ENCOURAGEMENT: self.ENCOURAGEMENT_TEMPLATES,
            ResponseStrategy.GROUNDING: self.GROUNDING_TEMPLATES,
        }
        
        # Get template for strategy
        template_list = templates.get(strategy, self.VALIDATION_TEMPLATES)
        template = random.choice(template_list)
        
        # Get emotion-specific addition
        emotion_response = ""
        if emotion in self.EMOTION_RESPONSES:
            emotion_response = random.choice(self.EMOTION_RESPONSES[emotion])
        
        # Format template
        emotion_adj = self._get_emotion_adjective(emotion)
        message = template.format(
            emotion=emotion.value,
            emotion_adj=emotion_adj,
            summary="you're going through a difficult time",
            topic="the situation"
        )
        
        # Combine with emotion response
        if emotion_response:
            message = f"{message} {emotion_response}"
        
        return message
    
    def _get_emotion_adjective(self, emotion: EmotionType) -> str:
        """Get adjective for emotion"""
        adjectives = {
            EmotionType.ANXIOUS: "overwhelming",
            EmotionType.SAD: "heavy",
            EmotionType.ANGRY: "frustrating",
            EmotionType.LONELY: "isolating",
            EmotionType.STRESSED: "intense",
            EmotionType.HAPPY: "wonderful",
            EmotionType.CALM: "peaceful",
        }
        return adjectives.get(emotion, "significant")
    
    def _get_follow_up_questions(self, emotion: EmotionType) -> List[str]:
        """Get relevant follow-up questions"""
        if emotion in self.FOLLOW_UP_QUESTIONS:
            return random.sample(
                self.FOLLOW_UP_QUESTIONS[emotion], 
                min(2, len(self.FOLLOW_UP_QUESTIONS[emotion]))
            )
        return ["Would you like to tell me more about that?"]
    
    def _get_coping_techniques(self, emotion: EmotionType) -> List[str]:
        """Get relevant coping techniques"""
        if emotion in self.COPING_TECHNIQUES:
            return self.COPING_TECHNIQUES[emotion][:2]
        return ["Deep breathing", "Mindfulness"]
    
    def _calculate_empathy_score(self, sentiment: SentimentResult) -> float:
        """Calculate empathy score for response quality"""
        # Base score
        score = 0.7
        
        # Adjust based on sentiment handling
        if sentiment.requires_attention:
            score = 0.9  # High empathy for crisis
        elif sentiment.intensity > 0.5:
            score = 0.85  # Higher empathy for intense emotions
        
        return round(score, 2)
    
    def reset_context(self):
        """Reset conversation context"""
        self.conversation_context = []


# Singleton instance
response_generator = ResponseGenerator()


def generate_counselling_response(
    user_input: str, 
    sentiment: SentimentResult
) -> CounsellingResponse:
    """
    Convenience function for generating counselling response
    
    Args:
        user_input: User's message
        sentiment: Sentiment analysis result
        
    Returns:
        CounsellingResponse object
    """
    return response_generator.generate_response(user_input, sentiment)
```
sentiment_analyzer.py
```
"""
Sentiment Analysis Module
Analyzes emotional states from text input using NLP techniques
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Optional
import re


class SentimentType(Enum):
    """Enumeration of sentiment categories"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class EmotionType(Enum):
    """Specific emotion categories for mental health context"""
    HAPPY = "happy"
    SAD = "sad"
    ANXIOUS = "anxious"
    ANGRY = "angry"
    FEARFUL = "fearful"
    HOPEFUL = "hopeful"
    CONFUSED = "confused"
    LONELY = "lonely"
    STRESSED = "stressed"
    CALM = "calm"
    NEUTRAL = "neutral"


@dataclass
class SentimentResult:
    """Result of sentiment analysis"""
    sentiment: SentimentType
    confidence: float
    emotions: List[EmotionType]
    intensity: float  # 0.0 to 1.0
    keywords: List[str]
    requires_attention: bool


class SentimentAnalyzer:
    """
    Sentiment Analysis Engine for Mental Health Context
    
    Uses keyword-based analysis with emotion detection
    optimized for mental health conversations.
    """
    
    # Keyword dictionaries for emotion detection
    POSITIVE_KEYWORDS = [
        "happy", "grateful", "thankful", "better", "good", "great",
        "hopeful", "excited", "calm", "peaceful", "relieved", "joy",
        "love", "appreciate", "wonderful", "amazing", "blessed"
    ]
    
    NEGATIVE_KEYWORDS = [
        "sad", "depressed", "anxious", "worried", "scared", "afraid",
        "angry", "frustrated", "upset", "stressed", "overwhelmed",
        "lonely", "isolated", "hopeless", "worthless", "tired",
        "exhausted", "hurt", "pain", "suffering", "difficult"
    ]
    
    ANXIETY_KEYWORDS = [
        "anxious", "anxiety", "worried", "nervous", "panic", "fear",
        "scared", "restless", "tense", "uneasy", "dread"
    ]
    
    DEPRESSION_KEYWORDS = [
        "depressed", "depression", "sad", "hopeless", "empty",
        "worthless", "numb", "tired", "exhausted", "alone"
    ]
    
    CRISIS_KEYWORDS = [
        "suicide", "suicidal", "kill myself", "end my life", "self-harm",
        "hurt myself", "don't want to live", "can't go on", "give up"
    ]
    
    def __init__(self):
        """Initialize the sentiment analyzer"""
        self.analysis_count = 0
    
    def analyze(self, text: str) -> SentimentResult:
        """
        Analyze text for sentiment and emotions
        
        Args:
            text: Input text to analyze
            
        Returns:
            SentimentResult with sentiment classification and emotions
        """
        self.analysis_count += 1
        text_lower = text.lower()
        words = self._tokenize(text_lower)
        
        # Count keyword matches
        positive_count = self._count_keywords(words, self.POSITIVE_KEYWORDS)
        negative_count = self._count_keywords(words, self.NEGATIVE_KEYWORDS)
        
        # Detect specific emotions
        emotions = self._detect_emotions(words)
        
        # Check for crisis indicators
        requires_attention = self._check_crisis(text_lower)
        
        # Determine sentiment
        sentiment, confidence = self._classify_sentiment(
            positive_count, negative_count, len(words)
        )
        
        # Calculate intensity
        intensity = self._calculate_intensity(
            positive_count, negative_count, len(words)
        )
        
        # Extract significant keywords
        keywords = self._extract_keywords(words)
        
        return SentimentResult(
            sentiment=sentiment,
            confidence=confidence,
            emotions=emotions,
            intensity=intensity,
            keywords=keywords,
            requires_attention=requires_attention
        )
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        # Remove punctuation and split
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()
    
    def _count_keywords(self, words: List[str], keywords: List[str]) -> int:
        """Count matching keywords"""
        return sum(1 for word in words if word in keywords)
    
    def _detect_emotions(self, words: List[str]) -> List[EmotionType]:
        """Detect specific emotions from text"""
        emotions = []
        
        # Check for anxiety
        if any(word in self.ANXIETY_KEYWORDS for word in words):
            emotions.append(EmotionType.ANXIOUS)
        
        # Check for depression indicators
        if any(word in self.DEPRESSION_KEYWORDS for word in words):
            emotions.append(EmotionType.SAD)
        
        # Check for anger
        if any(word in ["angry", "mad", "frustrated", "furious"] for word in words):
            emotions.append(EmotionType.ANGRY)
        
        # Check for loneliness
        if any(word in ["lonely", "alone", "isolated"] for word in words):
            emotions.append(EmotionType.LONELY)
        
        # Check for stress
        if any(word in ["stressed", "overwhelmed", "pressure"] for word in words):
            emotions.append(EmotionType.STRESSED)
        
        # Check for happiness
        if any(word in ["happy", "joy", "excited", "grateful"] for word in words):
            emotions.append(EmotionType.HAPPY)
        
        # Default to neutral if no emotions detected
        if not emotions:
            emotions.append(EmotionType.NEUTRAL)
        
        return emotions
    
    def _check_crisis(self, text: str) -> bool:
        """Check for crisis indicators"""
        return any(keyword in text for keyword in self.CRISIS_KEYWORDS)
    
    def _classify_sentiment(
        self, 
        positive: int, 
        negative: int, 
        total_words: int
    ) -> tuple:
        """Classify overall sentiment"""
        if total_words == 0:
            return SentimentType.NEUTRAL, 0.5
        
        if positive > negative:
            confidence = min(0.9, 0.5 + (positive - negative) / total_words)
            return SentimentType.POSITIVE, confidence
        elif negative > positive:
            confidence = min(0.9, 0.5 + (negative - positive) / total_words)
            return SentimentType.NEGATIVE, confidence
        elif positive > 0 and negative > 0:
            return SentimentType.MIXED, 0.6
        else:
            return SentimentType.NEUTRAL, 0.7
    
    def _calculate_intensity(
        self, 
        positive: int, 
        negative: int, 
        total_words: int
    ) -> float:
        """Calculate emotional intensity (0.0 to 1.0)"""
        if total_words == 0:
            return 0.0
        
        emotional_words = positive + negative
        intensity = min(1.0, emotional_words / max(1, total_words) * 3)
        return round(intensity, 2)
    
    def _extract_keywords(self, words: List[str]) -> List[str]:
        """Extract significant emotional keywords"""
        all_keywords = (
            self.POSITIVE_KEYWORDS + 
            self.NEGATIVE_KEYWORDS + 
            self.ANXIETY_KEYWORDS
        )
        return [word for word in words if word in all_keywords][:5]


# Singleton instance
sentiment_analyzer = SentimentAnalyzer()


def analyze_sentiment(text: str) -> SentimentResult:
    """
    Convenience function for sentiment analysis
    
    Args:
        text: Text to analyze
        
    Returns:
        SentimentResult object
    """
    return sentiment_analyzer.analyze(text)
```
session_manager.py
```
"""
Session Manager
Manages conversation sessions and state
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from dataclasses import dataclass, field


@dataclass
class Message:
    """Single conversation message"""
    message_id: str
    role: str  # 'user' or 'agent'
    content: str
    sentiment: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Session:
    """Conversation session"""
    session_id: str
    user_id: Optional[str]
    created_at: datetime
    last_activity: datetime
    status: str  # 'active', 'paused', 'ended'
    messages: List[Message] = field(default_factory=list)
    sentiment_history: List[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class SessionManager:
    """
    Manages conversation sessions for the mental health chatbot
    
    Handles session lifecycle, message storage, and session analytics.
    """
    
    def __init__(self, session_timeout_minutes: int = 30):
        """
        Initialize session manager
        
        Args:
            session_timeout_minutes: Session timeout in minutes
        """
        self.sessions: Dict[str, Session] = {}
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
    
    def create_session(self, user_id: Optional[str] = None) -> Session:
        """
        Create a new conversation session
        
        Args:
            user_id: Optional user identifier
            
        Returns:
            New Session object
        """
        session_id = self._generate_session_id()
        now = datetime.now()
        
        session = Session(
            session_id=session_id,
            user_id=user_id,
            created_at=now,
            last_activity=now,
            status="active"
        )
        
        self.sessions[session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """
        Get session by ID
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session object or None if not found
        """
        session = self.sessions.get(session_id)
        
        if session and self._is_session_expired(session):
            self.end_session(session_id, reason="timeout")
            return None
        
        return session
    
    def add_message(
        self, 
        session_id: str, 
        role: str, 
        content: str,
        sentiment: Optional[str] = None
    ) -> Optional[Message]:
        """
        Add a message to a session
        
        Args:
            session_id: Session identifier
            role: Message sender role ('user' or 'agent')
            content: Message content
            sentiment: Optional sentiment classification
            
        Returns:
            Message object or None if session not found
        """
        session = self.get_session(session_id)
        if not session:
            return None
        
        message = Message(
            message_id=self._generate_message_id(),
            role=role,
            content=content,
            sentiment=sentiment
        )
        
        session.messages.append(message)
        session.last_activity = datetime.now()
        
        if sentiment:
            session.sentiment_history.append({
                "sentiment": sentiment,
                "timestamp": message.timestamp.isoformat()
            })
        
        return message
    
    def end_session(
        self, 
        session_id: str, 
        reason: Optional[str] = None
    ) -> bool:
        """
        End a conversation session
        
        Args:
            session_id: Session identifier
            reason: Optional reason for ending
            
        Returns:
            True if session was ended, False if not found
        """
        session = self.sessions.get(session_id)
        if not session:
            return False
        
        session.status = "ended"
        session.metadata["end_reason"] = reason or "user_ended"
        session.metadata["ended_at"] = datetime.now().isoformat()
        
        return True
    
    def get_session_summary(self, session_id: str) -> Optional[dict]:
        """
        Get summary of a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session summary dict or None
        """
        session = self.sessions.get(session_id)
        if not session:
            return None
        
        user_messages = [m for m in session.messages if m.role == "user"]
        agent_messages = [m for m in session.messages if m.role == "agent"]
        
        # Calculate average sentiment
        sentiments = [s["sentiment"] for s in session.sentiment_history]
        avg_sentiment = self._calculate_average_sentiment(sentiments)
        
        duration = (session.last_activity - session.created_at).total_seconds()
        
        return {
            "session_id": session_id,
            "status": session.status,
            "created_at": session.created_at.isoformat(),
            "duration_seconds": duration,
            "total_messages": len(session.messages),
            "user_messages": len(user_messages),
            "agent_messages": len(agent_messages),
            "average_sentiment": avg_sentiment,
            "sentiment_history": session.sentiment_history
        }
    
    def get_conversation_history(
        self, 
        session_id: str, 
        limit: Optional[int] = None
    ) -> List[dict]:
        """
        Get conversation history for a session
        
        Args:
            session_id: Session identifier
            limit: Optional limit on messages returned
            
        Returns:
            List of message dicts
        """
        session = self.sessions.get(session_id)
        if not session:
            return []
        
        messages = session.messages
        if limit:
            messages = messages[-limit:]
        
        return [
            {
                "message_id": m.message_id,
                "role": m.role,
                "content": m.content,
                "sentiment": m.sentiment,
                "timestamp": m.timestamp.isoformat()
            }
            for m in messages
        ]
    
    def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions
        
        Returns:
            Number of sessions cleaned up
        """
        expired = []
        for session_id, session in self.sessions.items():
            if self._is_session_expired(session):
                expired.append(session_id)
        
        for session_id in expired:
            self.end_session(session_id, reason="timeout")
        
        return len(expired)
    
    def get_active_session_count(self) -> int:
        """Get count of active sessions"""
        return sum(
            1 for s in self.sessions.values() 
            if s.status == "active"
        )
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        return f"session_{uuid.uuid4().hex[:12]}"
    
    def _generate_message_id(self) -> str:
        """Generate unique message ID"""
        return f"msg_{uuid.uuid4().hex[:8]}"
    
    def _is_session_expired(self, session: Session) -> bool:
        """Check if session has expired"""
        if session.status != "active":
            return False
        return datetime.now() - session.last_activity > self.session_timeout
    
    def _calculate_average_sentiment(self, sentiments: List[str]) -> Optional[str]:
        """Calculate average sentiment from list"""
        if not sentiments:
            return None
        
        sentiment_scores = {
            "positive": 1,
            "neutral": 0,
            "negative": -1,
            "mixed": 0
        }
        
        scores = [sentiment_scores.get(s, 0) for s in sentiments]
        avg = sum(scores) / len(scores)
        
        if avg > 0.3:
            return "positive"
        elif avg < -0.3:
            return "negative"
        else:
            return "neutral"


# Singleton instance
session_manager = SessionManager()
```
utils.py
```
"""
Utility Functions
Common utilities for the mental health chatbot backend
"""

import re
import uuid
import hashlib
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MindfulAI")


# ============================================
# Text Processing Utilities
# ============================================

def clean_text(text: str) -> str:
    """
    Clean and normalize text input
    
    Args:
        text: Raw text input
        
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,!?\'"-]', '', text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    """
    Simple word tokenization
    
    Args:
        text: Text to tokenize
        
    Returns:
        List of tokens
    """
    text = clean_text(text.lower())
    return text.split()


def extract_keywords(text: str, min_length: int = 3) -> List[str]:
    """
    Extract keywords from text
    
    Args:
        text: Text to extract from
        min_length: Minimum word length
        
    Returns:
        List of keywords
    """
    # Common stop words to filter
    stop_words = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'could', 'should', 'may', 'might', 'must', 'shall',
        'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
        'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
        'through', 'during', 'before', 'after', 'above', 'below',
        'between', 'under', 'again', 'further', 'then', 'once',
        'here', 'there', 'when', 'where', 'why', 'how', 'all',
        'each', 'few', 'more', 'most', 'other', 'some', 'such',
        'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
        'too', 'very', 'just', 'and', 'but', 'if', 'or', 'because',
        'until', 'while', 'although', 'i', 'me', 'my', 'myself',
        'we', 'our', 'you', 'your', 'he', 'him', 'she', 'her',
        'it', 'its', 'they', 'them', 'their', 'what', 'which',
        'who', 'whom', 'this', 'that', 'these', 'those', 'am',
        'im', 'ive', 'dont', 'cant', 'wont', 'didnt', 'isnt',
        'arent', 'wasnt', 'werent', 'hasnt', 'havent', 'hadnt',
        'doesnt', 'really', 'like', 'feel', 'feeling', 'think',
        'know', 'get', 'got', 'going', 'want', 'make'
    }
    
    tokens = tokenize(text)
    keywords = [
        word for word in tokens 
        if len(word) >= min_length and word not in stop_words
    ]
    
    # Return unique keywords
    return list(dict.fromkeys(keywords))


def word_count(text: str) -> int:
    """Count words in text"""
    return len(tokenize(text))


# ============================================
# ID Generation Utilities
# ============================================

def generate_id(prefix: str = "") -> str:
    """
    Generate a unique ID
    
    Args:
        prefix: Optional prefix for the ID
        
    Returns:
        Unique ID string
    """
    unique_id = uuid.uuid4().hex[:12]
    if prefix:
        return f"{prefix}_{unique_id}"
    return unique_id


def generate_session_id() -> str:
    """Generate a unique session ID"""
    return generate_id("session")


def generate_message_id() -> str:
    """Generate a unique message ID"""
    return generate_id("msg")


def hash_text(text: str) -> str:
    """
    Create a hash of text (for anonymization)
    
    Args:
        text: Text to hash
        
    Returns:
        SHA256 hash
    """
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# ============================================
# Time Utilities
# ============================================

def get_timestamp() -> str:
    """Get current ISO timestamp"""
    return datetime.now().isoformat()


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string
    """
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def seconds_since(start_time: datetime) -> float:
    """Calculate seconds since a given time"""
    return (datetime.now() - start_time).total_seconds()


# ============================================
# Validation Utilities
# ============================================

def validate_session_id(session_id: str) -> bool:
    """
    Validate session ID format
    
    Args:
        session_id: Session ID to validate
        
    Returns:
        True if valid
    """
    pattern = r'^session_[a-f0-9]{12}$'
    return bool(re.match(pattern, session_id))


def validate_message_content(content: str, max_length: int = 5000) -> tuple:
    """
    Validate message content
    
    Args:
        content: Message content
        max_length: Maximum allowed length
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not content or not content.strip():
        return False, "Message content cannot be empty"
    
    if len(content) > max_length:
        return False, f"Message exceeds maximum length of {max_length} characters"
    
    return True, None


def sanitize_input(text: str) -> str:
    """
    Sanitize user input for security
    
    Args:
        text: Raw input
        
    Returns:
        Sanitized text
    """
    # Remove potential script tags
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Limit length
    return text[:5000]


# ============================================
# Response Helpers
# ============================================

def create_success_response(data: Any, message: str = "Success") -> Dict:
    """Create a standardized success response"""
    return {
        "success": True,
        "message": message,
        "data": data,
        "timestamp": get_timestamp()
    }


def create_error_response(
    error: str, 
    detail: Optional[str] = None,
    code: Optional[str] = None
) -> Dict:
    """Create a standardized error response"""
    return {
        "success": False,
        "error": error,
        "detail": detail,
        "error_code": code,
        "timestamp": get_timestamp()
    }


# ============================================
# Logging Helpers
# ============================================

def log_info(message: str, **kwargs):
    """Log info message with context"""
    extra = " ".join(f"{k}={v}" for k, v in kwargs.items())
    logger.info(f"{message} {extra}".strip())


def log_error(message: str, error: Optional[Exception] = None, **kwargs):
    """Log error message with context"""
    extra = " ".join(f"{k}={v}" for k, v in kwargs.items())
    if error:
        logger.error(f"{message} {extra} error={str(error)}")
    else:
        logger.error(f"{message} {extra}".strip())


def log_warning(message: str, **kwargs):
    """Log warning message with context"""
    extra = " ".join(f"{k}={v}" for k, v in kwargs.items())
    logger.warning(f"{message} {extra}".strip())

```



### 🔄 Workflow

```
User Input (Voice/Text)
        ↓
Speech-to-Text Conversion
        ↓
Text Preprocessing
        ↓
Sentiment Analysis Engine
        ↓
Emotion Detection & Intensity Scoring
        ↓
Response Strategy Selection
        ↓
Empathetic Response Generation
        ↓
Text + Voice Output
```


## 📂 Project Structure

```
MindfulAI/
│
├── backend/
│   ├── main.py
│   ├── sentiment_analyzer.py
│   ├── response_generator.py
│   ├── session_manager.py
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── VoiceOrb.jsx
│   │   │   ├── ChatInterface.jsx
│   │   │   └── Sidebar.jsx
│   │   └── App.jsx
│   └── package.json
│
└── README.md
```



## ⚙️ Installation Guide

### 🔹 Step 1: Clone Repository

```bash
git clone https://github.com/your-username/MindfulAI.git
cd MindfulAI
```


### 🔹 Step 2: Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate    # Ubuntu
venv\Scripts\activate       # Windows
```

Install dependencies:

```bash
pip install fastapi uvicorn httpx pydantic python-dotenv
```

Run backend:

```bash
uvicorn main:app --reload --port 8000
```


### 🔹 Step 3: Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Access at:

```
http://localhost:5173
```



## 📊 Output

* 🎤 Voice-based interaction
* 📊 Real-time emotion detection
* 💬 Empathetic AI responses
* 📂 Conversation transcript display
* 🚨 Crisis alert detection


## 📈 Results and Impact

MindfulAI improves mental health accessibility by:

* 🌍 Providing 24/7 emotional support
* 💰 Reducing cost barriers
* 🧘 Offering a judgment-free environment
* ⚡ Delivering instant responses
* 🧠 Detecting emotional distress in real time

This project demonstrates the effectiveness of AI-driven therapeutic assistance and lays the foundation for:

* AI-based mental wellness platforms
* Intelligent emotional support systems
* Digital therapeutic applications


## 🔮 Future Enhancements

* 🤖 BERT-based advanced sentiment classification
* 🌐 Multi-language support
* 📱 Mobile application deployment
* 📊 Long-term mood tracking dashboard
* 🧑‍⚕️ Professional therapist integration
* 🧘 Guided meditation and breathing exercises
* 🔍 Facial emotion recognition integration

## 📚 References

1. Devlin, J., et al., “BERT: Pre-training of Deep Bidirectional Transformers,” NAACL-HLT, 2019.
2. Vaswani, A., et al., “Attention Is All You Need,” NeurIPS, 2017.
3. Research papers on AI-based Mental Health Chatbots and Sentiment Analysis (2023–2025).

