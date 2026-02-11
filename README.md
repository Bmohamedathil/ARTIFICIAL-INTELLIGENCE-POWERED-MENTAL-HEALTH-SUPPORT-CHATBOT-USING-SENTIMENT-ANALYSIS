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

