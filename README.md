# News Sentiment Analyzer (AI + Stock News)

A lightweight end-to-end AI system that scrapes live stock market news from stocktitan.net and analyzes sentiment using a transformer-based model (FinBERT).

---

## What it does

- Scrapes live financial news from StockTitan  
- Extracts ticker symbols, titles, tags, and article URLs  
- Sends headlines to an AI sentiment analysis service (Flask API)  
- Uses a FinBERT transformer model for sentiment classification  
- Returns sentiment labels with confidence scores  
- The output is returned as raw JSON and displayed as formatted text in the UI  

---

## Architecture

- **FastAPI (`api/`)** → Simple web UI + request handling  
- **Scraper (`scraper/`)** → Collects live news articles  
- **AI Service (`finbert_service/`)** → Sentiment analysis API (Flask + FinBERT model)  
- **Data (`data/`)** → Sample and stored outputs  
- **Experimental (`experimental/`)** → Prototypes and testing code  

### Data flow

User → FastAPI UI → Scraper → AI Service → Sentiment Results → UI  

---

## How to run

### 1. Start AI service

Run:
python finbert_service/server.py

---

### 2. Start FastAPI app

Run:
uvicorn api.main:app --reload in a separate terminal

---

### 3. Open in browser

Open:
http://127.0.0.1:8000

---

### 4. Run the Analyzer

1. Provide number of articles to analyze and click "Analyze"  
2. NOTE: Wait ~5 seconds between requests to avoid overloading the scraper and AI service  

---

## Features

- Live stock news scraping  
- Batch AI sentiment analysis using FinBERT transformer model  
- Simple web interface  
- Modular design (scraper / API / AI service separated)  

---

## Notes

- Uses Hugging Face `ProsusAI/finbert` transformer model locally  
- Built for AI experimentation and learning  
- Data source: StockTitan live news feed  


## Future improvements
- Better, more usable UI
- Improved reports
- Deeper analysis of full article body