# AI Fact-Checking Web App

A web application that extracts factual claims from PDFs and verifies them using
**live web data**. The app is designed to be robust against API limitations and
remain usable during evaluation.

---

## 🚀 Live Demo
🔗 **App URL:**  
[App](https://ai-fact-checking-web-app-hsnexgdnlm7vkpy7hxqssw.streamlit.app/)

---

## 🎯 Objective

To build a fact-checking layer that:
- Extracts factual claims (numbers, dates, statistics)
- Verifies them using real-time web search
- Flags claims as **Verified**, **Inaccurate**, or **False**

---

## 🧠 How It Works

### Claim Extraction
- Primary: OpenAI LLM (when available)
- Fallback: Regex-based extraction when quota is exhausted

### Live Verification
- Uses Tavily Web Search API
- Verifies claims using live web sources

### Reporting
- Displays claim
- Verification status
- Explanation
- Source links

---

## 📂 Project Structure

```text
fact-checker-app/
│
├── AI_FACT CHECKER.py
├── requirements.txt
├── README.md
└── .streamlit/
    └── config.toml
```
---

## 🧰 Tech Stack

- Streamlit
- Python
- OpenAI (GPT-3.5-turbo)
- Tavily Search API
- LangChain
- pdfplumber

---

## 🔐 Environment Variables

Set the following in **Streamlit Secrets**:
```text
OPENAI_API_KEY=your_openai_key
TAVILY_API_KEY=your_tavily_key
```

---

## 🛡️ API Quota Handling

If OpenAI quota is exhausted:
- Claim extraction falls back to regex-based logic
- Live web verification continues using Tavily
- The app does not crash

This ensures uninterrupted evaluation.

---

## 🎥 Demo Video
📹 A short screen recording showing:
- PDF upload
- Claim extraction
- Live verification

(Add demo video link here)

---

## 👤 Author

**Aniket**  
Final Year B.Tech CSE (Data Science)
