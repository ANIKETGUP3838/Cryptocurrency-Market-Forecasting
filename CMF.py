import streamlit as st
import pdfplumber
import os
import re

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from openai import RateLimitError

# =====================================================
# STREAMLIT CONFIG
# =====================================================
st.set_page_config(page_title="AI Fact Checker", layout="wide")
st.title("🕵️ AI Fact-Checking Web App")
st.write("Upload a PDF to verify factual claims using **live web data**.")

# =====================================================
# LOAD API KEYS (MANDATORY)
# =====================================================
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY", os.getenv("TAVILY_API_KEY"))

if not OPENAI_API_KEY or not TAVILY_API_KEY:
    st.error("❌ API keys missing. Live verification cannot run.")
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# =====================================================
# INITIALIZE MODELS
# =====================================================
try:
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0
    )
    search_tool = TavilySearchResults(max_results=3)
except Exception as e:
    st.error("❌ Failed to initialize APIs")
    st.exception(e)
    st.stop()

quota_exhausted = False  # 🔑 important flag

# =====================================================
# FUNCTIONS
# =====================================================
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text


def regex_claim_extraction(text):
    """Fallback extraction without OpenAI"""
    lines = text.split("\n")
    claims = [l for l in lines if re.search(r"\d", l)]
    return claims[:5]


def extract_claims(text):
    global quota_exhausted

    prompt = f"""
    Extract ONLY factual, verifiable claims from the text below.
    Claims must include numbers, dates, statistics, or measurable facts.
    Return each claim on a new line.

    TEXT:
    {text}
    """

    if quota_exhausted:
        return regex_claim_extraction(text)

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        lines = response.content.split("\n")
        claims = [l.strip("-• ") for l in lines if re.search(r"\d", l)]
        return list(dict.fromkeys(claims))

    except RateLimitError:
        quota_exhausted = True
        st.warning(
            "⚠️ OpenAI quota exhausted.\n"
            "Using fallback claim extraction (no LLM)."
        )
        return regex_claim_extraction(text)

    except Exception as e:
        st.error("❌ OpenAI claim extraction failed")
        st.exception(e)
        st.stop()


def verify_claim(claim):
    try:
        search_results = search_tool.run(claim)
    except Exception as e:
        st.error("❌ Tavily search failed")
        st.exception(e)
        st.stop()

    prompt = f"""
    Claim: {claim}

    Search Results:
    {search_results}

    Classify the claim as:
    - Verified
    - Inaccurate
    - False

    Respond in this format:
    Status: <Verified/Inaccurate/False>
    Explanation: <1–2 lines explanation>
    """

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content, search_results

    except RateLimitError:
        st.error(
            "❌ OpenAI quota exhausted during verification.\n"
            "Live verification cannot continue."
        )
        st.stop()

    except Exception as e:
        st.error("❌ OpenAI verification failed")
        st.exception(e)
        st.stop()

# =====================================================
# UI
# =====================================================
uploaded_file = st.file_uploader("📄 Upload PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("📖 Reading PDF..."):
        text = extract_text_from_pdf(uploaded_file)

    with st.spinner("🧠 Extracting factual claims..."):
        claims = extract_claims(text)

    if not claims:
        st.warning("No factual claims detected.")
        st.stop()

    st.subheader("📌 Extracted Claims")
    for i, claim in enumerate(claims, 1):
        st.markdown(f"**{i}. {claim}**")

    st.subheader("🔍 Live Verification Results")

    for i, claim in enumerate(claims, 1):
        with st.spinner(f"Verifying claim {i}..."):
            verdict, sources = verify_claim(claim)

        st.markdown("---")
        st.markdown(f"### Claim {i}")
        st.markdown(f"**{claim}**")
        st.markdown(verdict)

        if sources:
            st.markdown("**Sources:**")
            st.json(sources)
