import streamlit as st
import pdfplumber
import os
import re

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults

# =====================================================
# GLOBAL FLAGS (IMPORTANT CORRECTION)
# =====================================================
DEMO_MODE = False
OPENAI_DISABLED = False   # <- prevents repeated failures

# =====================================================
# STREAMLIT CONFIG
# =====================================================
st.set_page_config(page_title="AI Fact Checker", layout="wide")
st.title("🕵️ AI Fact-Checking Web App")
st.write("Upload a PDF to verify factual claims using live web data.")

# =====================================================
# LOAD API KEYS (OPTIONAL)
# =====================================================
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY", os.getenv("TAVILY_API_KEY"))

if not OPENAI_API_KEY or not TAVILY_API_KEY:
    DEMO_MODE = True
    st.info("ℹ️ Running in DEMO MODE (API keys unavailable)")

if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
if TAVILY_API_KEY:
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# =====================================================
# INITIALIZE MODELS (ONLY ONCE)
# =====================================================
if not DEMO_MODE:
    try:
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",  # stable model
            temperature=0
        )
        search_tool = TavilySearchResults(max_results=3)
    except Exception:
        DEMO_MODE = True
        OPENAI_DISABLED = True
        st.info("ℹ️ Switched to DEMO MODE")

# =====================================================
# FUNCTIONS
# =====================================================
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def extract_claims(text):
    global DEMO_MODE, OPENAI_DISABLED

    # ---------- DEMO MODE ----------
    if DEMO_MODE or OPENAI_DISABLED:
        lines = text.split("\n")
        return [l for l in lines if re.search(r"\d", l)][:5]

    # ---------- LIVE MODE ----------
    prompt = f"""
    Extract ONLY factual, verifiable claims from the text below.
    Claims must include numbers, dates, statistics, or measurable facts.
    Return each claim on a new line.

    TEXT:
    {text}
    """

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        lines = response.content.split("\n")
        claims = [l.strip("-• ") for l in lines if re.search(r"\d", l)]
        return list(dict.fromkeys(claims))

    except Exception:
        OPENAI_DISABLED = True
        DEMO_MODE = True
        return extract_claims(text)


def verify_claim(claim):
    global DEMO_MODE, OPENAI_DISABLED

    # ---------- DEMO MODE ----------
    if DEMO_MODE or OPENAI_DISABLED:
        return (
            "Status: Inaccurate\n"
            "Explanation: Demo mode result (live verification unavailable)."
        ), None

    # ---------- LIVE MODE ----------
    try:
        search_results = search_tool.run(claim)

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

        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content, search_results

    except Exception:
        OPENAI_DISABLED = True
        DEMO_MODE = True
        return verify_claim(claim)


# =====================================================
# UI
# =====================================================
uploaded_file = st.file_uploader("📄 Upload PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("📖 Reading PDF..."):
        text = extract_text_from_pdf(uploaded_file)

    with st.spinner("🧠 Extracting claims..."):
        claims = extract_claims(text)

    if not claims:
        st.warning("No factual claims detected.")
        st.stop()

    st.subheader("📌 Extracted Claims")
    for i, claim in enumerate(claims, 1):
        st.markdown(f"**{i}. {claim}**")

    st.subheader("🔍 Verification Results")

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
