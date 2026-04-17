import json
import re
from urllib.parse import urlparse

import requests
import streamlit as st
from bs4 import BeautifulSoup

from openai import OpenAI


st.set_page_config(page_title="SEO Content Gap Analyzer", layout="wide")

USER_AGENT = "Mozilla/5.0"

BLOCKED_DOMAINS = {
    "youtube.com",
    "instagram.com",
    "facebook.com",
    "linkedin.com",
    "tiktok.com",
}


# -----------------------
# helpers
# -----------------------

def clean_text(text):
    return re.sub(r"\s+", " ", text or "").strip()


def fetch_page(url):

    try:

        r = requests.get(
            url,
            timeout=20,
            headers={"User-Agent": USER_AGENT}
        )

        html = r.text

        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text = clean_text(soup.get_text(" "))

        return html, text[:20000]

    except:
        return "", ""


def extract_metadata(html):

    soup = BeautifulSoup(html, "html.parser")

    title = ""
    h1 = ""
    meta = ""

    if soup.title:
        title = soup.title.get_text(strip=True)

    h1_tag = soup.find("h1")
    if h1_tag:
        h1 = h1_tag.get_text(strip=True)

    meta_tag = soup.find("meta", attrs={"name": "description"})
    if meta_tag:
        meta = meta_tag.get("content", "")

    return {
        "title": title,
        "h1": h1,
        "meta": meta
    }


# -----------------------
# SERP
# -----------------------

def get_serp(keyword, api_key, language, country):

    url = "https://google.serper.dev/search"

    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json"
    }

    payload = {
        "q": keyword,
        "hl": language,
        "gl": country,
        "num": 10
    }

    r = requests.post(url, json=payload, headers=headers)

    data = r.json()

    results = []

    for item in data.get("organic", []):

        link = item.get("link")

        if not link:
            continue

        domain = urlparse(link).netloc

        if any(d in domain for d in BLOCKED_DOMAINS):
            continue

        results.append({
            "title": item.get("title"),
            "link": link,
            "snippet": item.get("snippet")
        })

    return results[:5]


def get_paa(keyword, api_key, language, country):

    url = "https://serpapi.com/search.json"

    params = {
        "engine": "google",
        "q": keyword,
        "hl": language,
        "gl": country,
        "api_key": api_key
    }

    r = requests.get(url, params=params)

    data = r.json()

    questions = []

    for q in data.get("related_questions", []):

        question = q.get("question")

        if question:
            questions.append(question)

    return questions[:10]


# -----------------------
# tree print
# -----------------------

def print_tree(tree, level=0):

    for node in tree:

        indent = "  " * level

        topic = node.get("topic", "topic")

        st.write(f"{indent}- {topic}")

        children = node.get("children", [])

        if children:
            print_tree(children, level + 1)


# -----------------------
# openai
# -----------------------

def extract_json(text):

    try:
        return json.loads(text)
    except:

        match = re.search(r"\{.*\}", text, re.DOTALL)

        if match:
            try:
                return json.loads(match.group(0))
            except:
                pass

    return {"tree": []}


def call_llm_json(client, prompt):

    try:

        response = client.responses.create(
            model="gpt-5.4",
            input=prompt
        )

        text = response.output_text

        return extract_json(text)

    except Exception as e:

        st.error(f"OpenAI error: {e}")

        return {"tree": []}


def call_llm_text(client, prompt):

    try:

        response = client.responses.create(
            model="gpt-5.4",
            input=prompt
        )

        return response.output_text

    except Exception as e:

        st.error(f"OpenAI error: {e}")

        return ""


# -----------------------
# sidebar
# -----------------------

st.sidebar.title("API Keys")

OPENAI_KEY = st.sidebar.text_input("OpenAI API Key", type="password")
SERPER_KEY = st.sidebar.text_input("Serper API Key", type="password")
SERPAPI_KEY = st.sidebar.text_input("SerpAPI Key", type="password")

st.sidebar.subheader("SERP Settings")

LANGUAGE = st.sidebar.selectbox(
    "Language",
    ["it", "en", "fr", "es", "de"]
)

COUNTRY = st.sidebar.selectbox(
    "Country",
    ["it", "us", "uk", "fr", "de", "es"]
)

client = OpenAI(api_key=OPENAI_KEY)


# -----------------------
# UI
# -----------------------

st.title("SEO Content Gap Analyzer")

keyword = st.text_input("Keyword")
h1 = st.text_input("Titolo H1")
url = st.text_input("URL")


# -----------------------
# analysis
# -----------------------

if st.button("Run Analysis"):

    serp = get_serp(keyword, SERPER_KEY, LANGUAGE, COUNTRY)

    paa = get_paa(keyword, SERPAPI_KEY, LANGUAGE, COUNTRY)

    html, text = fetch_page(url)

    competitors_text = ""

    for c in serp:

        _, ctext = fetch_page(c["link"])

        competitors_text += ctext[:3000]


    prompt = f"""
You are a senior SEO strategist.

Output language must be: {LANGUAGE}

Analyze the content and produce SEO gap analysis.

KEYWORD:
{keyword}

PAA:
{paa}

SOURCE CONTENT:
{text[:6000]}

COMPETITOR CONTENT:
{competitors_text[:8000]}

Return JSON structure:

tree:
 topic
 status
 children
"""

    gap = call_llm_json(client, prompt)

    st.session_state["gap"] = gap
    st.session_state["source_text"] = text


# -----------------------
# show results
# -----------------------

if "gap" in st.session_state:

    gap_data = st.session_state["gap"]

    tree = gap_data.get("tree", [])

    st.subheader("Content Gap Structure")

    print_tree(tree)

    st.divider()

    if st.button("Generate Optimized Content"):

        prompt = f"""
Write an SEO optimized article.

Language: {LANGUAGE}

Keyword:
{keyword}

H1 must stay:
{h1}

Base content:
{st.session_state['source_text'][:8000]}
"""

        optimized = call_llm_text(client, prompt)

        st.session_state["optimized"] = optimized

        st.write(optimized)


# -----------------------
# export
# -----------------------

if "optimized" in st.session_state:

    st.download_button(
        label="Export TXT",
        data=st.session_state["optimized"],
        file_name="seo_optimized_content.txt",
        mime="text/plain"
    )
