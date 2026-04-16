import json
import re
from io import BytesIO
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

import requests
import streamlit as st
from bs4 import BeautifulSoup
import networkx as nx
import plotly.graph_objects as go
from openai import OpenAI

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="SEO Gap Analyzer & Optimizer", layout="wide")

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

BLOCKED_DOMAINS = {
    "youtube.com",
    "youtu.be",
    "instagram.com",
    "facebook.com",
    "tiktok.com",
    "pinterest.com",
    "linkedin.com",
}

DEFAULT_MODEL = "gpt-5"

# =========================
# SESSION STATE
# =========================

def init_state() -> None:
    defaults = {
        "analysis_done": False,
        "optimization_done": False,
        "serp_data": None,
        "gap_data": None,
        "optimized_output": None,
        "source_page": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()

# =========================
# HELPERS
# =========================

def clean_text(text: str) -> str:
    return re.sub(r"\\s+", " ", text or "").strip()


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_page(url: str) -> Tuple[str, str]:
    try:
        response = requests.get(
            url,
            timeout=20,
            headers={"User-Agent": USER_AGENT},
        )
        response.raise_for_status()
        html = response.text
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg", "iframe"]):
            tag.decompose()
        text = clean_text(soup.get_text(" "))
        return html, text[:25000]
    except Exception:
        return "", ""


@st.cache_data(show_spinner=False, ttl=3600)
def extract_metadata(html: str) -> Dict[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""
    h1_tag = soup.find("h1")
    meta_tag = soup.find("meta", attrs={"name": "description"})
    return {
        "title": title,
        "h1": h1_tag.get_text(strip=True) if h1_tag else "",
        "meta_description": meta_tag.get("content", "").strip() if meta_tag else "",
    }


@st.cache_data(show_spinner=False, ttl=3600)
def get_competitors(keyword: str, num_results: int, serper_key: str, hl: str, gl: str) -> List[Dict[str, Any]]:
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": serper_key,
        "Content-Type": "application/json",
    }

    competitors: List[Dict[str, Any]] = []
    seen_urls = set()
    start = 0

    while len(competitors) < num_results and start <= 90:
        payload = {
            "q": keyword,
            "gl": gl,
            "hl": hl,
            "num": 10,
            "start": start,
        }
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        organic = data.get("organic", [])
        if not organic:
            break

        for item in organic:
            link = clean_text(item.get("link", ""))
            if not link:
                continue

            netloc = urlparse(link).netloc.lower()
            if any(domain in netloc for domain in BLOCKED_DOMAINS):
                continue
            if link.rstrip("/") in seen_urls:
                continue

            seen_urls.add(link.rstrip("/"))
            competitors.append(
                {
                    "position": item.get("position"),
                    "title": item.get("title", ""),
                    "link": link.rstrip("/"),
                    "snippet": item.get("snippet", ""),
                }
            )
            if len(competitors) >= num_results:
                break
        start += 10

    return competitors[:num_results]


@st.cache_data(show_spinner=False, ttl=3600)
def get_people_also_ask(keyword: str, serpapi_key: str, hl: str, gl: str) -> List[str]:
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google",
        "q": keyword,
        "hl": hl,
        "gl": gl,
        "api_key": serpapi_key,
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    questions = []
    seen = set()
    for item in data.get("related_questions", []):
        question = clean_text(item.get("question", ""))
        if question and question not in seen:
            seen.add(question)
            questions.append(question)
    return questions[:12]


def enrich_urls(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    enriched = []
    for item in items:
        html, text = fetch_page(item["link"])
        metadata = extract_metadata(html) if html else {"title": "", "h1": "", "meta_description": ""}
        enriched.append(
            {
                **item,
                "html_title": metadata["title"],
                "h1": metadata["h1"],
                "meta_description": metadata["meta_description"],
                "text": text,
            }
        )
    return enriched


def build_openai_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def call_llm_json(client: OpenAI, model: str, prompt: str) -> Dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Sei un SEO strategist senior. Rispondi sempre con JSON valido."},
            {"role": "user", "content": prompt},
        ],
    )
    content = response.choices[0].message.content or "{}"
    return json.loads(content)


def call_llm_text(client: OpenAI, model: str, prompt: str) -> str:
    response = client.chat.completions.create(
        model=model,
        temperature=0.5,
        messages=[
            {"role": "system", "content": "Sei un SEO copywriter senior."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content or ""


def build_tree_graph(gap_tree):

    G = nx.DiGraph()
    G.add_node("Gap Analysis")

    for macro in gap_tree:
        macro_name = f"{macro['topic']} ({macro['status']})"
        G.add_edge("Gap Analysis", macro_name)

        for child in macro.get("children", []):
            child_name = f"{child['topic']} ({child['status']})"
            G.add_edge(macro_name, child_name)

    pos = nx.spring_layout(G)

    edge_x = []
    edge_y = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1),
        hoverinfo='none',
        mode='lines'
    )

    node_x = []
    node_y = []
    text = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        text.append(node)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=text,
        textposition="top center",
        hoverinfo='text',
        marker=dict(size=20)
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(showlegend=False)

    return fig

# =========================
# SIDEBAR
# =========================

st.sidebar.title("Configuration")
SERPER_KEY = st.sidebar.text_input("Serper.dev API Key", type="password")
SERPAPI_KEY = st.sidebar.text_input("SerpAPI Key", type="password")
OPENAI_KEY = st.sidebar.text_input("OpenAI API Key", type="password")
MODEL_NAME = st.sidebar.text_input("OpenAI model", value=DEFAULT_MODEL)
COUNTRY = st.sidebar.text_input("Country code", value="it")
LANGUAGE = st.sidebar.text_input("Language code", value="it")
NUM_RESULTS = st.sidebar.slider("Competitor da analizzare", 3, 10, 5)

# =========================
# UI
# =========================

st.title("SEO Gap Analyzer & Optimizer")

keyword = st.text_input("Keyword")
input_h1 = st.text_input("Titolo H1")
source_url = st.text_input("URL da analizzare")

if st.button("Esegui analisi"):

    competitors = get_competitors(keyword, NUM_RESULTS, SERPER_KEY, LANGUAGE, COUNTRY)
    paa = get_people_also_ask(keyword, SERPAPI_KEY, LANGUAGE, COUNTRY)

    source_html, source_text = fetch_page(source_url)
    source_meta = extract_metadata(source_html)

    enriched_competitors = enrich_urls(competitors)

    client = build_openai_client(OPENAI_KEY)

    prompt = f"""
Analizza il contenuto e fai una gap analysis.

KEYWORD: {keyword}
H1: {input_h1}

PEOPLE ALSO ASK:
{paa}

SOURCE CONTENT:
{source_text[:8000]}
"""

    gap_data = call_llm_json(client, MODEL_NAME, prompt)

    st.session_state.gap_data = gap_data

if "gap_data" in st.session_state:

    st.subheader("Grafo Gap Analysis")

    fig = build_tree_graph(st.session_state.gap_data.get("tree", []))

    st.plotly_chart(fig, use_container_width=True)

    if st.button("Genera contenuto ottimizzato"):

        client = build_openai_client(OPENAI_KEY)

        prompt = f"""
Ottimizza questo contenuto per SEO.

KEYWORD: {keyword}
H1: {input_h1}

CONTENT:
{source_text[:10000]}
"""

        optimized = call_llm_text(client, MODEL_NAME, prompt)

        st.code(optimized)
