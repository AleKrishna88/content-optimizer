import json
import re
from urllib.parse import urlparse

import requests
import streamlit as st
from bs4 import BeautifulSoup

import networkx as nx
import plotly.graph_objects as go

from openai import OpenAI


st.set_page_config(page_title="SEO Content Optimizer", layout="wide")

USER_AGENT = "Mozilla/5.0"

BLOCKED_DOMAINS = {
    "youtube.com",
    "instagram.com",
    "facebook.com",
    "linkedin.com",
    "tiktok.com",
}


# -----------------------
# helper
# -----------------------

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def fetch_page(url: str):
    try:
        r = requests.get(
            url,
            timeout=20,
            headers={"User-Agent": USER_AGENT},
        )
        r.raise_for_status()

        html = r.text
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text = clean_text(soup.get_text(" "))
        return html, text[:20000]

    except Exception:
        return "", ""


def extract_metadata(html: str):
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
        "meta": meta,
    }


# -----------------------
# SERP
# -----------------------

def get_serp(keyword: str, api_key: str):
    url = "https://google.serper.dev/search"

    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }

    payload = {
        "q": keyword,
        "hl": HL,
        "gl": GL,
        "num": 10,
    }

    r = requests.post(url, json=payload, headers=headers, timeout=30)
    r.raise_for_status()

    data = r.json()
    results = []

    for item in data.get("organic", []):
        link = item.get("link")
        if not link:
            continue

        domain = urlparse(link).netloc.lower()
        if any(d in domain for d in BLOCKED_DOMAINS):
            continue

        results.append(
            {
                "title": item.get("title", ""),
                "link": link,
                "snippet": item.get("snippet", ""),
            }
        )

    return results[:5]


def get_paa(keyword: str, api_key: str):
    url = "https://serpapi.com/search.json"

    params = {
        "engine": "google",
        "q": keyword,
        "hl": HL,
        "gl": GL,
        "api_key": api_key,
    }

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()

    data = r.json()
    questions = []

    for q in data.get("related_questions", []):
        question = q.get("question")
        if question:
            questions.append(question)

    return questions[:10]


# -----------------------
# graph
# -----------------------

def build_tree_graph(tree):
    G = nx.DiGraph()
    G.add_node("Gap Analysis")

    for macro in tree:
        macro_topic = macro.get("topic", "Topic")
        macro_status = macro.get("status", "unknown")
        macro_node = f"{macro_topic} ({macro_status})"
        G.add_edge("Gap Analysis", macro_node)

        for child in macro.get("children", []):
            child_topic = child.get("topic", "Subtopic")
            child_status = child.get("status", "unknown")
            child_node = f"{child_topic} ({child_status})"
            G.add_edge(macro_node, child_node)

    pos = nx.spring_layout(G, seed=42)

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
        mode="lines",
        hoverinfo="none",
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
        mode="markers+text",
        text=text,
        textposition="top center",
        hoverinfo="text",
        marker=dict(size=20),
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=20, b=20))

    return fig


# -----------------------
# OPENAI
# -----------------------

def extract_json_from_text(text: str):
    text = (text or "").strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    return {"tree": [], "raw_output": text}


def call_llm_json(client: OpenAI, prompt: str):
    try:
        response = client.responses.create(
            model="gpt-5.4",
            input=prompt,
        )

        text = getattr(response, "output_text", "") or ""
        return extract_json_from_text(text)

    except Exception as e:
        st.error(f"OpenAI error: {e}")
        return {"tree": []}


def call_llm_text(client: OpenAI, prompt: str):
    try:
        response = client.responses.create(
            model="gpt-5.4",
            input=prompt,
        )
        return getattr(response, "output_text", "") or ""

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

HL = st.sidebar.selectbox("Language (hl)", ["it", "en", "fr", "es", "de"])
GL = st.sidebar.selectbox("Country (gl)", ["it", "us", "uk", "fr", "de", "es"])

client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None


# -----------------------
# session state
# -----------------------

if "gap" not in st.session_state:
    st.session_state.gap = None

if "source_text" not in st.session_state:
    st.session_state.source_text = ""

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False


# -----------------------
# UI
# -----------------------

st.title("SEO Content Gap Analyzer")

keyword = st.text_input("Keyword")
h1 = st.text_input("Titolo H1")
url = st.text_input("URL da analizzare")


if st.button("Esegui Analisi"):
    if not OPENAI_KEY or not SERPER_KEY or not SERPAPI_KEY:
        st.error("Inserisci tutte le API key nella sidebar.")
        st.stop()

    if not keyword or not h1 or not url:
        st.error("Keyword, H1 e URL sono obbligatori.")
        st.stop()

    try:
        serp = get_serp(keyword, SERPER_KEY)
        paa = get_paa(keyword, SERPAPI_KEY)
        html, text = fetch_page(url)
        meta = extract_metadata(html)

        if not text:
            st.error("Non sono riuscito a estrarre il contenuto dalla URL indicata.")
            st.stop()

        competitors_text = ""
        competitors_meta = []

        for c in serp:
            c_html, c_text = fetch_page(c["link"])
            c_meta = extract_metadata(c_html) if c_html else {"title": "", "h1": "", "meta": ""}
            competitors_meta.append(
                {
                    "url": c["link"],
                    "title": c.get("title", ""),
                    "html_title": c_meta.get("title", ""),
                    "h1": c_meta.get("h1", ""),
                    "meta": c_meta.get("meta", ""),
                }
            )
            competitors_text += f"\nURL: {c['link']}\nTITLE: {c_meta.get('title', '')}\nH1: {c_meta.get('h1', '')}\nCONTENT:\n{c_text[:2000]}\n"

        prompt = f"""
Sei un SEO strategist senior.

La lingua di output deve essere: {HL}

Analizza il contenuto sorgente confrontandolo con competitor e PAA.
Rispondi SOLO con JSON valido.

La struttura deve essere ESATTAMENTE:
{{
  "summary": {{
    "search_intent": "string",
    "overall_verdict": "string",
    "priority_actions": ["string"]
  }},
  "tree": [
    {{
      "topic": "string",
      "status": "covered|partial|missing",
      "children": [
        {{
          "topic": "string",
          "status": "covered|partial|missing"
        }}
      ]
    }}
  ],
  "missing_paa": ["string"],
  "recommended_headings": ["string"]
}}

KEYWORD:
{keyword}

H1 DICHIARATO:
{h1}

URL SORGENTE:
{url}

METADATI SORGENTE:
{json.dumps(meta, ensure_ascii=False)}

PAA:
{json.dumps(paa, ensure_ascii=False)}

CONTENUTO SORGENTE:
{text[:6000]}

COMPETITOR:
{competitors_text[:8000]}
"""

        gap = call_llm_json(client, prompt)

        st.session_state.gap = gap
        st.session_state.source_text = text
        st.session_state.analysis_done = True

    except Exception as e:
        st.error(f"Errore durante l'analisi: {e}")
