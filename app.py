import json
import re
from io import BytesIO
from urllib.parse import urlparse

import requests
import streamlit as st
from bs4 import BeautifulSoup
import networkx as nx
import plotly.graph_objects as go
from openai import OpenAI


st.set_page_config(page_title="SEO Gap Analyzer & Optimizer", layout="wide")

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

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def fetch_page(url: str):
    try:
        r = requests.get(
            url,
            timeout=25,
            headers={"User-Agent": USER_AGENT},
        )
        r.raise_for_status()

        html = r.text
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style", "noscript", "iframe", "svg"]):
            tag.decompose()

        text = clean_text(soup.get_text(" "))
        return html, text[:30000]

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
# serp
# -----------------------

def get_serp(keyword: str, api_key: str, hl: str, gl: str, num_results: int):
    url = "https://google.serper.dev/search"

    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }

    payload = {
        "q": keyword,
        "hl": hl,
        "gl": gl,
        "num": min(max(num_results, 1), 10),
    }

    r = requests.post(url, json=payload, headers=headers, timeout=30)

    if r.status_code != 200:
        raise Exception(f"Serper error {r.status_code}: {r.text}")

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
                "position": item.get("position", ""),
            }
        )

    return results[:num_results]


def get_paa(keyword: str, api_key: str, hl: str, gl: str):
    url = "https://serpapi.com/search.json"

    params = {
        "engine": "google",
        "q": keyword,
        "hl": hl,
        "gl": gl,
        "api_key": api_key,
    }

    r = requests.get(url, params=params, timeout=30)

    if r.status_code != 200:
        raise Exception(f"SerpAPI error {r.status_code}: {r.text}")

    data = r.json()
    questions = []

    for q in data.get("related_questions", []):
        question = q.get("question")
        if question:
            questions.append(question)

    return questions[:12]


# -----------------------
# graph
# -----------------------

def build_tree_graph(tree):
    G = nx.DiGraph()
    root = "Gap Map"
    G.add_node(root)

    for macro in tree:
        macro_label = f"{macro.get('topic', 'Topic')} [{macro.get('status', 'unknown')}]"
        G.add_edge(root, macro_label)

        for child in macro.get("children", []):
            child_label = f"{child.get('topic', 'Subtopic')} [{child.get('status', 'unknown')}]"
            G.add_edge(macro_label, child_label)

            for grandchild in child.get("children", []):
                grand_label = f"{grandchild.get('topic', 'Node')} [{grandchild.get('status', 'unknown')}]"
                G.add_edge(child_label, grand_label)

    pos = nx.spring_layout(G, seed=42, k=1.2)

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
    texts = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        texts.append(node)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=texts,
        textposition="top center",
        hoverinfo="text",
        marker=dict(size=18),
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20),
        height=700,
    )
    return fig


# -----------------------
# openai helpers
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
        return {"tree": [], "raw_output": str(e)}


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

st.sidebar.subheader("Search Settings")
LANGUAGE = st.sidebar.selectbox("Language (hl / output)", ["it", "en", "fr", "es", "de"], index=0)
COUNTRY = st.sidebar.selectbox("Country (gl)", ["it", "us", "uk", "fr", "de", "es"], index=0)

client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None


# -----------------------
# session state
# -----------------------

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

if "optimization_done" not in st.session_state:
    st.session_state.optimization_done = False

if "gap_data" not in st.session_state:
    st.session_state.gap_data = None

if "competitors" not in st.session_state:
    st.session_state.competitors = []


# -----------------------
# UI
# -----------------------

st.title("SEO Content Gap Analyzer & Optimizer")

keyword = st.text_input("Keyword")
source_url = st.text_input("URL articolo")
num_competitors = st.number_input("Numero competitor da analizzare", min_value=1, max_value=10, value=5)


if st.button("Esegui Analisi"):

    serp = get_serp(keyword, SERPER_KEY, LANGUAGE, COUNTRY, int(num_competitors))
    paa = get_paa(keyword, SERPAPI_KEY, LANGUAGE, COUNTRY)

    source_html, source_text = fetch_page(source_url)
    source_meta = extract_metadata(source_html)

    enriched_competitors = []

    for c in serp:
        c_html, c_text = fetch_page(c["link"])
        c_meta = extract_metadata(c_html)

        enriched_competitors.append(
            {
                "position": c.get("position"),
                "title": c.get("title"),
                "link": c.get("link"),
                "html_title": c_meta.get("title"),
                "h1": c_meta.get("h1"),
                "meta": c_meta.get("meta"),
                "text": c_text,
            }
        )

    # SALVATAGGIO COMPETITOR
    st.session_state.competitors = enriched_competitors

    prompt = f"""
    Analyze SEO gap for keyword {keyword}.
    """

    gap_data = call_llm_json(client, prompt)

    st.session_state.analysis_done = True
    st.session_state.gap_data = gap_data


# -----------------------
# RESULTS
# -----------------------

if st.session_state.analysis_done:

    gap_data = st.session_state.gap_data
    tree = gap_data.get("tree", [])

    fig = build_tree_graph(tree)

    st.subheader("Grafico ad albero")
    st.plotly_chart(fig, use_container_width=True)

    # NUOVA SEZIONE: COMPETITOR
    st.subheader("Competitor analizzati")

    competitors_list = st.session_state.get("competitors", [])

    for i, comp in enumerate(competitors_list, start=1):
        st.write(
            f"{i}. {comp.get('html_title') or comp.get('title')} — {comp.get('link')}"
        )
st.divider()
st.subheader("Ottimizzazione del contenuto")

proceed = st.radio(
    "Vuoi procedere con l'ottimizzazione del contenuto originale?",
    ["No", "Sì"],
    horizontal=True,
)

if proceed == "Sì":

    if st.button("Genera contenuto ottimizzato"):

        optimization_prompt = f"""
You are a senior SEO copywriter.

Write in this language: {LANGUAGE}

Rewrite and optimize the original article.

Requirements:
- preserve existing useful information
- integrate missing information from gap analysis
- integrate relevant PAA
- improve semantic coverage
- respect SEO best practices
- avoid keyword stuffing

Return EXACTLY in this format:

TITLE TAG: ...
META DESCRIPTION: ...
ARTICLE HTML:
...

KEYWORD:
{keyword}

SOURCE URL:
{source_url}

GAP ANALYSIS:
{json.dumps(st.session_state.gap_data, ensure_ascii=False)}

ORIGINAL CONTENT:
{source_text[:12000]}
"""

        optimized = call_llm_text(client, optimization_prompt)

        st.session_state.optimized_output = optimized

        st.subheader("Output HTML")
        st.code(optimized, language="html")

        st.download_button(
            label="Scarica TXT",
            data=optimized,
            file_name="seo_optimized_content.txt",
            mime="text/plain",
        )
