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
# prompts
# -----------------------

def build_gap_prompt(
    language: str,
    keyword: str,
    source_url: str,
    source_meta: dict,
    source_text: str,
    competitors: list,
    paa: list,
):
    competitor_blocks = []

    for item in competitors:
        competitor_blocks.append(
            f"""
URL: {item.get('link', '')}
POSITION: {item.get('position', '')}
SERP TITLE: {item.get('title', '')}
HTML TITLE: {item.get('html_title', '')}
H1: {item.get('h1', '')}
META: {item.get('meta', '')}
CONTENT:
{item.get('text', '')[:2500]}
"""
        )

    return f"""
You are a senior SEO strategist.

Output language must be: {language}

Analyze the source article against:
1. aggregated competitor content
2. People Also Ask questions

Return ONLY valid JSON.
Do not include prose before or after the JSON.

Required JSON structure:
{{
  "summary": {{
    "search_intent": "string",
    "overall_verdict": "string",
    "priority_actions": ["string", "string"]
  }},
  "tree": [
    {{
      "topic": "string",
      "status": "covered|partial|missing",
      "children": [
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
      ]
    }}
  ],
  "present_information": [
    {{
      "section": "string",
      "items": ["string", "string"]
    }}
  ],
  "missing_information": [
    {{
      "section": "string",
      "items": ["string", "string"]
    }}
  ],
  "weak_information": [
    {{
      "section": "string",
      "items": ["string", "string"]
    }}
  ],
  "missing_paa": ["string"],
  "recommended_headings": ["string"],
  "entities_and_related_keywords": [
    {{
      "entity": "string",
      "related_keywords": ["string", "string"]
    }}
  ]
}}

KEYWORD:
{keyword}

SOURCE URL:
{source_url}

SOURCE METADATA:
{json.dumps(source_meta, ensure_ascii=False)}

SOURCE CONTENT:
{source_text[:9000]}

PAA:
{json.dumps(paa, ensure_ascii=False)}

COMPETITOR CONTENT:
{"".join(competitor_blocks)[:18000]}
"""


def build_optimization_prompt(
    language: str,
    keyword: str,
    source_url: str,
    source_meta: dict,
    source_text: str,
    gap_data: dict,
):
    return f"""
You are a senior SEO copywriter.

Write in this language: {language}

Task:
Rewrite and optimize the original article for SEO.

Requirements:
- preserve the existing useful information
- integrate missing information from the gap analysis
- integrate relevant missing PAA naturally
- improve completeness, semantic coverage, readability and search intent alignment
- keep the article coherent and non-repetitive
- avoid keyword stuffing
- output must be in HTML
- include one H1, descriptive H2/H3, lists when useful, and an FAQ section if relevant

Return EXACTLY in this format:

TITLE TAG: ...
META DESCRIPTION: ...
ARTICLE HTML:
...

KEYWORD:
{keyword}

SOURCE URL:
{source_url}

SOURCE METADATA:
{json.dumps(source_meta, ensure_ascii=False)}

GAP ANALYSIS:
{json.dumps(gap_data, ensure_ascii=False)}

ORIGINAL CONTENT:
{source_text[:12000]}
"""


def parse_optimized_output(raw_text: str):
    result = {
        "title_tag": "",
        "meta_description": "",
        "article_html": raw_text or "",
    }

    if "TITLE TAG:" in raw_text and "META DESCRIPTION:" in raw_text and "ARTICLE HTML:" in raw_text:
        try:
            after_title = raw_text.split("TITLE TAG:", 1)[1]
            result["title_tag"] = after_title.split("META DESCRIPTION:", 1)[0].strip()

            after_meta = after_title.split("META DESCRIPTION:", 1)[1]
            result["meta_description"] = after_meta.split("ARTICLE HTML:", 1)[0].strip()

            result["article_html"] = after_meta.split("ARTICLE HTML:", 1)[1].strip()
        except Exception:
            pass

    return result


def render_hierarchical_list(title: str, sections: list):
    st.subheader(title)
    if not sections:
        st.write("-")
        return

    for section in sections:
        st.write(f"**{section.get('section', 'Section')}**")
        for item in section.get("items", []):
            st.write(f"- {item}")


def make_txt_export(parsed_output: dict):
    content = []
    content.append(f"TITLE TAG: {parsed_output.get('title_tag', '')}")
    content.append(f"META DESCRIPTION: {parsed_output.get('meta_description', '')}")
    content.append("")
    content.append("ARTICLE HTML:")
    content.append(parsed_output.get("article_html", ""))

    txt = "\n".join(content)
    return BytesIO(txt.encode("utf-8"))


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

if "source_text" not in st.session_state:
    st.session_state.source_text = ""

if "source_meta" not in st.session_state:
    st.session_state.source_meta = {}

if "source_url" not in st.session_state:
    st.session_state.source_url = ""

if "keyword" not in st.session_state:
    st.session_state.keyword = ""

if "optimized_output" not in st.session_state:
    st.session_state.optimized_output = None


# -----------------------
# ui
# -----------------------

st.title("SEO Content Gap Analyzer & Optimizer")

keyword = st.text_input("Keyword")
source_url = st.text_input("URL articolo")
num_competitors = st.number_input("Numero competitor da analizzare", min_value=1, max_value=10, value=5, step=1)

if st.button("Esegui Analisi"):
    if not OPENAI_KEY or not SERPER_KEY or not SERPAPI_KEY:
        st.error("Inserisci tutte le API key nella sidebar.")
        st.stop()

    if not keyword or not source_url:
        st.error("Keyword e URL articolo sono obbligatori.")
        st.stop()

    try:
        serp = get_serp(keyword, SERPER_KEY, LANGUAGE, COUNTRY, int(num_competitors))
        paa = get_paa(keyword, SERPAPI_KEY, LANGUAGE, COUNTRY)
        source_html, source_text = fetch_page(source_url)
        source_meta = extract_metadata(source_html)

        if not source_text:
            st.error("Non sono riuscito a estrarre il contenuto dalla URL proprietaria.")
            st.stop()

        enriched_competitors = []
        for c in serp:
            c_html, c_text = fetch_page(c["link"])
            c_meta = extract_metadata(c_html) if c_html else {"title": "", "h1": "", "meta": ""}
            enriched_competitors.append(
                {
                    "position": c.get("position", ""),
                    "title": c.get("title", ""),
                    "link": c.get("link", ""),
                    "snippet": c.get("snippet", ""),
                    "html_title": c_meta.get("title", ""),
                    "h1": c_meta.get("h1", ""),
                    "meta": c_meta.get("meta", ""),
                    "text": c_text,
                }
            )

        prompt = build_gap_prompt(
            language=LANGUAGE,
            keyword=keyword,
            source_url=source_url,
            source_meta=source_meta,
            source_text=source_text,
            competitors=enriched_competitors,
            paa=paa,
        )

        gap_data = call_llm_json(client, prompt)

        st.session_state.analysis_done = True
        st.session_state.optimization_done = False
        st.session_state.gap_data = gap_data
        st.session_state.source_text = source_text
        st.session_state.source_meta = source_meta
        st.session_state.source_url = source_url
        st.session_state.keyword = keyword
        st.session_state.optimized_output = None

    except Exception as e:
        st.error(f"Errore durante l'analisi: {e}")


if st.session_state.analysis_done and st.session_state.gap_data:
    gap_data = st.session_state.gap_data

    if not isinstance(gap_data, dict):
        gap_data = {"tree": [], "raw_output": str(gap_data)}

    summary = gap_data.get("summary", {})
    tree = gap_data.get("tree", [])
    present_information = gap_data.get("present_information", [])
    weak_information = gap_data.get("weak_information", [])
    missing_information = gap_data.get("missing_information", [])
    missing_paa = gap_data.get("missing_paa", [])
    recommended_headings = gap_data.get("recommended_headings", [])
    entities_keywords = gap_data.get("entities_and_related_keywords", [])

    st.subheader("Sintesi")
    if summary:
        st.write(f"**Intento di ricerca:** {summary.get('search_intent', '-')}")
        st.write(f"**Verdetto:** {summary.get('overall_verdict', '-')}")
        for item in summary.get("priority_actions", []):
            st.write(f"- {item}")

    st.subheader("Grafico ad albero")
    fig = build_tree_graph(tree)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Entità e keyword correlate")
    if entities_keywords:
        for entity in entities_keywords:
            st.write(f"**{entity.get('entity', 'Entity')}**")
            for kw in entity.get("related_keywords", []):
                st.write(f"- {kw}")
    else:
        st.write("-")

    render_hierarchical_list("Informazioni presenti", present_information)
    render_hierarchical_list("Informazioni deboli / da rafforzare", weak_information)
    render_hierarchical_list("Informazioni mancanti", missing_information)

    if missing_paa:
        st.subheader("PAA mancanti")
        for item in missing_paa:
            st.write(f"- {item}")

    if recommended_headings:
        st.subheader("Heading consigliati")
        for item in recommended_headings:
            st.write(f"- {item}")

    if "raw_output" in gap_data:
        st.subheader("Debug output modello")
        st.code(gap_data["raw_output"])

    st.divider()
    st.subheader("Ottimizzazione del contenuto")

    proceed = st.radio(
        "Vuoi procedere con l'ottimizzazione del contenuto originale?",
        ["No", "Sì"],
        horizontal=True,
    )

    if proceed == "Sì" and st.button("Genera contenuto ottimizzato"):
        prompt = build_optimization_prompt(
            language=LANGUAGE,
            keyword=st.session_state.keyword,
            source_url=st.session_state.source_url,
            source_meta=st.session_state.source_meta,
            source_text=st.session_state.source_text,
            gap_data=gap_data,
        )

        optimized_raw = call_llm_text(client, prompt)
        parsed_output = parse_optimized_output(optimized_raw)

        st.session_state.optimization_done = True
        st.session_state.optimized_output = parsed_output

if st.session_state.optimization_done and st.session_state.optimized_output:
    output = st.session_state.optimized_output

    st.subheader("Output ottimizzato")
    st.write(f"**Title Tag:** {output.get('title_tag', '')}")
    st.write(f"**Meta Description:** {output.get('meta_description', '')}")

    st.subheader("HTML")
    st.code(output.get("article_html", ""), language="html")

    txt_buffer = make_txt_export(output)
    st.download_button(
        label="Scarica TXT",
        data=txt_buffer,
        file_name="seo_optimized_content.txt",
        mime="text/plain",
    )
