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
# helper
# -----------------------

def clean_text(text):
    return re.sub(r"\s+", " ", text or "").strip()


def fetch_page(url):
    try:

        r = requests.get(
            url,
            timeout=25,
            headers={"User-Agent": USER_AGENT},
        )

        html = r.text
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text = clean_text(soup.get_text(" "))

        return html, text[:30000]

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

def get_serp(keyword, api_key, hl, gl, num):

    url = "https://google.serper.dev/search"

    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json"
    }

    payload = {
        "q": keyword,
        "hl": hl,
        "gl": gl,
        "num": num
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
            "snippet": item.get("snippet"),
            "position": item.get("position")
        })

    return results[:num]


def get_paa(keyword, api_key, hl, gl):

    url = "https://serpapi.com/search.json"

    params = {
        "engine": "google",
        "q": keyword,
        "hl": hl,
        "gl": gl,
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
# graph
# -----------------------

def build_tree_graph(tree):

    G = nx.DiGraph()
    root = "Gap Analysis"

    G.add_node(root)

    for macro in tree:

        macro_node = f"{macro.get('topic')} ({macro.get('status')})"
        G.add_edge(root, macro_node)

        for child in macro.get("children", []):

            child_node = f"{child.get('topic')} ({child.get('status')})"
            G.add_edge(macro_node, child_node)

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
        mode="lines"
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
        text=texts,
        mode="markers+text",
        textposition="top center",
        marker=dict(size=18)
    )

    fig = go.Figure(data=[edge_trace, node_trace])

    fig.update_layout(
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20)
    )

    return fig


# -----------------------
# OPENAI
# -----------------------

def extract_json_from_text(text):

    try:
        return json.loads(text)
    except:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass

    return {"tree": []}


def call_llm_json(client, prompt):

    response = client.responses.create(
        model="gpt-5.4",
        input=prompt
    )

    text = getattr(response, "output_text", "")

    return extract_json_from_text(text)


def call_llm_text(client, prompt):

    response = client.responses.create(
        model="gpt-5.4",
        input=prompt
    )

    return getattr(response, "output_text", "")


# -----------------------
# sidebar
# -----------------------

st.sidebar.title("API Keys")

OPENAI_KEY = st.sidebar.text_input("OpenAI", type="password")
SERPER_KEY = st.sidebar.text_input("Serper", type="password")
SERPAPI_KEY = st.sidebar.text_input("SerpAPI", type="password")

st.sidebar.subheader("Search settings")

LANGUAGE = st.sidebar.selectbox("Language", ["it", "en", "fr", "es", "de"])
COUNTRY = st.sidebar.selectbox("Country", ["it", "us", "uk", "fr", "de", "es"])

client = OpenAI(api_key=OPENAI_KEY)


# -----------------------
# session state
# -----------------------

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

if "competitors" not in st.session_state:
    st.session_state.competitors = []

if "gap_data" not in st.session_state:
    st.session_state.gap_data = None


# -----------------------
# UI
# -----------------------

st.title("SEO Content Gap Analyzer")

keyword = st.text_input("Keyword")

source_url = st.text_input("URL articolo")

num_competitors = st.number_input(
    "Numero competitor",
    min_value=1,
    max_value=10,
    value=5
)


if st.button("Esegui Analisi"):

    serp = get_serp(keyword, SERPER_KEY, LANGUAGE, COUNTRY, num_competitors)

    paa = get_paa(keyword, SERPAPI_KEY, LANGUAGE, COUNTRY)

    html, source_text = fetch_page(source_url)

    enriched_competitors = []

    competitors_text = ""

    for c in serp:

        c_html, c_text = fetch_page(c["link"])

        c_meta = extract_metadata(c_html)

        enriched_competitors.append({
            "position": c.get("position"),
            "title": c.get("title"),
            "link": c.get("link"),
            "html_title": c_meta.get("title"),
            "h1": c_meta.get("h1"),
            "meta": c_meta.get("meta"),
            "text": c_text
        })

        competitors_text += c_text[:2000]

    st.session_state.competitors = enriched_competitors

    prompt = f"""
Analyze SEO gap for keyword {keyword}.
"""

    gap = call_llm_json(client, prompt)

    st.session_state.gap_data = gap
    st.session_state.analysis_done = True


# -----------------------
# RESULTS
# -----------------------

if st.session_state.analysis_done:

    tree = st.session_state.gap_data.get("tree", [])

    st.subheader("Grafico ad albero")

    fig = build_tree_graph(tree)

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Competitor analizzati")

    for i, comp in enumerate(st.session_state.competitors, start=1):

        st.write(
            f"{i}. {comp.get('html_title') or comp.get('title')} — {comp.get('link')}"
        )

    st.divider()

    st.subheader("Ottimizzazione del contenuto")

    proceed = st.radio(
        "Vuoi procedere con l'ottimizzazione?",
        ["No", "Sì"],
        horizontal=True
    )

    if proceed == "Sì":

        if st.button("Genera contenuto ottimizzato"):

            optimization_prompt = f"""
You are a senior SEO copywriter and content strategist.

Your task is to rewrite and optimize the original article for SEO.

LANGUAGE:
{LANGUAGE}

PRIMARY KEYWORD:
{keyword}

SOURCE URL:
{source_url}

IMPORTANT REQUIREMENTS:

The optimized article MUST:

• preserve all useful information already present in the original content  
• integrate the missing information identified during the gap analysis  
• be MORE COMPLETE than the original article  
• NEVER be shorter than the original article  
• expand explanations where competitors provide deeper coverage  

WRITING STYLE:

• maintain the same tone of voice as the original article  
• keep the text fluent, natural and discursive  
• avoid keyword stuffing completely  
• use the keyword and related entities naturally  

FORMATTING RULES:

• structure the article with clear H1, H2 and H3 headings  
• use bullet lists ONLY when they improve clarity  
• use tables ONLY when they genuinely help explain comparisons or structured data  
• avoid unnecessary lists or formatting  

SEO REQUIREMENTS:

• align the article with the search intent behind the keyword  
• integrate entities and related topics from the gap analysis  
• cover missing PAA questions naturally when relevant  
• improve semantic coverage of the topic  

OUTPUT FORMAT:

Return EXACTLY in this format:

TITLE TAG: ...
META DESCRIPTION: ...
ARTICLE HTML:
...

CONTEXT DATA

GAP ANALYSIS:
{json.dumps(st.session_state.gap_data, ensure_ascii=False)}

ORIGINAL CONTENT:
{source_text[:12000]}
"""

            optimized = call_llm_text(client, optimization_prompt)

            st.subheader("Output HTML")

            st.code(optimized, language="html")

            st.download_button(
                label="Scarica TXT",
                data=optimized,
                file_name="seo_content.txt",
                mime="text/plain"
            )
