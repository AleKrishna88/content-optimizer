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

def get_serp(keyword, api_key):

    url = "https://google.serper.dev/search"

    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json"
    }

    payload = {
        "q": keyword,
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


def get_paa(keyword, api_key):

    url = "https://serpapi.com/search.json"

    params = {
        "engine": "google",
        "q": keyword,
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

    G.add_node("Gap Analysis")

    for macro in tree:

        macro_node = f"{macro.get('topic')} ({macro.get('status')})"

        G.add_edge("Gap Analysis", macro_node)

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
        marker=dict(size=20)
    )

    fig = go.Figure(data=[edge_trace, node_trace])

    fig.update_layout(showlegend=False)

    return fig


# -----------------------
# openai
# -----------------------

def call_llm_json(client, prompt):

    response = client.chat.completions.create(

        model="gpt-5",

        temperature=0.2,

        response_format={"type": "json_object"},

        messages=[
            {"role": "system", "content": "Rispondi solo con JSON valido"},
            {"role": "user", "content": prompt}
        ]
    )

    content = response.choices[0].message.content

    try:
        return json.loads(content)

    except:

        return {
            "tree": []
        }


def call_llm_text(client, prompt):

    response = client.chat.completions.create(

        model="gpt-5",

        temperature=0.6,

        messages=[
            {"role": "system", "content": "Sei un SEO copywriter"},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content


# -----------------------
# sidebar
# -----------------------

st.sidebar.title("API")

OPENAI_KEY = st.sidebar.text_input("OpenAI", type="password")

SERPER_KEY = st.sidebar.text_input("Serper", type="password")

SERPAPI_KEY = st.sidebar.text_input("SerpAPI", type="password")


client = OpenAI(api_key=OPENAI_KEY)


# -----------------------
# UI
# -----------------------

st.title("SEO Content Optimizer")


keyword = st.text_input("Keyword")

h1 = st.text_input("H1")

url = st.text_input("URL")


if st.button("Analyze"):

    serp = get_serp(keyword, SERPER_KEY)

    paa = get_paa(keyword, SERPAPI_KEY)

    html, text = fetch_page(url)

    meta = extract_metadata(html)

    competitors_text = ""

    for c in serp:

        _, ctext = fetch_page(c["link"])

        competitors_text += ctext[:4000]


    prompt = f"""

Analizza il contenuto e fai gap analysis.

KEYWORD:
{keyword}

PAA:
{paa}

CONTENT:
{text[:8000]}

COMPETITOR:
{competitors_text[:12000]}

Rispondi JSON con:

tree:
topic
status
children
"""

    gap = call_llm_json(client, prompt)

    st.session_state.gap = gap


# -----------------------
# show graph
# -----------------------

if "gap" in st.session_state:

    gap_data = st.session_state.gap

    tree = []

    if isinstance(gap_data, dict):

        tree = gap_data.get("tree", [])

    fig = build_tree_graph(tree)

    st.plotly_chart(fig, use_container_width=True)


    if st.button("Generate optimized content"):

        prompt = f"""

Scrivi contenuto SEO per keyword:

{keyword}

Mantieni H1:

{h1}

Content:

{text[:10000]}
"""

        optimized = call_llm_text(client, prompt)

        st.write(optimized)
