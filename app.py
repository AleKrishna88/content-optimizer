import json
import re
from io import BytesIO
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

import requests
import streamlit as st
from bs4 import BeautifulSoup
import graphviz
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
    return re.sub(r"\s+", " ", text or "").strip()


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
            {
                "role": "system",
                "content": "Sei un SEO strategist senior. Rispondi sempre con JSON valido.",
            },
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
            {
                "role": "system",
                "content": "Sei un SEO copywriter senior specializzato in ottimizzazione on-page.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content or ""


def build_gap_prompt(keyword: str, input_h1: str, source_url: str, source_page: Dict[str, Any], competitors: List[Dict[str, Any]], paa: List[str]) -> str:
    competitor_block = "\n\n".join(
        [
            (
                f"[COMPETITOR {idx}]\n"
                f"URL: {item['link']}\n"
                f"SERP TITLE: {item.get('title', '')}\n"
                f"HTML TITLE: {item.get('html_title', '')}\n"
                f"H1: {item.get('h1', '')}\n"
                f"META DESCRIPTION: {item.get('meta_description', '')}\n"
                f"CONTENT: {item.get('text', '')[:7000]}"
            )
            for idx, item in enumerate(competitors, start=1)
        ]
    )

    paa_block = "\n".join([f"- {question}" for question in paa]) or "- Nessuna PAA disponibile"

    return f"""
Analizza il contenuto sorgente e confrontalo con competitor e People Also Ask.

OBIETTIVO:
- svolgere una gap analysis SEO semantica
- individuare topic coperti, topic mancanti, topic deboli, FAQ mancanti, intenti non coperti, evidenze/utilità mancanti
- restituire una struttura ad albero che poi verrà visualizzata in Streamlit

INPUT:
KEYWORD: {keyword}
H1 DICHIARATO DALL'UTENTE: {input_h1}
URL SORGENTE: {source_url}

CONTENUTO SORGENTE:
TITLE: {source_page.get('title', '')}
H1: {source_page.get('h1', '')}
META DESCRIPTION: {source_page.get('meta_description', '')}
CONTENT: {source_page.get('text', '')[:9000]}

PEOPLE ALSO ASK:
{paa_block}

COMPETITOR:
{competitor_block}

RESTITUISCI SOLO JSON con questa struttura:
{{
  "summary": {{
    "search_intent": "...",
    "overall_verdict": "...",
    "priority_actions": ["...", "...", "..."]
  }},
  "tree": [
    {{
      "topic": "Macro area",
      "status": "covered|partial|missing",
      "reason": "...",
      "children": [
        {{
          "topic": "Subtopic",
          "status": "covered|partial|missing",
          "reason": "...",
          "evidence": ["...", "..."],
          "suggested_heading": "..."
        }}
      ]
    }}
  ],
  "missing_paa": ["..."],
  "missing_entities": ["..."],
  "recommended_headings": ["..."],
  "content_brief": {{
    "must_have_sections": ["..."],
    "faq_to_add": ["..."],
    "internal_notes": ["..."]
  }}
}}

REGOLE:
- Massimo 6 macro aree.
- Ogni macro area deve avere da 2 a 6 figli.
- Lo status deve essere coerente con il confronto.
- I suggested_heading devono essere heading SEO realmente utili.
- I missing_paa devono includere solo domande davvero non coperte o coperte male.
"""


def build_optimization_prompt(keyword: str, input_h1: str, source_url: str, source_page: Dict[str, Any], gap_data: Dict[str, Any]) -> str:
    return f"""
Ottimizza il contenuto della pagina in ottica SEO usando la gap analysis fornita.

INPUT:
KEYWORD TARGET: {keyword}
H1 OBBLIGATORIO: {input_h1}
URL: {source_url}

CONTENUTO ATTUALE:
TITLE: {source_page.get('title', '')}
H1: {source_page.get('h1', '')}
META DESCRIPTION: {source_page.get('meta_description', '')}
CONTENT: {source_page.get('text', '')[:12000]}

GAP ANALYSIS JSON:
{json.dumps(gap_data, ensure_ascii=False, indent=2)}

RESTITUISCI ESATTAMENTE nel seguente formato:
TITLE TAG: ...
META DESCRIPTION: ...
OPTIMIZED ARTICLE HTML: ...

VINCOLI:
- Non cambiare l'H1: deve essere esattamente {input_h1}
- Mantieni il focus sulla keyword e sugli intenti emersi
- Integra i topic mancanti e le PAA mancanti in modo naturale, non come elenco meccanico
- Genera HTML pronto per CMS
- Un solo H1 all'inizio
- Usa H2 e H3 descrittivi
- Usa liste e tabelle solo dove aggiungono chiarezza
- Evidenzia le entità principali con <strong>
- Chiudi con almeno 4 FAQ in HTML
- Evita tono artificiale, ripetizioni e keyword stuffing
- Il contenuto deve essere più completo, più utile e più aggiornabile rispetto al testo sorgente
"""


def parse_optimized_output(raw: str) -> Dict[str, str]:
    title_tag = ""
    meta_description = ""
    article_html = raw.strip()

    if "TITLE TAG:" in raw and "META DESCRIPTION:" in raw and "OPTIMIZED ARTICLE HTML:" in raw:
        after_title = raw.split("TITLE TAG:", 1)[1]
        title_tag = after_title.split("META DESCRIPTION:", 1)[0].strip()
        after_meta = after_title.split("META DESCRIPTION:", 1)[1]
        meta_description = after_meta.split("OPTIMIZED ARTICLE HTML:", 1)[0].strip()
        article_html = after_meta.split("OPTIMIZED ARTICLE HTML:", 1)[1].strip()

    return {
        "title_tag": title_tag,
        "meta_description": meta_description,
        "article_html": article_html,
    }


def build_tree_graph(gap_tree: List[Dict[str, Any]]) -> Digraph:
    dot = Digraph()
    dot.attr(rankdir="LR")
    dot.node("root", "Gap Analysis")

    for i, macro in enumerate(gap_tree):
        macro_id = f"macro_{i}"
        dot.node(macro_id, f"{macro['topic']}\n[{macro['status']}]")
        dot.edge("root", macro_id)

        for j, child in enumerate(macro.get("children", [])):
            child_id = f"child_{i}_{j}"
            label = f"{child['topic']}\n[{child['status']}]"
            dot.node(child_id, label)
            dot.edge(macro_id, child_id)

    return dot


def export_txt(gap_data: Dict[str, Any], optimized_output: Dict[str, str] | None) -> BytesIO:
    payload = {
        "gap_analysis": gap_data,
        "optimized_output": optimized_output,
    }
    buffer = BytesIO()
    buffer.write(json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"))
    buffer.seek(0)
    return buffer


# =========================
# SIDEBAR
# =========================
st.sidebar.title("Configuration")
SERPER_KEY = st.sidebar.text_input("Serper.dev API Key", type="password", value=st.secrets.get("SERPER_API_KEY", ""))
SERPAPI_KEY = st.sidebar.text_input("SerpAPI Key", type="password", value=st.secrets.get("SERPAPI_KEY", ""))
OPENAI_KEY = st.sidebar.text_input("OpenAI API Key", type="password", value=st.secrets.get("OPENAI_API_KEY", ""))
MODEL_NAME = st.sidebar.text_input("OpenAI model", value=st.secrets.get("OPENAI_MODEL", DEFAULT_MODEL))
COUNTRY = st.sidebar.text_input("Country code", value="it")
LANGUAGE = st.sidebar.text_input("Language code", value="it")
NUM_RESULTS = st.sidebar.slider("Competitor da analizzare", 3, 10, 5)

# =========================
# UI
# =========================
st.title("SEO Gap Analyzer & Optimizer")
st.write(
    "Analizza una SERP, confronta il contenuto della tua URL con competitor e PAA, "
    "mostra la gap analysis ad albero e, solo se richiesto, genera una versione SEO ottimizzata."
)

with st.form("analysis_form"):
    keyword = st.text_input("Keyword")
    input_h1 = st.text_input("Titolo H1")
    source_url = st.text_input("URL da analizzare")
    submitted = st.form_submit_button("Esegui analisi")

if submitted:
    if not SERPER_KEY or not SERPAPI_KEY or not OPENAI_KEY:
        st.error("Inserisci tutte le API key richieste nella sidebar o in Streamlit secrets.")
        st.stop()
    if not keyword.strip() or not input_h1.strip() or not source_url.strip():
        st.error("Keyword, H1 e URL sono obbligatori.")
        st.stop()

    st.session_state.analysis_done = False
    st.session_state.optimization_done = False
    st.session_state.serp_data = None
    st.session_state.gap_data = None
    st.session_state.optimized_output = None

    with st.spinner("Recupero SERP, PAA e contenuti da confrontare..."):
        competitors = get_competitors(keyword, NUM_RESULTS, SERPER_KEY, LANGUAGE, COUNTRY)
        paa = get_people_also_ask(keyword, SERPAPI_KEY, LANGUAGE, COUNTRY)
        source_html, source_text = fetch_page(source_url)
        source_meta = extract_metadata(source_html) if source_html else {"title": "", "h1": "", "meta_description": ""}

        if not source_text:
            st.error("Non sono riuscito a estrarre contenuto dalla URL sorgente.")
            st.stop()

        enriched_competitors = enrich_urls(competitors)
        source_page = {
            **source_meta,
            "text": source_text,
            "url": source_url,
        }

    with st.spinner("Elaboro la gap analysis..."):
        client = build_openai_client(OPENAI_KEY)
        gap_prompt = build_gap_prompt(keyword, input_h1, source_url, source_page, enriched_competitors, paa)
        gap_data = call_llm_json(client, MODEL_NAME, gap_prompt)

    st.session_state.analysis_done = True
    st.session_state.serp_data = {
        "competitors": enriched_competitors,
        "paa": paa,
    }
    st.session_state.source_page = source_page
    st.session_state.gap_data = gap_data

if st.session_state.analysis_done and st.session_state.gap_data:
    serp_data = st.session_state.serp_data
    gap_data = st.session_state.gap_data

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("People Also Ask")
        if serp_data["paa"]:
            for question in serp_data["paa"]:
                st.write(f"- {question}")
        else:
            st.caption("Nessuna PAA trovata.")

        st.subheader("Competitor analizzati")
        for comp in serp_data["competitors"]:
            st.markdown(f"- **{comp.get('html_title') or comp.get('title') or comp['link']}**  ")
            st.caption(comp["link"])

    with col2:
        st.subheader("Sintesi gap analysis")
        summary = gap_data.get("summary", {})
        st.write(f"**Intento di ricerca:** {summary.get('search_intent', '-')}")
        st.write(f"**Verdetto:** {summary.get('overall_verdict', '-')}")
        priorities = summary.get("priority_actions", [])
        if priorities:
            st.write("**Azioni prioritarie**")
            for item in priorities:
                st.write(f"- {item}")

    st.subheader("Grafo ad albero")
    tree = gap_data.get("tree", [])
    if tree:
        st.graphviz_chart(build_tree_graph(tree), use_container_width=True)
    else:
        st.info("La struttura ad albero non è disponibile.")

    st.subheader("Dettaglio gap")
    for macro in tree:
        with st.expander(f"{macro.get('topic', 'Area')} [{macro.get('status', 'n/a')}]"):
            st.write(macro.get("reason", ""))
            for child in macro.get("children", []):
                st.markdown(f"**{child.get('topic', '')}** — {child.get('status', '')}")
                st.write(child.get("reason", ""))
                evidence = child.get("evidence", [])
                if evidence:
                    st.write("Evidenze:")
                    for item in evidence:
                        st.write(f"- {item}")
                suggested_heading = child.get("suggested_heading")
                if suggested_heading:
                    st.caption(f"Heading suggerito: {suggested_heading}")

    st.subheader("Opportunità rilevate")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write("**PAA mancanti**")
        for item in gap_data.get("missing_paa", []):
            st.write(f"- {item}")
    with c2:
        st.write("**Entità mancanti**")
        for item in gap_data.get("missing_entities", []):
            st.write(f"- {item}")
    with c3:
        st.write("**Heading raccomandati**")
        for item in gap_data.get("recommended_headings", []):
            st.write(f"- {item}")

    st.markdown("---")
    st.subheader("Ottimizzazione del contenuto")
    st.write("Vuoi procedere con la generazione della versione ottimizzata del contenuto analizzato?")

    if st.button("Procedi con ottimizzazione"):
        with st.spinner("Genero il contenuto SEO ottimizzato..."):
            client = build_openai_client(OPENAI_KEY)
            optimization_prompt = build_optimization_prompt(
                keyword=keyword,
                input_h1=input_h1,
                source_url=source_url,
                source_page=st.session_state.source_page,
                gap_data=gap_data,
            )
            raw_output = call_llm_text(client, MODEL_NAME, optimization_prompt)
            st.session_state.optimized_output = parse_optimized_output(raw_output)
            st.session_state.optimization_done = True

if st.session_state.optimization_done and st.session_state.optimized_output:
    optimized_output = st.session_state.optimized_output
    st.success("Ottimizzazione completata.")
    st.subheader("SEO metadata")
    st.write("**Title tag**")
    st.write(optimized_output.get("title_tag", ""))
    st.write("**Meta description**")
    st.write(optimized_output.get("meta_description", ""))

    st.subheader("Optimized article HTML")
    st.code(optimized_output.get("article_html", ""), language="html")

if st.session_state.gap_data:
    txt_file = export_txt(st.session_state.gap_data, st.session_state.optimized_output)
    st.download_button(
        label="Scarica export JSON/TXT",
        data=txt_file,
        file_name="seo_gap_analysis_export.json",
        mime="application/json",
    )
