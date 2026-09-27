"""Agentic RAG CV Matcher - Streamlit Frontend.

Pages: Home, Query, Documents, Dashboard, Evaluation
Backend: FastAPI at http://localhost:8000
"""

import streamlit as st
import httpx
import os
import time
import logging
from streamlit.runtime.scriptrunner import get_script_run_ctx

# --- Config ---------------------------------------------------------------

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
API_TIMEOUT = 120.0
API_KEY_HEADER = "X-OpenRouter-Key"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="CV Matcher",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- API Helpers ----------------------------------------------------------


def _has_script_context() -> bool:
    """Whether the current thread can safely call st.* / touch session_state.

    Streamlit runs each session's script in its own ScriptRunner thread --
    never the process's literal threading.main_thread() -- so that's not a
    valid check here. The background ThreadPoolExecutor threads used for
    parallel uploads genuinely have no script context (it isn't propagated
    to them), which is what we actually need to detect.
    """
    return get_script_run_ctx() is not None


def _auth_headers() -> dict:
    """Attach the user-supplied OpenRouter key and/or API auth token, if any.

    Only reads st.session_state when a script context is present -
    background upload threads have none, and uploads don't need an LLM key
    anyway (document ingestion never calls OpenRouter). API_AUTH_TOKEN comes
    from the deployment environment (see README) and must match the backend's
    api_auth_token when the backend is deployed with auth enabled.
    """
    headers = {}
    if _has_script_context():
        key = st.session_state.get("openrouter_api_key", "")
        if key:
            headers[API_KEY_HEADER] = key
    api_token = os.getenv("API_AUTH_TOKEN", "").strip()
    if api_token:
        headers["X-API-Token"] = api_token
    return headers


def api_get(path: str, **kwargs):
    """GET request to FastAPI backend."""
    is_main_thread = _has_script_context()
    kwargs.setdefault("timeout", API_TIMEOUT)
    kwargs["headers"] = {**_auth_headers(), **kwargs.get("headers", {})}
    try:
        resp = httpx.get(f"{API_BASE}{path}", **kwargs)
        resp.raise_for_status()
        return resp.json()
    except httpx.ConnectError as e:
        msg = f"Cannot connect to API at {API_BASE}. Is the backend running?"
        logger.error(msg)
        if is_main_thread:
            st.error(msg)
        return None
    except httpx.HTTPStatusError as e:
        msg = f"API error: {e.response.status_code} - {e.response.text[:200]}"
        logger.error(msg)
        if is_main_thread:
            st.error(msg)
        return None
    except Exception as e:
        msg = f"Request failed: {str(e)}"
        logger.error(msg)
        if is_main_thread:
            st.error(msg)
        return None


def api_post(path: str, json_data=None, files=None, **kwargs):
    """POST request to FastAPI backend."""
    is_main_thread = _has_script_context()
    kwargs.setdefault("timeout", API_TIMEOUT)
    kwargs["headers"] = {**_auth_headers(), **kwargs.get("headers", {})}
    try:
        resp = httpx.post(
            f"{API_BASE}{path}",
            json=json_data,
            files=files,
            **kwargs,
        )
        if resp.status_code in (400, 413, 422):
            try:
                error_data = resp.json()
                if isinstance(error_data, dict):
                    logger.error(f"API Error {resp.status_code} for {path}: {error_data.get('detail')}")
                    return error_data
            except Exception:
                pass
        resp.raise_for_status()
        return resp.json()
    except httpx.ConnectError as e:
        msg = f"Cannot connect to API at {API_BASE}. Is the backend running?"
        logger.error(msg)
        if is_main_thread:
            st.error(msg)
        return None
    except httpx.HTTPStatusError as e:
        msg = f"API error: {e.response.status_code} - {e.response.text[:200]}"
        logger.error(msg)
        if is_main_thread:
            st.error(msg)
        return {"detail": f"API error {e.response.status_code}: {e.response.text[:200]}"}
    except Exception as e:
        msg = f"Request failed: {str(e)}"
        logger.error(msg)
        if is_main_thread:
            st.error(msg)
        return None


def api_delete(path: str, **kwargs):
    """DELETE request to FastAPI backend."""
    is_main_thread = _has_script_context()
    kwargs.setdefault("timeout", API_TIMEOUT)
    kwargs["headers"] = {**_auth_headers(), **kwargs.get("headers", {})}
    try:
        resp = httpx.delete(f"{API_BASE}{path}", **kwargs)
        resp.raise_for_status()
        return resp.json()
    except httpx.ConnectError as e:
        msg = f"Cannot connect to API at {API_BASE}. Is the backend running?"
        logger.error(msg)
        if is_main_thread:
            st.error(msg)
        return None
    except httpx.HTTPStatusError as e:
        msg = f"API error: {e.response.status_code} - {e.response.text[:200]}"
        logger.error(msg)
        if is_main_thread:
            st.error(msg)
        return None
    except Exception as e:
        msg = f"Delete failed: {str(e)}"
        logger.error(msg)
        if is_main_thread:
            st.error(msg)
        return None


# --- Sidebar --------------------------------------------------------------

st.sidebar.title("🔍 CV Matcher")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "🔎 Query", "📄 Documents", "📊 Dashboard", "🧪 Evaluation"],
    index=0,
)

st.sidebar.divider()
st.sidebar.markdown("### 🔑 OpenRouter API Key")
st.sidebar.caption("Sent with each request as a header; never stored server-side.")

if "openrouter_api_key" not in st.session_state:
    st.session_state.openrouter_api_key = ""
if "key_validation" not in st.session_state:
    st.session_state.key_validation = None


def _on_api_key_change():
    st.session_state.key_validation = None  # invalidate stale check on edit


st.sidebar.text_input(
    "API Key",
    type="password",
    label_visibility="collapsed",
    placeholder="sk-or-...",
    key="openrouter_api_key",
    on_change=_on_api_key_change,
)

if st.sidebar.button("Validate Key", use_container_width=True):
    if not st.session_state.openrouter_api_key:
        st.session_state.key_validation = {"valid": False, "detail": "Enter a key first."}
    else:
        with st.sidebar.spinner("Checking key..."):
            st.session_state.key_validation = api_get("/api/key/validate")

validation = st.session_state.key_validation
if validation:
    if validation.get("valid"):
        usage = validation.get("usage")
        limit = validation.get("limit")
        extra = ""
        if usage is not None:
            limit_str = f"${limit:.2f}" if limit is not None else "no limit"
            extra = f" (usage ${usage:.4f} / {limit_str})"
        st.sidebar.success(f"✅ Key is valid{extra}")
    else:
        st.sidebar.error(f"❌ {validation.get('detail', 'Invalid key')}")

st.sidebar.divider()
st.sidebar.caption("Agentic RAG for CV Expertise Matching")
st.sidebar.caption("v0.1.0")


@st.cache_data(ttl=60)
def get_cached_stats():
    """Fetch dashboard stats with a cache duration of 60 seconds to prevent blocking on reruns."""
    return api_get("/api/dashboard/stats")


# --- Components -----------------------------------------------------------

def score_bar(score: float, show_label: bool = True):
    """Render a colored score bar. Score is 0-100."""
    if score >= 75:
        color = "green"
    elif score >= 50:
        color = "orange"
    else:
        color = "red"

    label = f"{score:.0f}%" if show_label else ""
    st.progress(min(score / 100, 1.0), text=f"Match: {label}" if show_label else None)


@st.dialog("Document Viewer", width="large")
def show_document_dialog(filename: str, chunk_text: str):
    st.markdown(f"Viewing source document **{filename}** with matched chunk highlighted:")

    # Fetch content
    with st.spinner("Loading document text..."):
        content_res = api_get(f"/api/documents/{filename}/content")

    if not content_res or "text" not in content_res:
        st.error("Failed to load document content. Make sure it is ingested and available.")
        return

    full_text = content_res["text"]

    # Highlight chunk
    import html as html_lib
    safe_full_text = html_lib.escape(full_text)
    safe_chunk_text = html_lib.escape(chunk_text)

    highlighted_html = ""
    if safe_chunk_text in safe_full_text:
        highlighted_html = safe_full_text.replace(
            safe_chunk_text,
            f'<mark id="highlighted-chunk" style="background-color: #ffeb3b; padding: 2px; font-weight: bold; border-radius: 2px; border: 1px solid #f57f17; color: black;">{safe_chunk_text}</mark>'
        )
    else:
        highlighted_html = safe_full_text + f"\n\n<hr/><mark id='highlighted-chunk' style='background-color: #ffeb3b; color: black;'>{safe_chunk_text}</mark>"

    # Scrollable container and scrollIntoView JS script
    viewer_html = f"""
    <div id="scroll-container" style="position: relative; height: 420px; overflow-y: scroll; border: 1px solid #ccc; padding: 15px; font-family: monospace; white-space: pre-wrap; background-color: #fafafa; border-radius: 4px; color: #333;">{highlighted_html}</div>
    <script>
      setTimeout(function() {{
        const element = document.getElementById("highlighted-chunk");
        if (element) {{
          element.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
        }}
      }}, 250);
    </script>
    """

    st.components.v1.html(viewer_html, height=450)


def match_card(match: dict):
    """Render a single match result card."""
    with st.container(border=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{match.get('person_name', 'Unknown')}**")
            st.caption(f"Source: {match.get('source_document', 'N/A')} | Section: {', '.join(match.get('sections', []))}")
        with col2:
            score = match.get("score", 0)
            if score >= 75:
                st.success(f"{score:.0f}%")
            elif score >= 50:
                st.warning(f"{score:.0f}%")
            else:
                st.error(f"{score:.0f}%")

        evidence = match.get("evidence", "")
        if evidence:
            st.markdown(f"**Evidence:** {evidence}")

        # Document viewer button
        matched_chunks = match.get("matched_chunks", [])
        if matched_chunks:
            source_doc = match.get("source_document", "")
            person_name = match.get("person_name", "Unknown")
            btn_key = f"btn_view_{source_doc}_{person_name.replace(' ', '_')}"
            if st.button(f"📄 View Source Chunk in Document ({source_doc})", key=btn_key):
                show_document_dialog(source_doc, matched_chunks[0])


def status_badge(text: str, level: str = "info"):
    """Render a status badge."""
    colors = {
        "success": "🟢",
        "warning": "🟡",
        "error": "🔴",
        "info": "🔵",
    }
    icon = colors.get(level, "⚪")
    st.markdown(f"{icon} **{text}**")


def render_search_interface(key_suffix: str = ""):
    # Query input
    query = st.text_area(
        "Ask a question about your team's expertise",
        placeholder="e.g., Find a Java developer with Spring Boot and Kafka experience",
        height=100,
        key=f"query_input_{key_suffix}",
    )

    res_key = f"search_results_{key_suffix}"
    last_q_key = f"last_query_{key_suffix}"

    if res_key not in st.session_state:
        st.session_state[res_key] = None
    if last_q_key not in st.session_state:
        st.session_state[last_q_key] = ""

    # Clear previous results if the query itself is modified
    if query != st.session_state[last_q_key]:
        st.session_state[res_key] = None
        st.session_state[last_q_key] = query

    col1, col2 = st.columns([1, 5])
    with col1:
        search_btn = st.button("🔍 Search", type="primary", disabled=False, key=f"search_btn_{key_suffix}")

    if search_btn:
        if not query.strip():
            st.warning("Please type a search query first!")
        else:
            with st.spinner("Searching knowledge base..."):
                result = api_post(
                    "/api/query/",
                    json_data={
                        "question": query,
                        "use_llm_planner": True,
                        "use_llm_validator": True,
                    }
                )
                st.session_state[res_key] = result

    result = st.session_state[res_key]
    if result:
        # Query type badge
        qt = result.get("query_type", "unknown")
        if qt == "out_of_scope":
            st.info(f"**Query type:** {qt}")
            st.warning(result.get("answer", ""))
            if result.get("rejection_reason"):
                st.caption(result["rejection_reason"])
            return

        # Retry indicator
        retries = result.get("retry_count", 0)
        if retries > 0:
            st.info(f"Answer was refined {retries} time(s) for quality.")

        # Validation status
        if result.get("validation_passed"):
            status_badge("Answer validated", "success")
        else:
            status_badge("Validation warning", "warning")
            if result.get("validation_feedback"):
                st.caption(result["validation_feedback"])

        # Answer
        st.markdown("---")
        st.markdown("### Answer")
        st.markdown(result.get("answer", "No answer generated."))

        # Match cards (Top 3 as requested)
        matches = result.get("matches", [])
        if matches:
            st.markdown("---")
            display_matches = matches[:3]
            st.markdown(f"### Top Matches ({len(display_matches)} of {len(matches)} found)")
            for match in display_matches:
                match_card(match)

        # --- Parallel File Uploader Helper ----------------------------------------

def upload_files_parallel(uploaded_files):
    import concurrent.futures
    total_files = len(uploaded_files)
    
    # Progress bar and status indicator
    progress_bar = st.progress(0, text="Uploading files in parallel...")
    success_count = 0
    completed = 0
    
    def upload_one_file(file):
        try:
            res = api_post(
                "/api/documents/upload",
                files={"file": (file.name, file.getvalue(), file.type)},
                timeout=120.0
            )
            return file.name, res
        except Exception as e:
            logger.error(f"Upload thread failed for {file.name}: {str(e)}")
            return file.name, None

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(upload_one_file, f): f for f in uploaded_files}
        
        for future in concurrent.futures.as_completed(futures):
            filename, res = future.result()
            completed += 1
            percent = min(completed / total_files, 1.0)
            progress_bar.progress(percent, text=f"Processed {completed}/{total_files} files...")
            
            if res and res.get("success"):
                success_count += 1
                if res.get("tainted"):
                    st.warning(f"⚠️ {filename}: potential injection warning")
                else:
                    st.success(f"✅ {filename}: {res.get('chunks_created', 0)} chunks")
            else:
                error_msg = res.get("detail") if res and isinstance(res, dict) else "Upload failed"
                st.error(f"❌ {filename}: {error_msg}")
                
    progress_bar.empty()
    return success_count


def render_uploader(key_prefix: str, label_visibility: str = "visible"):
    """Render a file uploader widget wired up to upload_files_parallel.

    Shared by the Home and Documents pages so accepted file types and the
    post-upload cache/rerun handling only need to change in one place.
    """
    state_key = f"{key_prefix}_key"
    if state_key not in st.session_state:
        st.session_state[state_key] = 0

    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "txt", "csv", "xlsx"],
        accept_multiple_files=True,
        label_visibility=label_visibility,
        key=f"{key_prefix}_{st.session_state[state_key]}",
    )

    if uploaded_files:
        success_count = upload_files_parallel(uploaded_files)
        if success_count > 0:
            st.session_state[state_key] += 1
            st.cache_data.clear()
            st.rerun()


# --- Page: Home -----------------------------------------------------------

def page_home():
    st.title("🔍 Agentic RAG CV Matcher")
    st.markdown("### Welcome to the Agentic RAG CV Expertise Matcher!")

    st.markdown("""
    This application utilizes an advanced **Agentic RAG** (Retrieval-Augmented Generation) pipeline to help you search, locate, and match internal experts based on their CV documents.
    """)

    # Check if knowledge base has documents (cached stats)
    stats = get_cached_stats()

    # Render premium system architecture layout
    with st.expander("🛠️ How the Agentic Architecture Works", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            #### System Workflow:
            1. **Planner Agent**: Classifies the query type (simple search, job description, out of scope) and extracts requirements.
            2. **LLM Guard Guardrails**: Scans queries and chunks to prevent prompt injection and ensure data safety.
            3. **CV Retriever**: Searches the **ChromaDB vector store** using `all-MiniLM-L6-v2` embeddings to pull candidate CV chunks.
            4. **Responder Agent**: Ranks candidates, computes match scores (0-100%), and writes a cited justification.
            5. **Validator Agent**: Self-corrects by validating the responder's output against the raw CV text. If issues (hallucinations, missing citations) are found, it triggers a refinement loop.
            """)
        with col2:
            st.markdown("""
            #### How to Use:
            1. Go to **📄 Documents** in the sidebar to upload candidate CVs (supported: **PDF, TXT, CSV, Excel**).
            2. Type your question directly in the search bar below or on the **🔎 Query** page.
            3. Click on **📄 View Source Chunk** next to any candidate to open their CV and see their matching qualifications highlighted and centered on your screen.
            4. View database graphs on **📊 Dashboard** or run tests on **🧪 Evaluation**.
            """)

    if stats and stats.get("total_documents", 0) > 0:
        st.markdown("---")
        st.markdown("### 📊 Database Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Ingested Documents", stats["total_documents"])
        col2.metric("Extracted Text Chunks", stats["total_chunks"])
        col3.metric("Queries Handled", stats.get("total_queries", 0))

        st.markdown("---")
        st.markdown("### 🔎 Query the Knowledge Base Directly")
        render_search_interface(key_suffix="home")
    else:
        st.warning("Knowledge base is currently empty. Upload CVs to get started.")

        # Upload zone
        st.markdown("---")
        st.markdown("### 📤 Upload CV Documents")
        st.markdown("Supported formats: **PDF**, **TXT**, **CSV**, **Excel (.xlsx)**")

        render_uploader("home_uploader", label_visibility="collapsed")

        # Synthetic data option
        st.markdown("---")
        st.markdown("### 🎲 Generate Sample Data for Testing")
        if st.button("Generate 20 Sample CVs"):
            with st.spinner("Generating synthetic CVs..."):
                result = api_post("/api/synthetic/generate", json_data={"count": 20})
                if result and result.get("success"):
                    st.success(result.get("message", "Done"))
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error("Generation failed")


# --- Page: Query ----------------------------------------------------------

def page_query():
    st.title("🔎 Query Knowledge Base")
    render_search_interface(key_suffix="query")


# --- Page: Documents ------------------------------------------------------

def page_documents():
    st.title("📄 Document Manager")

    result = api_get("/api/documents/")
    if result is None:
        return

    docs = result.get("documents", [])
    total_chunks = result.get("total_chunks", 0)

    if not docs:
        st.info("No documents uploaded yet. Go to **🏠 Home** to upload CVs.")
        st.markdown("---")
        st.markdown("### 📤 Upload CV Documents")
        render_uploader("doc_uploader")
        return

    # Define select_all change callback to update session state values safely
    def on_select_all_change():
        val = st.session_state.get("select_all_checkbox", False)
        for d in docs:
            st.session_state[f"check_{d['filename']}"] = val

    # Determine which files are currently selected in session_state
    selected_files = []
    for doc in docs:
        state_key = f"check_{doc['filename']}"
        if state_key not in st.session_state:
            st.session_state[state_key] = False
        if st.session_state[state_key]:
            selected_files.append(doc['filename'])

    col1, col2 = st.columns(2)
    col1.metric("Total Documents", len(docs))
    col2.metric("Total Chunks", total_chunks)

    st.markdown("---")

    # Select all and single global remove button at the top
    col_sel_all, col_del_btn = st.columns([1.5, 4])
    with col_sel_all:
        select_all = st.checkbox("Select All", key="select_all_checkbox", on_change=on_select_all_change)
    with col_del_btn:
        if selected_files:
            if st.button(f"🗑️ Remove Selected ({len(selected_files)})", type="primary", key="batch_delete_btn"):
                with st.spinner("Removing selected documents..."):
                    for filename in selected_files:
                        api_delete(f"/api/documents/{filename}")
                st.success(f"Successfully removed {len(selected_files)} document(s)!")
                st.cache_data.clear()
                # Clean up deleted checkboxes from session state
                for filename in selected_files:
                    st.session_state.pop(f"check_{filename}", None)
                st.session_state.pop("select_all_checkbox", None)
                st.rerun()
        else:
            st.button("🗑️ Remove Selected", disabled=True, key="batch_delete_disabled")

    st.markdown("---")

    for doc in docs:
        col_info, col_chunks, col_status, col_check = st.columns([4, 1.5, 1.5, 1])
        with col_info:
            st.markdown(f"**{doc['filename']}**")
            st.caption(f"Sections: {', '.join(doc.get('sections', [])[:5])}")
        with col_chunks:
            st.metric("Chunks", doc["chunk_count"])
        with col_status:
            if doc.get("tainted"):
                st.warning("⚠️ Tainted")
            else:
                st.success("✅ Clean")
        with col_check:
            # Checkbox placed on the right side, using collasped labels to keep UI neat
            st.checkbox("", key=f"check_{doc['filename']}", label_visibility="collapsed")

    # Upload more
    st.markdown("---")
    st.markdown("### 📤 Upload More Documents")
    render_uploader("doc_uploader")


# --- Page: Dashboard ------------------------------------------------------

def page_dashboard():
    st.title("📊 Dashboard")

    stats = api_get("/api/dashboard/stats")
    if not stats:
        return

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Documents", stats.get("total_documents", 0))
    col2.metric("Chunks", stats.get("total_chunks", 0))
    col3.metric("Queries", stats.get("total_queries", 0))
    col4.metric("Avg Match", f"{stats.get('avg_match_score', 0):.1f}%")

    st.markdown("---")

    def draw_pie_chart(data_dict: dict, title: str):
        import matplotlib.pyplot as plt
        import pandas as pd
        
        if not data_dict:
            return
            
        df = pd.DataFrame(list(data_dict.items()), columns=["Label", "Count"])
        df = df.sort_values(by="Count", ascending=False)
        
        # Group small items into 'Other' if there are too many (e.g. > 7)
        if len(df) > 7:
            top_n = df.iloc[:6]
            other_sum = df.iloc[6:]["Count"].sum()
            other_df = pd.DataFrame([{"Label": "Other", "Count": other_sum}])
            df = pd.concat([top_n, other_df], ignore_index=True)
            
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        
        wedges, texts, autotexts = ax.pie(
            df["Count"],
            labels=df["Label"],
            autopct="%1.1f%%",
            startangle=140,
            colors=plt.cm.Pastel1.colors if len(df) <= 9 else plt.cm.tab20.colors,
            textprops=dict(color="#333333")
        )
        
        # Style percentages
        for autotext in autotexts:
            autotext.set_fontsize(9)
            autotext.set_weight('bold')
            
        ax.axis("equal")
        st.markdown(f"### {title}")
        st.pyplot(fig)

    col_left, col_right = st.columns(2)
    
    with col_left:
        # Format distribution
        formats = stats.get("documents_by_format", {})
        if formats:
            draw_pie_chart(formats, "Documents by Format")

    with col_right:
        # Section distribution
        sections = stats.get("chunks_by_section", {})
        if sections:
            draw_pie_chart(sections, "Chunks by Section")

    # Query performance
    if stats.get("total_queries", 0) > 0:
        st.markdown("---")
        st.markdown("### Query Performance Summary")
        col1, col2 = st.columns(2)
        col1.metric("Avg Match Score", f"{stats.get('avg_match_score', 0):.1f}%")
        col2.metric("Avg Latency", f"{stats.get('avg_latency_ms', 0):.0f}ms")


# --- Page: Evaluation -----------------------------------------------------

def _poll_evaluation_progress():
    """Poll GET /api/evaluation/progress until the background run finishes,
    rendering a live progress bar instead of blocking on one long request."""
    progress_bar = st.progress(0, text="Starting evaluation...")
    status = "running"
    progress = {}

    while status == "running":
        progress = api_get("/api/evaluation/progress")
        if progress is None:
            progress_bar.empty()
            st.error("Lost connection while polling evaluation progress.")
            return
        status = progress.get("status", "error")
        completed = progress.get("completed", 0)
        total = progress.get("total", 0) or 1
        current_question = progress.get("current_question") or ""

        label = f"Evaluating... {completed}/{total}"
        if current_question:
            label += f" - {current_question[:60]}"
        progress_bar.progress(min(completed / total, 1.0), text=label)

        if status == "running":
            time.sleep(1.5)

    progress_bar.empty()
    if status == "done":
        result = progress.get("result") or {}
        st.success(f"Completed: {result.get('passed', 0)}/{result.get('total_questions', 0)} passed")
    else:
        st.error(f"Evaluation failed: {progress.get('error') or 'unknown error'}")


def page_evaluation():
    st.title("🧪 Evaluation")

    col1, col2 = st.columns([1, 3])
    with col1:
        run_clicked = st.button("▶️ Run Evaluation", type="primary")

    # Check for an in-flight run (covers both a fresh click and a page reload
    # while a previously started run is still going).
    current_progress = api_get("/api/evaluation/progress")
    already_running = bool(current_progress and current_progress.get("status") == "running")

    if run_clicked and not already_running:
        api_post("/api/evaluation/start")
        already_running = True

    if already_running:
        _poll_evaluation_progress()
        st.cache_data.clear()

    # Load existing results
    eval_data = api_get("/api/evaluation/results")
    if not eval_data or not eval_data.get("latest"):
        st.info("No evaluation results yet. Click 'Run Evaluation' to start.")
        return

    latest = eval_data["latest"]

    # Summary metrics
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Pass Rate", f"{latest.get('pass_rate', 0)}%")
    col2.metric("Passed", latest.get("passed", 0))
    col3.metric("Failed", latest.get("failed", 0))
    col4.metric("Avg Latency", f"{latest.get('avg_latency_ms', 0):.0f}ms")

    # Results table
    results = latest.get("results", [])
    if results:
        st.markdown("### Results")
        import pandas as pd
        df = pd.DataFrame(results)
        display_cols = ["question", "category", "passed", "query_type", "top_score", "latency_ms"]
        available = [c for c in display_cols if c in df.columns]
        st.dataframe(df[available], use_container_width=True)

    # Failure modes
    failure_modes = latest.get("failure_modes", {})
    if failure_modes:
        st.markdown("### Failure Modes")
        for mode, count in failure_modes.items():
            st.warning(f"**{mode}**: {count} occurrence(s)")


# --- Router ---------------------------------------------------------------

if page == "🏠 Home":
    page_home()
elif page == "🔎 Query":
    page_query()
elif page == "📄 Documents":
    page_documents()
elif page == "📊 Dashboard":
    page_dashboard()
elif page == "🧪 Evaluation":
    page_evaluation()
