"""
app.py
======
MAIN APPLICATION: Smart Exam Paper Generator using NLP
=======================================================

Run with:
    streamlit run app.py

Architecture:
    Streamlit frontend → NLP pipeline modules (utils/)
    ┌──────────────┐     ┌──────────────────────────────────────────────┐
    │  app.py      │────▶│  file_processor → preprocessor              │
    │  (UI/Router) │     │  → keyword_extractor → question_generator   │
    └──────────────┘     │  → difficulty_classifier → summarizer       │
                         │  → visualizer → exam_formatter              │
                         └──────────────────────────────────────────────┘
"""

# ─────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────
import os, sys, time, io, datetime
import streamlit as st

# Add project root to path so utils imports work
sys.path.insert(0, os.path.dirname(__file__))

from utils.file_processor       import extract_text, get_file_stats
from utils.preprocessor         import full_preprocess, get_named_entities, get_noun_phrases
from utils.keyword_extractor    import extract_all_keywords, get_top_keywords
from utils.question_generator   import generate_all_questions
from utils.difficulty_classifier import classify_all_questions, get_difficulty_stats
from utils.summarizer           import generate_full_summary
from utils.visualizer           import (
    plot_word_frequency, plot_tfidf_keywords,
    plot_word_cloud, plot_pos_distribution,
    plot_difficulty_distribution, plot_nlp_stats,
)
from utils.exam_formatter       import format_exam_paper, format_answer_key


# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be first Streamlit call)
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title  = "Smart Exam Paper Generator",
    page_icon   = "🎓",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)


# ═══════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Global ── */
body { font-family: 'Segoe UI', sans-serif; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    color: white;
}
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 15px; }

/* ── Metric cards ── */
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 12px; padding: 18px 24px;
    color: white; text-align: center; margin-bottom: 10px;
}
.metric-card h2 { margin:0; font-size: 2rem; }
.metric-card p  { margin:0; opacity: 0.85; font-size: 0.9rem; }

/* ── Section headers ── */
.section-header {
    background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-size: 1.6rem; font-weight: 700; margin-bottom: 6px;
}

/* ── Question cards ── */
.q-card {
    background: #f8f9ff;
    border-left: 5px solid #4C72B0;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 14px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.q-card-hard  { border-left-color: #C44E52; }
.q-card-medium{ border-left-color: #DD8452; }
.q-card-easy  { border-left-color: #55A868; }

/* ── Difficulty badges ── */
.badge-easy   { background:#55A868; color:white; border-radius:4px; padding:2px 8px; font-size:0.75rem; }
.badge-medium { background:#DD8452; color:white; border-radius:4px; padding:2px 8px; font-size:0.75rem; }
.badge-hard   { background:#C44E52; color:white; border-radius:4px; padding:2px 8px; font-size:0.75rem; }

/* ── Answer highlight ── */
.answer-box {
    background:#e8f5e9; border:1px solid #a5d6a7;
    border-radius:6px; padding:8px 14px; margin-top:6px;
    font-weight:600; color:#2e7d32;
}

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 14px; padding: 36px 40px;
    color: white; margin-bottom: 28px;
}
.hero h1 { font-size: 2.4rem; margin-bottom: 6px; }
.hero p  { opacity: 0.8; font-size: 1.1rem; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE  (persist data across rerenders)
# ═══════════════════════════════════════════════════════════════════════════
def _init_state():
    defaults = {
        "raw_text":       "",
        "preprocessed":   None,
        "keywords":       None,
        "questions":      None,
        "annotated_qs":   None,
        "summary":        None,
        "exam_paper":     "",
        "answer_key":     "",
        "file_stats":     None,
        "subject_name":   "General Studies",
        "processing_done": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎓 Smart Exam Generator")
    st.markdown("*NLP-Powered Question Engine*")
    st.markdown("---")

    page = st.radio(
        "📌 Navigate",
        [
            "🏠 Home & Upload",
            "🔬 NLP Analysis",
            "🔑 Keywords & NER",
            "❓ Questions",
            "📊 Visualizations",
            "📄 Exam Paper",
            "🔍 Search",
            "ℹ️ NLP Concepts",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**⚙️ Settings**")
    st.session_state.subject_name = st.text_input(
        "Subject Name", value=st.session_state.subject_name
    )
    num_mcq   = st.slider("# MCQs",            2, 15, 5)
    num_fib   = st.slider("# Fill in Blanks",  2, 15, 5)
    num_short = st.slider("# Short Questions", 2, 10, 5)
    num_long  = st.slider("# Long Questions",  1,  6, 3)
    show_ans  = st.checkbox("Show Answers in Exam Paper", value=False)

    st.markdown("---")
    st.markdown(
        "<small>Built with ❤️ using Streamlit + NLTK + spaCy + scikit-learn</small>",
        unsafe_allow_html=True
    )


# ═══════════════════════════════════════════════════════════════════════════
# HELPER: DIFFICULTY BADGE HTML
# ═══════════════════════════════════════════════════════════════════════════
def _diff_badge(difficulty: str) -> str:
    cls = {
        "Easy":   "badge-easy",
        "Medium": "badge-medium",
        "Hard":   "badge-hard",
    }.get(difficulty, "badge-medium")
    return f'<span class="{cls}">{difficulty}</span>'


# ═══════════════════════════════════════════════════════════════════════════
# GATE: Require processing for all other pages
# ═══════════════════════════════════════════════════════════════════════════
def _require_processing():
    if not st.session_state.processing_done:
        st.warning("⚠️ Please upload a document and click **Generate Exam Paper** on the Home page first.")
        st.stop()


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1 – HOME & UPLOAD
# ═══════════════════════════════════════════════════════════════════════════
if page == "🏠 Home & Upload":

    st.markdown("""
    <div class="hero">
        <h1>🎓 Smart Exam Paper Generator</h1>
        <p>Upload a PDF or TXT study document → Let NLP do the rest.<br>
        MCQs · Fill-in-the-blanks · Short & Long Questions · Auto-answers · Difficulty Classification</p>
    </div>
    """, unsafe_allow_html=True)

    # ── File Upload ──
    st.markdown('<p class="section-header">📂 Upload Your Study Material</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drag & drop or click to browse",
        type=["pdf", "txt"],
        help="Supports PDF and plain-text files.",
    )

    # ── Paste text option ──
    st.markdown("**OR paste text directly:**")
    pasted_text = st.text_area(
        "Paste study notes here", height=180,
        placeholder="Paste any educational text, study notes, or article…",
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        process_btn = st.button("🚀 Generate Exam Paper", use_container_width=True, type="primary")

    if process_btn:
        raw = ""
        if uploaded_file:
            with st.spinner("📖 Extracting text…"):
                raw = extract_text(uploaded_file)
        elif pasted_text.strip():
            raw = pasted_text.strip()
        else:
            st.warning("⚠️ Please upload a file or paste some text first.")
            st.stop()

        if raw.startswith("❌"):
            st.error(raw)
            st.stop()

        st.session_state.raw_text   = raw
        st.session_state.file_stats = get_file_stats(raw)

        # ── Run the full NLP pipeline with progress bar ──
        progress = st.progress(0, text="Starting NLP pipeline…")

        with st.spinner("🔬 Preprocessing text…"):
            st.session_state.preprocessed = full_preprocess(raw)
        progress.progress(20, text="Preprocessing done ✅")

        with st.spinner("🔑 Extracting keywords…"):
            st.session_state.keywords = extract_all_keywords(raw, top_n=20)
        progress.progress(40, text="Keyword extraction done ✅")

        with st.spinner("📝 Summarizing content…"):
            st.session_state.summary = generate_full_summary(raw)
        progress.progress(55, text="Summarization done ✅")

        with st.spinner("❓ Generating questions…"):
            qs = generate_all_questions(
                raw,
                num_mcq=num_mcq, num_fib=num_fib,
                num_short=num_short, num_long=num_long,
            )
            st.session_state.questions = qs
        progress.progress(75, text="Questions generated ✅")

        with st.spinner("🎯 Classifying difficulty…"):
            st.session_state.annotated_qs = classify_all_questions(qs)
        progress.progress(90, text="Difficulty classified ✅")

        with st.spinner("📄 Formatting exam paper…"):
            st.session_state.exam_paper  = format_exam_paper(
                st.session_state.annotated_qs,
                subject=st.session_state.subject_name,
                show_answers=show_ans,
            )
            st.session_state.answer_key  = format_answer_key(st.session_state.annotated_qs)
        progress.progress(100, text="Done! 🎉")
        time.sleep(0.4)
        progress.empty()

        st.session_state.processing_done = True
        st.success("✅ Exam paper generated successfully! Use the sidebar to navigate.")

    # ── Show file stats if processed ──
    if st.session_state.processing_done and st.session_state.file_stats:
        st.markdown("---")
        st.markdown("### 📊 Document Overview")
        fs = st.session_state.file_stats
        c1, c2, c3, c4 = st.columns(4)
        for col, label, val, icon in [
            (c1, "Characters",  fs["characters"], "📝"),
            (c2, "Words",       fs["words"],       "📖"),
            (c3, "Sentences",   fs["sentences"],   "📋"),
            (c4, "Paragraphs",  fs["paragraphs"],  "📄"),
        ]:
            col.markdown(
                f'<div class="metric-card"><h2>{icon} {val:,}</h2><p>{label}</p></div>',
                unsafe_allow_html=True
            )

        st.markdown("### 📃 Extracted Text Preview")
        preview = st.session_state.raw_text[:2000]
        if len(st.session_state.raw_text) > 2000:
            preview += "\n\n… [truncated for preview] …"
        st.text_area("Raw text (first 2000 chars)", preview, height=250)

        # Summary quick-view
        if st.session_state.summary:
            st.markdown("### 📑 Auto-Generated Summary")
            st.info(st.session_state.summary["short_summary"])

    elif not st.session_state.processing_done:
        # Instructions when no file uploaded yet
        st.markdown("---")
        st.markdown("### 📌 How It Works")
        steps = [
            ("1️⃣ Upload", "Upload a PDF or TXT study document."),
            ("2️⃣ Process", "Click **Generate Exam Paper** to run the NLP pipeline."),
            ("3️⃣ Explore", "Navigate the sections: NLP Analysis, Keywords, Questions, Visualizations."),
            ("4️⃣ Download", "Download the formatted exam paper and answer key as text files."),
        ]
        cols = st.columns(4)
        for col, (title, desc) in zip(cols, steps):
            col.markdown(f"**{title}**")
            col.markdown(desc)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2 – NLP ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🔬 NLP Analysis":
    _require_processing()
    pre = st.session_state.preprocessed

    st.markdown('<p class="section-header">🔬 NLP Preprocessing Pipeline</p>', unsafe_allow_html=True)
    st.markdown(
        "This page shows every step of the NLP preprocessing pipeline, "
        "demonstrating how raw text is transformed into clean, analyzable data."
    )

    tabs = st.tabs([
        "📌 Sentences", "✂️ Tokens", "🔡 Lowercase",
        "🚫 No Punct", "🗑️ No Stopwords", "🌿 Lemmas",
        "🏷️ POS Tags", "📊 Stats"
    ])

    with tabs[0]:
        st.markdown("**Sentence Segmentation** – splitting document into individual sentences using NLTK Punkt tokenizer.")
        for i, s in enumerate(pre["sentences"][:20], 1):
            st.markdown(f"`{i}.` {s}")
        if len(pre["sentences"]) > 20:
            st.info(f"… and {len(pre['sentences'])-20} more sentences.")

    with tabs[1]:
        st.markdown("**Tokenization** – splitting text into words/punctuation tokens using NLTK `word_tokenize`.")
        st.code(str(pre["tokens"][:60]), language="python")
        st.metric("Total Tokens", len(pre["tokens"]))

    with tabs[2]:
        st.markdown("**Lowercasing** – normalizing all tokens to lowercase so 'Machine' == 'machine'.")
        st.code(str(pre["lowercase_tokens"][:60]), language="python")

    with tabs[3]:
        st.markdown("**Punctuation Removal** – keeping only alphabetic tokens, removing digits and symbols.")
        st.code(str(pre["clean_tokens"][:60]), language="python")
        col1, col2 = st.columns(2)
        col1.metric("Before removal", len(pre["tokens"]))
        col2.metric("After removal",  len(pre["clean_tokens"]))

    with tabs[4]:
        st.markdown("**Stopword Removal** – eliminating common function words (the, is, at, which…) using NLTK's 179-word English stopword list.")
        st.code(str(pre["no_stopwords"][:60]), language="python")
        col1, col2, col3 = st.columns(3)
        col1.metric("Before",   len(pre["clean_tokens"]))
        col2.metric("After",    len(pre["no_stopwords"]))
        col3.metric("Removed",  len(pre["clean_tokens"]) - len(pre["no_stopwords"]))

    with tabs[5]:
        st.markdown("**Lemmatization** – reducing words to their dictionary base form using NLTK WordNetLemmatizer.")
        st.markdown("Examples: `running→run`, `studies→study`, `better→good`")
        st.code(str(pre["lemmas"][:60]), language="python")

    with tabs[6]:
        st.markdown("**POS Tagging** – assigning grammatical labels to each token using NLTK's averaged perceptron tagger.")
        st.markdown("Tags: `NN`=Noun, `VB`=Verb, `JJ`=Adjective, `RB`=Adverb, `IN`=Preposition")
        df_rows = [{"Word": w, "POS Tag": t} for w, t in pre["pos_tags"][:50]]
        st.dataframe(df_rows, use_container_width=True, height=350)

    with tabs[7]:
        st.markdown("**Pipeline Statistics**")
        stats_data = {
            "Sentences":      len(pre["sentences"]),
            "Raw Tokens":     len(pre["tokens"]),
            "Clean Tokens":   len(pre["clean_tokens"]),
            "After Stopwords":len(pre["no_stopwords"]),
            "Lemmas":         len(pre["lemmas"]),
            "Unique Lemmas":  len(set(pre["lemmas"])),
        }
        col1, col2 = st.columns(2)
        for i, (k, v) in enumerate(stats_data.items()):
            (col1 if i % 2 == 0 else col2).metric(k, v)
        img = plot_nlp_stats(stats_data)
        if img:
            st.image(img, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3 – KEYWORDS & NER
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🔑 Keywords & NER":
    _require_processing()
    kw = st.session_state.keywords

    st.markdown('<p class="section-header">🔑 Keyword Extraction & Named Entity Recognition</p>', unsafe_allow_html=True)

    tabs = st.tabs(["📊 TF-IDF Keywords", "🔢 Frequency Keywords", "📌 Noun Phrases", "🏷️ Named Entities"])

    with tabs[0]:
        st.markdown("**TF-IDF Keywords** – words that are important in *this* document but rare across typical documents.")
        tfidf_kws = kw["tfidf_keywords"]
        if tfidf_kws:
            rows = [{"Keyword": k, "TF-IDF Score": round(v, 4)} for k, v in tfidf_kws]
            st.dataframe(rows, use_container_width=True)
        else:
            st.info("No TF-IDF keywords extracted.")

    with tabs[1]:
        st.markdown("**Frequency Keywords** – most common content words after preprocessing.")
        freq_kws = kw["frequency_keywords"]
        if freq_kws:
            rows = [{"Keyword": k, "Frequency": v} for k, v in freq_kws]
            st.dataframe(rows, use_container_width=True)

    with tabs[2]:
        st.markdown("**Noun Phrases** – multi-word expressions identified by spaCy's dependency parser.")
        np_kws = kw["noun_phrases"]
        if np_kws:
            rows = [{"Noun Phrase": k, "Frequency": v} for k, v in np_kws]
            st.dataframe(rows, use_container_width=True)
        else:
            st.info("No noun phrases extracted.")

    with tabs[3]:
        st.markdown("**Named Entity Recognition** – real-world objects detected by spaCy's NER model.")
        st.markdown("Types: `PERSON`, `ORG` (organization), `GPE` (location), `DATE`, `MONEY`, `LAW` …")
        entities = kw["entities"]
        if entities:
            for label, items in entities.items():
                with st.expander(f"**{label}** ({len(items)} entities)", expanded=(label in ["ORG","PERSON","GPE"])):
                    st.write(", ".join(items))
        else:
            st.info("No named entities detected in the text.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4 – QUESTIONS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "❓ Questions":
    _require_processing()
    qs  = st.session_state.annotated_qs or st.session_state.questions

    st.markdown('<p class="section-header">❓ Generated Questions</p>', unsafe_allow_html=True)

    if not qs:
        st.error("No questions generated yet.")
        st.stop()

    tabs = st.tabs(["🅰️ MCQs", "📝 Fill Blanks", "📋 Short Qs", "📜 Long Qs", "📊 Difficulty Stats"])

    # ── MCQs ──
    with tabs[0]:
        mcqs = qs.get("mcqs", [])
        st.markdown(f"**{len(mcqs)} Multiple Choice Questions**")
        for i, q in enumerate(mcqs, 1):
            diff = q.get("difficulty", "Medium")
            st.markdown(
                f'<div class="q-card q-card-{diff.lower()}">'
                f'<b>Q{i}.</b> {_diff_badge(diff)} &nbsp; {q["question"]}'
                f'</div>',
                unsafe_allow_html=True
            )
            cols = st.columns(2)
            opts = q["options"]
            opt_list = list(opts.items())
            for j, (label, text) in enumerate(opt_list):
                col = cols[j % 2]
                is_correct = (label == q["answer"])
                prefix = "✅" if is_correct else "⬜"
                col.markdown(f"{prefix} **({label})** {text}")
            st.markdown(
                f'<div class="answer-box">✅ Correct Answer: ({q["answer"]}) {q["answer_text"]}</div>',
                unsafe_allow_html=True
            )
            st.markdown("<br>", unsafe_allow_html=True)

    # ── Fill in Blanks ──
    with tabs[1]:
        fibs = qs.get("fill_blanks", [])
        st.markdown(f"**{len(fibs)} Fill-in-the-Blank Questions**")
        for i, q in enumerate(fibs, 1):
            diff = q.get("difficulty", "Medium")
            st.markdown(
                f'<div class="q-card q-card-{diff.lower()}">'
                f'<b>Q{i}.</b> {_diff_badge(diff)}<br><br>'
                f'📝 {q["question"]}<br>'
                f'<small>{q.get("hint","")}</small>'
                f'</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div class="answer-box">✅ Answer: {q["answer"]}</div>',
                unsafe_allow_html=True
            )
            st.markdown("<br>", unsafe_allow_html=True)

    # ── Short Questions ──
    with tabs[2]:
        short_qs = qs.get("short_questions", [])
        st.markdown(f"**{len(short_qs)} Short Answer Questions** (5 marks each)")
        for i, q in enumerate(short_qs, 1):
            diff = q.get("difficulty", "Medium")
            with st.expander(f"Q{i}. [{diff}]  {q['question'][:80]}…"):
                st.markdown(f"**Question:** {q['question']}")
                st.markdown("---")
                st.markdown(f"**Model Answer:** {q['answer']}")

    # ── Long Questions ──
    with tabs[3]:
        long_qs = qs.get("long_questions", [])
        st.markdown(f"**{len(long_qs)} Long / Descriptive Questions** (attempt any 2)")
        for i, q in enumerate(long_qs, 1):
            diff   = q.get("difficulty", "Hard")
            marks  = q.get("marks", 10)
            with st.expander(f"Q{i}. [{diff}] [{marks} Marks]  {q['question'][:80]}…"):
                st.markdown(f"**Question:** {q['question']}")
                st.markdown("---")
                st.markdown(f"**Model Answer:** {q['answer']}")

    # ── Difficulty Stats ──
    with tabs[4]:
        stats = get_difficulty_stats(qs)
        st.markdown("**Question Difficulty Distribution**")
        col1, col2, col3 = st.columns(3)
        col1.metric("🟢 Easy",   stats.get("Easy",   0))
        col2.metric("🟡 Medium", stats.get("Medium", 0))
        col3.metric("🔴 Hard",   stats.get("Hard",   0))
        img = plot_difficulty_distribution(stats)
        if img:
            st.image(img, use_container_width=False)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 5 – VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📊 Visualizations":
    _require_processing()
    pre = st.session_state.preprocessed
    kw  = st.session_state.keywords
    qs  = st.session_state.annotated_qs or st.session_state.questions

    st.markdown('<p class="section-header">📊 NLP Visualizations</p>', unsafe_allow_html=True)

    tabs = st.tabs([
        "☁️ Word Cloud", "📊 Word Frequency", "📈 TF-IDF Chart",
        "🏷️ POS Distribution", "🎯 Difficulty Chart"
    ])

    with tabs[0]:
        st.markdown("**Word Cloud** – visual representation of word importance (size ∝ frequency/importance).")
        img = plot_word_cloud(" ".join(pre["lemmas"]))
        if img:
            st.image(img, use_container_width=True)

    with tabs[1]:
        st.markdown("**Word Frequency Chart** – top lemmatized tokens by raw count.")
        img = plot_word_frequency(pre["lemmas"], top_n=20)
        if img:
            st.image(img, use_container_width=True)

    with tabs[2]:
        st.markdown("**TF-IDF Keyword Chart** – keywords ranked by TF-IDF score.")
        img = plot_tfidf_keywords(kw["tfidf_keywords"], top_n=15)
        if img:
            st.image(img, use_container_width=True)

    with tabs[3]:
        st.markdown("**POS Tag Distribution** – proportion of different grammatical categories.")
        img = plot_pos_distribution(pre["pos_tags"])
        if img:
            st.image(img, use_container_width=True)

    with tabs[4]:
        st.markdown("**Difficulty Distribution** – how many questions fall in each difficulty tier.")
        if qs:
            stats = get_difficulty_stats(qs)
            img   = plot_difficulty_distribution(stats)
            if img:
                st.image(img, use_container_width=False)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 6 – EXAM PAPER
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📄 Exam Paper":
    _require_processing()

    st.markdown('<p class="section-header">📄 Formatted Exam Paper</p>', unsafe_allow_html=True)

    # Regenerate on-the-fly if settings changed
    if st.session_state.annotated_qs:
        exam_text = format_exam_paper(
            st.session_state.annotated_qs,
            subject=st.session_state.subject_name,
            show_answers=show_ans,
        )
        key_text = format_answer_key(st.session_state.annotated_qs)
    else:
        exam_text = st.session_state.exam_paper
        key_text  = st.session_state.answer_key

    tabs = st.tabs(["📋 Exam Paper", "🗝️ Answer Key", "📝 Summary"])

    with tabs[0]:
        st.code(exam_text, language=None)
        st.download_button(
            label     = "⬇️ Download Exam Paper (.txt)",
            data      = exam_text,
            file_name = f"exam_paper_{datetime.date.today()}.txt",
            mime      = "text/plain",
            use_container_width=True,
        )

    with tabs[1]:
        st.code(key_text, language=None)
        st.download_button(
            label     = "⬇️ Download Answer Key (.txt)",
            data      = key_text,
            file_name = f"answer_key_{datetime.date.today()}.txt",
            mime      = "text/plain",
            use_container_width=True,
        )

    with tabs[2]:
        summary = st.session_state.summary
        if summary:
            st.markdown("### 📑 Short Summary")
            st.info(summary["short_summary"])

            st.markdown("### 📖 Medium Summary")
            st.success(summary["medium_summary"])

            st.markdown("### 🔑 Key Topics")
            topics_html = " &nbsp;·&nbsp; ".join(
                f"**{t}**" for t in summary["key_topics"]
            )
            st.markdown(topics_html)

            st.markdown("### 💡 Important Concepts")
            for c in summary["concept_list"]:
                with st.expander(f"**{c['concept']}**"):
                    st.markdown(c["definition"])


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 7 – SEARCH
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🔍 Search":
    _require_processing()

    st.markdown('<p class="section-header">🔍 Search Questions & Concepts</p>', unsafe_allow_html=True)
    query = st.text_input("🔎 Enter keyword or phrase to search:", placeholder="e.g. machine learning, neuron, algorithm")

    if query.strip():
        q_lower = query.lower()
        results = []

        qs = st.session_state.annotated_qs or st.session_state.questions or {}
        for qtype, question_list in qs.items():
            for q in question_list:
                if q_lower in q.get("question", "").lower() or q_lower in q.get("answer", "").lower():
                    results.append({"type": qtype, **q})

        st.markdown(f"**Found {len(results)} result(s) for '{query}'**")

        if results:
            for r in results:
                diff     = r.get("difficulty", "")
                qtype_label = r["type"].replace("_", " ").title()
                st.markdown(
                    f'<div class="q-card">'
                    f'<b>[{qtype_label}]</b> {_diff_badge(diff)}<br><br>'
                    f'{r["question"]}<br><br>'
                    f'<small>Answer: {str(r.get("answer",""))[:120]}…</small>'
                    f'</div>',
                    unsafe_allow_html=True
                )
        else:
            st.info("No matching questions found. Try a different keyword.")

        # Search in raw text
        st.markdown("---")
        st.markdown("**📄 Occurrences in Document Text:**")
        raw = st.session_state.raw_text
        lines = raw.split("\n")
        hits  = [(i+1, l) for i, l in enumerate(lines) if q_lower in l.lower()]
        if hits:
            for lineno, line in hits[:10]:
                highlighted = line.replace(
                    query, f"**:orange[{query}]**"
                )
                st.markdown(f"Line {lineno}: {highlighted}")
        else:
            st.info("Keyword not found in the raw document text.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 8 – NLP CONCEPTS EXPLAINER
# ═══════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ NLP Concepts":

    st.markdown('<p class="section-header">ℹ️ NLP Concepts Reference</p>', unsafe_allow_html=True)
    st.markdown("A quick-reference guide for every NLP concept implemented in this project. Perfect for viva preparation!")

    concepts = [
        ("1. Tokenization", """
**Definition:** Breaking raw text into individual tokens (words, subwords, or characters).

**Implementation:** NLTK `word_tokenize()` using Penn Treebank rules.

**Example:**
- Input:  `"Machine learning is amazing!"`
- Output: `['Machine', 'learning', 'is', 'amazing', '!']`

**Why it matters:** Every NLP operation works on tokens, not raw strings. Tokenization is the entry point of every NLP pipeline.
"""),
        ("2. Stopword Removal", """
**Definition:** Removing high-frequency, low-information words (the, is, at, which, …).

**Implementation:** NLTK `stopwords.words('english')` – 179 words.

**Why it matters:** Reduces vocabulary size by ~30-40%, helps models focus on content words.
"""),
        ("3. Lemmatization", """
**Definition:** Reducing a word to its lemma (dictionary base form).

**Implementation:** NLTK `WordNetLemmatizer` querying the WordNet lexical database.

**Examples:** running→run, better→good, studies→study

**Difference from Stemming:** Lemmatization produces real words; stemming just strips suffixes (runs→run, but studies→studi).
"""),
        ("4. POS Tagging", """
**Definition:** Assigning each token a grammatical label (Noun, Verb, Adjective, …).

**Implementation:** NLTK `pos_tag()` – averaged perceptron tagger trained on Penn Treebank.

**Common tags:** NN=Noun, VB=Verb, JJ=Adjective, RB=Adverb, IN=Preposition, DT=Determiner

**Uses in this project:** Identifying nouns for MCQ answer candidates; filtering content words.
"""),
        ("5. Named Entity Recognition (NER)", """
**Definition:** Detecting and classifying real-world entities (people, places, dates, orgs).

**Implementation:** spaCy `en_core_web_sm` NER pipeline component.

**Entity types:** PERSON, ORG, GPE (country/city), DATE, MONEY, LAW, NORP (nationalities).

**Uses in this project:** Generating entity-based questions (Who? Where? When?).
"""),
        ("6. TF-IDF", """
**Definition:** Term Frequency × Inverse Document Frequency – a numerical statistic reflecting word importance.

**Formula:**
```
TF(t,d)  = count(t in d) / total words in d
IDF(t)   = log( N / df(t) )    where N = num docs, df = doc frequency
TF-IDF   = TF × IDF
```

**Implementation:** scikit-learn `TfidfVectorizer` treating each sentence as a document.

**Uses in this project:** Keyword extraction, sentence scoring for summarization.
"""),
        ("7. Text Summarization", """
**Definition:** Condensing a long document into a shorter version preserving key information.

**Type implemented:** Extractive – selects the most important *existing* sentences.

**Algorithm:**
1. Score sentences by TF-IDF weight sum
2. Normalize by sentence length
3. Return top-k sentences in original order

**Alternative (not implemented):** Abstractive summarization (T5/BART) paraphrases content.
"""),
        ("8. Question Generation", """
**Definition:** Automatically creating quiz questions from text.

**Approach used:** Rule-based NLP (no GPU required)
- Select keyword-rich sentences (scored by TF-IDF + NER)
- Transform sentences using wh-question templates
- Mask keywords for fill-in-the-blank
- Generate distractors from document noun pool

**Limitation:** Rule-based systems can't match GPT-4 quality but are explainable and fast.
"""),
        ("9. Difficulty Classification", """
**Definition:** Labelling each question as Easy, Medium, or Hard.

**Features used:**
1. **Length score** (0-3): Longer questions are harder.
2. **Vocabulary score** (0-4): Long words, low stopword ratio, domain terminology.
3. **Syntactic score** (0-3): Subordinating conjunctions, comma clauses.

**Thresholds:** Score<3.5=Easy, 3.5-6.5=Medium, >6.5=Hard.
"""),
        ("10. Semantic Similarity", """
**Definition:** Measuring how similar two pieces of text are in meaning.

**Approach:** Cosine similarity between TF-IDF vectors.

**Uses in this project:** Distractor diversity – we avoid distractors too similar to the correct answer, and too similar to each other.

**Advanced alternative:** `sentence-transformers` (BERT embeddings) for richer semantic comparison.
"""),
    ]

    for title, body in concepts:
        with st.expander(title, expanded=False):
            st.markdown(body)

    st.markdown("---")
    st.markdown("### 🎤 Viva Q&A Cheat Sheet")
    viva_qs = [
        ("What is the NLP pipeline in this project?",
         "Raw Text → Tokenization → Stopword Removal → Lemmatization → POS Tagging → TF-IDF → NER → Question Generation → Difficulty Classification → Formatted Exam Paper."),
        ("Why use TF-IDF instead of raw frequency?",
         "Raw frequency rewards common words (the, is). TF-IDF penalizes words that appear in every document via IDF, highlighting words unique to the current document."),
        ("What is the difference between stemming and lemmatization?",
         "Stemming blindly strips suffixes (studies→studi – not a real word). Lemmatization uses a dictionary (WordNet) to find the true base form (studies→study)."),
        ("How do you generate MCQ distractors?",
         "We collect all nouns, noun-phrases, and named entities from the document, rank by frequency, exclude the correct answer, and pick the top 3 as distractors. They're topically related but factually wrong."),
        ("What is Named Entity Recognition?",
         "NER is a sequence labelling task where a model identifies and classifies named objects in text into categories like PERSON, ORG, GPE, DATE. We use spaCy's pre-trained en_core_web_sm model."),
        ("How is difficulty classified?",
         "Three signals: (1) question length, (2) vocabulary complexity (avg word length + domain terms), (3) syntactic depth (subordinating conjunctions). A composite score maps to Easy/Medium/Hard."),
        ("What is extractive summarization?",
         "Selecting the most informative existing sentences from the document, rather than generating new text. We score sentences by TF-IDF weight sum normalized by length."),
        ("Why use Streamlit?",
         "Streamlit converts Python scripts into interactive web apps with zero HTML/CSS/JS. It's ideal for ML/NLP demos because every widget re-runs the script automatically on change."),
    ]
    for q, a in viva_qs:
        with st.expander(f"❓ {q}"):
            st.success(a)
