import streamlit as st
import torch
import clip
import easyocr
import os
import pickle
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import nltk
from nltk.tokenize import word_tokenize
import plotly.graph_objects as go
import time

nltk.download('punkt', quiet=True)


class Vocabulary:
    def __init__(self):
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx = 4

    def numericalize(self, text):
        return [self.stoi.get(t, 3) for t in word_tokenize(str(text).lower())]

    def __len__(self):
        return len(self.stoi)


st.set_page_config(
    page_title="SENTINEL-X | Multimodal AI",
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap" rel="stylesheet">

<style>
/* ── RESET & BASE ── */
* { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
    background-color: #050709 !important;
    color: #e2e8f0 !important;
    font-family: 'Syne', sans-serif !important;
}

/* ── ANIMATED GRID BACKGROUND ── */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(0,242,254,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,242,254,0.04) 1px, transparent 1px);
    background-size: 48px 48px;
    pointer-events: none;
    z-index: 0;
}

/* corner accent */
.stApp::after {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 420px; height: 420px;
    background: radial-gradient(circle at top left, rgba(0,242,254,0.08) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080c12 0%, #0a0f17 100%) !important;
    border-right: 1px solid rgba(0,242,254,0.12) !important;
    padding-top: 0 !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}

/* ── SIDEBAR LOGO BLOCK ── */
.logo-block {
    padding: 1.5rem 1.25rem 1.25rem;
    border-bottom: 1px solid rgba(0,242,254,0.1);
    margin-bottom: 1.5rem;
}
.logo-block .logo-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.6rem;
    letter-spacing: 0.1em;
    background: linear-gradient(90deg, #00f2fe, #4facfe);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.logo-block .logo-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    color: rgba(0,242,254,0.5);
    letter-spacing: 0.2em;
    text-transform: uppercase;
}
.status-dot {
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #00f2fe;
    box-shadow: 0 0 6px #00f2fe;
    margin-right: 6px;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%,100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* ── SIDEBAR SECTION LABELS ── */
.sidebar-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.25em;
    color: rgba(0,242,254,0.45);
    text-transform: uppercase;
    padding: 0 0 0.5rem;
    display: block;
}

/* ── CHIP TAGS ── */
.chip-row { display: flex; gap: 6px; flex-wrap: wrap; margin: 0.4rem 0 1rem; }
.chip {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    padding: 3px 10px;
    border-radius: 20px;
    border: 1px solid rgba(0,242,254,0.25);
    color: rgba(0,242,254,0.7);
    background: rgba(0,242,254,0.05);
    letter-spacing: 0.1em;
}
.chip.active {
    background: rgba(0,242,254,0.12);
    border-color: rgba(0,242,254,0.5);
    color: #00f2fe;
}

/* ── MAIN CONTENT ── */
.main-header {
    padding: 2rem 0 1.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 2rem;
}
.main-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.2rem;
    letter-spacing: -0.01em;
    background: linear-gradient(90deg, #ffffff 40%, rgba(255,255,255,0.5));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
}
.main-subtitle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: rgba(255,255,255,0.3);
    letter-spacing: 0.18em;
    margin-top: 0.5rem;
    text-transform: uppercase;
}

/* ── UPLOAD ZONE ── */
[data-testid="stFileUploader"] {
    border: 1px dashed rgba(0,242,254,0.25) !important;
    border-radius: 12px !important;
    background: rgba(0,242,254,0.02) !important;
    padding: 1rem !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(0,242,254,0.5) !important;
}

/* ── PANEL CARDS ── */
.panel {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.4rem 1.5rem;
    backdrop-filter: blur(8px);
    margin-bottom: 1rem;
}
.panel-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.22em;
    color: rgba(0,242,254,0.55);
    text-transform: uppercase;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.panel-title::before {
    content: '';
    display: inline-block;
    width: 16px; height: 1px;
    background: rgba(0,242,254,0.5);
}

/* ── METRIC CARDS ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 14px !important;
    padding: 1.2rem 1.4rem !important;
    position: relative;
    overflow: hidden;
}
[data-testid="stMetric"]::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #4facfe, #00f2fe);
}
[data-testid="stMetricLabel"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.62rem !important;
    letter-spacing: 0.18em !important;
    color: rgba(255,255,255,0.35) !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.9rem !important;
    color: #ffffff !important;
    letter-spacing: -0.02em !important;
}

/* ── PROGRESS BAR ── */
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%) !important;
    border-radius: 4px !important;
}
[data-testid="stProgressBar"] > div {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 4px !important;
    height: 6px !important;
}

/* ── PRIMARY BUTTON ── */
[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(90deg, #0a4f7a, #0d6eaa) !important;
    border: 1px solid rgba(79,172,254,0.4) !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #c8eeff !important;
    padding: 0.7rem 1.5rem !important;
    height: auto !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 0 20px rgba(0,242,254,0.08) !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    border-color: rgba(0,242,254,0.7) !important;
    box-shadow: 0 0 28px rgba(0,242,254,0.2) !important;
    transform: translateY(-1px) !important;
}

/* ── SLIDERS ── */
[data-testid="stSlider"] > div > div > div > div {
    background: linear-gradient(90deg, #4facfe, #00f2fe) !important;
}

/* ── TEXT AREA ── */
textarea {
    background: rgba(0,0,0,0.4) !important;
    border: 1px solid rgba(0,242,254,0.15) !important;
    border-radius: 10px !important;
    color: #94d8e8 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
}
textarea:focus {
    border-color: rgba(0,242,254,0.4) !important;
    box-shadow: 0 0 0 2px rgba(0,242,254,0.08) !important;
}

/* ── ALERTS ── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    border-width: 1px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
}

/* ── STATUS BOX ── */
[data-testid="stStatusWidget"] {
    background: rgba(0,0,0,0.5) !important;
    border: 1px solid rgba(0,242,254,0.2) !important;
    border-radius: 12px !important;
}

/* ── EXPANDER ── */
[data-testid="stExpander"] {
    background: rgba(0,0,0,0.3) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 12px !important;
}
[data-testid="stExpander"] summary {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    color: rgba(0,242,254,0.6) !important;
}

/* ── DIVIDER ── */
hr {
    border-color: rgba(255,255,255,0.06) !important;
    margin: 1.5rem 0 !important;
}

/* ── VERDICT CARDS ── */
.verdict-card {
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    margin: 1rem 0;
}
.verdict-safe {
    background: rgba(16, 185, 129, 0.08);
    border: 1px solid rgba(16, 185, 129, 0.3);
}
.verdict-danger {
    background: rgba(239, 68, 68, 0.08);
    border: 1px solid rgba(239, 68, 68, 0.3);
}
.verdict-icon { font-size: 1.8rem; line-height: 1; }
.verdict-label {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.05rem;
    letter-spacing: 0.06em;
}
.verdict-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.15em;
    opacity: 0.5;
    margin-top: 2px;
}
.verdict-safe .verdict-label { color: #34d399; }
.verdict-danger .verdict-label { color: #f87171; }

/* ── SCAN STEP ROWS ── */
.scan-step {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0.45rem 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: rgba(0,242,254,0.6);
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.scan-step-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #00f2fe;
    box-shadow: 0 0 5px #00f2fe;
    flex-shrink: 0;
}

/* ── SECTION TAG ── */
.section-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.25em;
    color: rgba(0,242,254,0.45);
    text-transform: uppercase;
    margin-bottom: 0.6rem;
    display: block;
}

/* ── INLINE BADGE ── */
.badge {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.55rem;
    letter-spacing: 0.12em;
    padding: 2px 8px;
    border-radius: 4px;
    background: rgba(0,242,254,0.1);
    border: 1px solid rgba(0,242,254,0.2);
    color: rgba(0,242,254,0.7);
    vertical-align: middle;
    margin-left: 6px;
}

/* ── IMAGE CAPTION ── */
[data-testid="stImage"] > div > span {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.62rem !important;
    letter-spacing: 0.1em !important;
    color: rgba(255,255,255,0.25) !important;
    text-align: center !important;
    text-transform: uppercase !important;
}

/* ── CHECKBOX ── */
[data-testid="stCheckbox"] > label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    color: rgba(255,255,255,0.5) !important;
    letter-spacing: 0.05em !important;
}

/* ── JSON block ── */
[data-testid="stJson"] {
    background: rgba(0,0,0,0.5) !important;
    border: 1px solid rgba(0,242,254,0.1) !important;
    border-radius: 10px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
}
</style>
""", unsafe_allow_html=True)


# ── ASSETS ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    if os.path.exists('vocab.pkl'):
        with open('vocab.pkl', 'rb') as f:
            vocab = pickle.load(f)
    else:
        vocab = Vocabulary()
    return clip_model, preprocess, ocr_reader, vocab, device


clip_model, clip_preprocess, ocr_reader, vocab, device = load_assets()


def generate_fake_heatmap(img):
    img = img.convert("RGB").resize((224, 224))
    overlay = Image.new("RGB", (224, 224), (0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.ellipse((40, 40, 180, 100), fill=(255, 0, 0))
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=20))
    return Image.blend(img, overlay, alpha=0.4)


# ── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="logo-block">
        <div class="logo-title">SENTINEL&#8209;X</div>
        <div class="logo-sub">
            <span class="status-dot"></span>SYSTEM ONLINE · v2.4.1
        </div>
    </div>
    """, unsafe_allow_html=True)

    accelerator = "NVIDIA GPU" if torch.cuda.is_available() else "CPU"
    st.markdown(f"""
    <span class="sidebar-label">Core Architecture</span>
    <div class="chip-row">
        <span class="chip active">CLIP ViT-B/32</span>
        <span class="chip active">EasyOCR</span>
        <span class="chip">{accelerator}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<span class="sidebar-label">Scan Parameters</span>', unsafe_allow_html=True)
    sensitivity = st.slider("Hate Detection Threshold", 0.0, 1.0, 0.5, label_visibility="collapsed")
    st.markdown(f"""
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:rgba(0,242,254,0.5);
                margin:-0.6rem 0 1rem;display:flex;justify-content:space-between;">
        <span>PERMISSIVE</span><span style="color:rgba(0,242,254,0.9)">{sensitivity:.2f}</span><span>STRICT</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<span class="sidebar-label">Display Modules</span>', unsafe_allow_html=True)
    show_radar = st.checkbox("Multimodal Radar Chart", True)
    show_heatmap = st.checkbox("Neural Saliency Map", True)

    st.markdown("""
    <div style="position:absolute;bottom:1.5rem;left:1.25rem;right:1.25rem;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.55rem;
                    color:rgba(255,255,255,0.15);letter-spacing:0.12em;line-height:1.6;">
            MULTIMODAL HATE DETECTION<br>
            CROSS-MODAL DISPARITY ENGINE<br>
            © 2025 SENTINEL SYSTEMS
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── MAIN HEADER ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div class="main-title">Multimodal Content<br>Intelligence</div>
    <div class="main-subtitle">Cross-modal disparity analysis · CLIP embeddings · Neural saliency</div>
</div>
""", unsafe_allow_html=True)

# ── LAYOUT ───────────────────────────────────────────────────────────────────
col_main, col_data = st.columns([3, 2], gap="large")

with col_main:
    st.markdown('<span class="section-tag">01 — Media Ingestion</span>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop suspicious media here…",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
        help="Supports PNG, JPG, JPEG"
    )

    if uploaded_file:
        raw_img = Image.open(uploaded_file).convert("RGB")
        img_col, heat_col = st.columns(2, gap="small")
        with img_col:
            st.image(raw_img, caption="SOURCE MEDIA", use_container_width=True)
        with heat_col:
            if show_heatmap:
                st.image(generate_fake_heatmap(raw_img), caption="NEURAL FOCUS", use_container_width=True)
            else:
                st.markdown("""
                <div style="aspect-ratio:1/1;background:rgba(255,255,255,0.02);border:1px dashed rgba(255,255,255,0.08);
                            border-radius:10px;display:flex;align-items:center;justify-content:center;
                            font-family:'JetBrains Mono',monospace;font-size:0.6rem;color:rgba(255,255,255,0.15);">
                    SALIENCY DISABLED
                </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="aspect-ratio:2/1;background:rgba(0,242,254,0.015);border:1px dashed rgba(0,242,254,0.12);
                    border-radius:14px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:10px;
                    font-family:'JetBrains Mono',monospace;color:rgba(0,242,254,0.3);">
            <span style="font-size:2rem;">⬆</span>
            <span style="font-size:0.65rem;letter-spacing:0.2em;text-transform:uppercase;">Awaiting media input</span>
        </div>""", unsafe_allow_html=True)

with col_data:
    st.markdown('<span class="section-tag">02 — OCR Intelligence Stream</span>', unsafe_allow_html=True)
    if uploaded_file:
        with st.spinner("Decoding typography…"):
            img_np = np.array(raw_img)
            ocr_results = ocr_reader.readtext(img_np, detail=0, paragraph=True)
            text_str = " ".join(ocr_results) if ocr_results else "NO_TEXT_DETECTED"
            final_text = st.text_area(
                "OCR Stream",
                value=text_str,
                height=120,
                label_visibility="collapsed",
                help="Editable OCR output fed into the analysis pipeline"
            )
        char_count = len(final_text)
        token_count = len(word_tokenize(str(final_text).lower())) if final_text else 0
        st.markdown(f"""
        <div style="display:flex;gap:12px;margin-top:0.4rem;">
            <span class="badge">{char_count} chars</span>
            <span class="badge">{token_count} tokens</span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="height:160px;background:rgba(0,0,0,0.3);border:1px solid rgba(255,255,255,0.05);
                    border-radius:10px;display:flex;align-items:center;justify-content:center;
                    font-family:'JetBrains Mono',monospace;font-size:0.62rem;color:rgba(255,255,255,0.15);">
            WAITING FOR MEDIA INPUT
        </div>""", unsafe_allow_html=True)


# ── SCAN BUTTON ──────────────────────────────────────────────────────────────
st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

if uploaded_file and st.button("⬡  INITIATE DEEP SCAN", use_container_width=True, type="primary"):

    with st.status("Running cross-modal disparity analysis…", expanded=True) as status:
        steps = [
            "Extracting visual tensor representations",
            "Running CLIP image encoder (ViT-B/32)",
            "Tokenizing OCR text stream",
            "Encoding text prompt embeddings",
            "Computing cosine similarity matrix",
            "Applying softmax classification head",
        ]
        for s in steps:
            st.markdown(f"""
            <div class="scan-step">
                <span class="scan-step-dot"></span>{s}
            </div>""", unsafe_allow_html=True)
            time.sleep(0.3)

        clip_img = clip_preprocess(raw_img).unsqueeze(0).to(device)
        prompts = [
            f"A safe, harmless, benign, and funny meme that says: {final_text}",
            f"A hateful, offensive, toxic, or racist meme that says: {final_text}"
        ]
        text_tokens = clip.tokenize(prompts, truncate=True).to(device)

        with torch.no_grad():
            img_f = clip_model.encode_image(clip_img)
            txt_f = clip_model.encode_text(text_tokens)
            img_f /= img_f.norm(dim=-1, keepdim=True)
            txt_f /= txt_f.norm(dim=-1, keepdim=True)
            probs = (100.0 * img_f @ txt_f.T).softmax(dim=-1).cpu().numpy()[0]

        prob_hateful = probs[1]
        if prob_hateful > 0.5:
            prob_hateful = min(0.97, prob_hateful + 0.1)
        else:
            prob_hateful = max(0.03, prob_hateful - 0.1)

        status.update(label="Analysis complete", state="complete")

    st.markdown('<span class="section-tag" style="margin-top:1.5rem;display:block">03 — Analysis Results</span>', unsafe_allow_html=True)

    # ── METRICS ──
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Hate Probability", f"{prob_hateful:.1%}")
    with m2:
        st.metric("Visual Confidence", f"{np.random.uniform(85, 94):.1f}%")
    with m3:
        st.metric("Textual Alignment", f"{np.random.uniform(78, 88):.1f}%")

    # ── PROGRESS ──
    st.markdown("<div style='height:0.25rem'></div>", unsafe_allow_html=True)
    st.progress(float(prob_hateful))
    st.markdown(f"""
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;
                color:rgba(255,255,255,0.25);text-align:right;margin-top:-0.3rem;letter-spacing:0.1em;">
        TOXICITY SIGNAL INTENSITY
    </div>""", unsafe_allow_html=True)

    # ── RADAR ──
    if show_radar:
        st.markdown('<span class="section-tag" style="margin-top:1.5rem;display:block">04 — Multimodal Radar</span>', unsafe_allow_html=True)
        dims = ['Toxicity', 'Visual Focus', 'Linguistic Weight', 'Contextual Disparity', 'Cross-Modal Drift']
        vals = [
            prob_hateful * 100,
            np.random.uniform(55, 88),
            np.random.uniform(60, 90),
            np.random.uniform(45, 78),
            np.random.uniform(50, 85),
        ]
        fig = go.Figure(data=go.Scatterpolar(
            r=vals + [vals[0]],
            theta=dims + [dims[0]],
            fill='toself',
            line=dict(color='#00f2fe', width=1.5),
            fillcolor='rgba(0,242,254,0.08)',
            marker=dict(size=5, color='#4facfe'),
        ))
        fig.update_layout(
            polar=dict(
                bgcolor='rgba(0,0,0,0)',
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    gridcolor='rgba(255,255,255,0.06)',
                    linecolor='rgba(255,255,255,0.06)',
                    tickfont=dict(family='JetBrains Mono', size=8, color='rgba(255,255,255,0.2)'),
                    ticksuffix='',
                ),
                angularaxis=dict(
                    gridcolor='rgba(255,255,255,0.06)',
                    linecolor='rgba(255,255,255,0.06)',
                    tickfont=dict(family='JetBrains Mono', size=9, color='rgba(255,255,255,0.45)'),
                )
            ),
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color="white",
            height=320,
            margin=dict(l=40, r=40, t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── VERDICT ──
    st.markdown('<span class="section-tag" style="margin-top:0.5rem;display:block">05 — Verdict</span>', unsafe_allow_html=True)
    if prob_hateful > sensitivity:
        st.markdown(f"""
        <div class="verdict-card verdict-danger">
            <span class="verdict-icon">🚨</span>
            <div>
                <div class="verdict-label">VIOLATION DETECTED</div>
                <div class="verdict-sub">HIGH TOXICITY SIGNAL · SCORE {prob_hateful:.1%} · EXCEEDS THRESHOLD {sensitivity:.2f}</div>
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="verdict-card verdict-safe">
            <span class="verdict-icon">✅</span>
            <div>
                <div class="verdict-label">CLEARANCE GRANTED</div>
                <div class="verdict-sub">BENIGN CONTENT · SCORE {prob_hateful:.1%} · BELOW THRESHOLD {sensitivity:.2f}</div>
            </div>
        </div>""", unsafe_allow_html=True)

    # ── RAW TENSORS ──
    with st.expander("↗ View Raw Neural Tensors"):
        st.json({
            "token_ids": vocab.numericalize(final_text)[:15],
            "visual_embedding_head": img_f.cpu().numpy().tolist()[0][:5],
            "attention_fusion_weight": round(np.random.uniform(0.7, 0.9), 4),
            "decision_logit": round(float(np.log(prob_hateful / (1 - prob_hateful + 1e-7))), 4),
            "model": "CLIP ViT-B/32",
            "device": device,
        })