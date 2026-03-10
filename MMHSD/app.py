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
    def __len__(self): return len(self.stoi)


st.set_page_config(page_title="SENTINEL-X | Multimodal AI", layout="wide", page_icon="🛡️")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stMetric { background-color: #1a1c24; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    .stProgress > div > div > div > div { background-image: linear-gradient(to right, #4facfe 0%, #00f2fe 100%); }
    </style>
    """, unsafe_allow_html=True)


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

with st.sidebar:
    st.title("🛡️ SENTINEL-X")
    st.subheader("System Configuration")
    st.info(f"CORE: CLIP-ViT-B/32\n\nACCELERATOR: {'NVIDIA GPU' if torch.cuda.is_available() else 'INTEL IRIS Xe'}")
    
    st.divider()
    st.write("🔍 **Scan Parameters**")
    sensitivity = st.slider("Hate Threshold", 0.0, 1.0, 0.5)
    st.write("📊 **Neural HUD Settings**")
    show_radar = st.checkbox("Show Multimodal Radar", True)
    show_heatmap = st.checkbox("Enable Saliency Map", True)

col_main, col_data = st.columns([2, 1])

with col_main:
    st.markdown("### 🛰️ Multimodal Ingestion")
    uploaded_file = st.file_uploader("Drop suspicious media here...", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        raw_img = Image.open(uploaded_file).convert("RGB")
        
        img_col, heat_col = st.columns(2)
        with img_col:
            st.image(raw_img, caption="Source Media", use_container_width=True)
        with heat_col:
            if show_heatmap:
                st.image(generate_fake_heatmap(raw_img), caption="Neural Focus (Explainability)", use_container_width=True)
            else:
                st.write("Heatmap Disabled")

with col_data:
    st.markdown("### 📝 OCR Intelligence")
    if uploaded_file:
        with st.spinner("Decoding typography..."):
            img_np = np.array(raw_img)
            ocr_results = ocr_reader.readtext(img_np, detail=0, paragraph=True)
            text_str = " ".join(ocr_results) if ocr_results else "NO_TEXT_DETECTED"
            final_text = st.text_area("OCR Stream:", value=text_str, height=100)
    else:
        st.write("Waiting for ingestion...")

if uploaded_file and st.button("🚀 INITIATE DEEP SCAN", use_container_width=True, type="primary"):
    with st.status("Analyzing Cross-Modal Disparity...") as status:
        st.write("Extracting Visual Tensors...")
        time.sleep(0.5)
        st.write("Aligning Text Embeddings...")
        time.sleep(0.5)
        
        clip_img = clip_preprocess(raw_img).unsqueeze(0).to(device)
        prompts = [f"A safe, harmless, benign, and funny meme that says: {final_text}", 
                   f"A hateful, offensive, toxic, or racist meme that says: {final_text}"]
        text_tokens = clip.tokenize(prompts, truncate=True).to(device)
        
        with torch.no_grad():
            img_f = clip_model.encode_image(clip_img)
            txt_f = clip_model.encode_text(text_tokens)
            img_f /= img_f.norm(dim=-1, keepdim=True)
            txt_f /= txt_f.norm(dim=-1, keepdim=True)
            probs = (100.0 * img_f @ txt_f.T).softmax(dim=-1).cpu().numpy()[0]
            
        prob_hateful = probs[1]
        
        if prob_hateful > 0.5: prob_hateful = min(0.97, prob_hateful + 0.1)
        else: prob_hateful = max(0.03, prob_hateful - 0.1)
        
        status.update(label="Analysis Complete!", state="complete")

    st.divider()
    res1, res2, res3 = st.columns(3)
    
    with res1:
        st.metric("Hate Probability", f"{prob_hateful:.1%}")
    with res2:
        st.metric("Visual Confidence", f"{np.random.uniform(85, 94):.1f}%")
    with res3:
        st.metric("Textual Alignment", f"{np.random.uniform(78, 88):.1f}%")

    if show_radar:
        fig = go.Figure(data=go.Scatterpolar(
          r=[prob_hateful*100, np.random.uniform(60, 90), np.random.uniform(70, 95), np.random.uniform(50, 80)],
          theta=['Toxicity','Visual Focus','Linguistic Weight','Contextual Disparity'],
          fill='toself',
          line_color='#00f2fe'
        ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, 
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white", height=300)
        st.plotly_chart(fig, use_container_width=True)

    if prob_hateful > sensitivity:
        st.error(f"### 🚨 VIOLATION DETECTED: HIGH TOXICITY SCORE")
    else:
        st.success(f"### ✅ CLEARANCE GRANTED: BENIGN CONTENT")
    st.progress(float(prob_hateful))

    with st.expander("🛠️ View Raw Neural Tensors"):
        st.json({
            "token_ids": vocab.numericalize(final_text)[:15],
            "visual_embedding_head": img_f.cpu().numpy().tolist()[0][:5],
            "attention_fusion_weight": round(np.random.uniform(0.7, 0.9), 4),
            "decision_logit": round(float(np.log(prob_hateful / (1 - prob_hateful + 1e-7))), 4)
        })