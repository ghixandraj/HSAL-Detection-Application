import streamlit as st
import re
import torch
import torch.nn as nn
import numpy as np
import os
import gdown
import requests
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModel
from safetensors.torch import load_file
import time
import streamlit.components.v1 as components

# ✅ Konfigurasi halaman
st.set_page_config(page_title="Hayu-IT: HSAL Analysis", page_icon="🧠", layout="wide")

# CSS Styling dengan tema 8-bit retro
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap" rel="stylesheet">
<style>
    /* Menyembunyikan tombol collapse sidebar bawaan Streamlit */
    button[title="Collapse sidebar"] {
        display: none !important;
    }
    
    button[title="Expand sidebar"] {
        display: none !important;
    }
            
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    
    /* Menyembunyikan tombol sidebar ketika di-hover */
    [data-testid="stSidebar"]:hover button[title="Collapse sidebar"] {
        display: none !important;
    }
    
    [data-testid="stSidebar"]:hover button[title="Expand sidebar"] {
        display: none !important;
    }
    
    /* Menyembunyikan semua tombol yang berkaitan dengan sidebar control */
    [data-testid="stSidebar"] button[kind="header"] {
        display: none !important;
    }
    
    [data-testid="stSidebar"] button[data-testid="baseButton-header"] {
        display: none !important;
    }
    
    /* Menyembunyikan elemen control sidebar lainnya */
    [data-testid="stSidebar"] > div > div:first-child button {
        display: none !important;
    }

    /* Background animasi 8-bit */
    @keyframes pixelMove {
        0% { background-position: 0 0; }
        100% { background-position: 32px 32px; }
    }

    @keyframes neonGlow {
        0%, 100% { text-shadow: 0 0 5px #00ff00, 0 0 10px #00ff00, 0 0 15px #00ff00; }
        50% { text-shadow: 0 0 2px #00ff00, 0 0 5px #00ff00, 0 0 8px #00ff00; }
    }

    @keyframes buttonBlink {
        0%, 50% { background-color: #ff0040; }
        51%, 100% { background-color: #ff4080; }
    }

    html, body, [class*="css"] {
        font-family: 'Press Start 2P', monospace !important;
        background: linear-gradient(45deg, #0a0a0a 25%, #1a1a1a 25%, #1a1a1a 50%, #0a0a0a 50%, #0a0a0a 75%, #1a1a1a 75%, #1a1a1a);
        background-size: 32px 32px;
        animation: pixelMove 2s linear infinite;
        color: #00ff00;
        overflow-x: hidden;
    }

    /* Sidebar styling 8-bit */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2a2a2a 0%, #1a1a1a 100%) !important;
        border-right: 4px solid #00ff00 !important;
        box-shadow: 
            4px 0 0 #004400,
            8px 0 0 #002200,
            inset -4px 0 0 #00ff00 !important;
        transition: transform 0.3s cubic-bezier(0.68, -0.55, 0.265, 1.55), opacity 0.3s ease;
        transform: translateX(0);
        opacity: 1;
        position: relative;
        z-index: 1;
    }

    [data-testid="stSidebar"].sidebar-hidden {
        transform: translateX(-100%);
        opacity: 0;
        pointer-events: none;
    }

    [data-testid="stSidebar"] + div {
        transition: all 0.3s cubic-bezier(0.68, -0.55, 0.265, 1.55);
    }

    [data-testid="stSidebar"].sidebar-hidden + div {
        margin-left: 0 !important;
    }

    /* Sidebar content styling */
    [data-testid="stSidebar"] .css-1d391kg {
        background: transparent !important;
    }

    [data-testid="stSidebar"] h2 {
        color: #ffff00 !important;
        text-shadow: 2px 2px 0px #666600;
        border-bottom: 2px solid #ffff00;
        padding-bottom: 8px;
        margin-bottom: 16px;
        font-size: 14px !important;
    }

    [data-testid="stSidebar"] p, [data-testid="stSidebar"] li {
        color: #00ff00 !important;
        font-size: 10px !important;
        line-height: 1.6 !important;
    }

    /* Header box dengan tema arcade */
    .header-box {
        background: linear-gradient(135deg, #ff0040, #ff4080, #8040ff, #4080ff);
        border: 6px solid #ffffff;
        border-radius: 0px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 
            0 0 0 4px #ff0040,
            0 0 0 8px #ffffff,
            8px 8px 0 8px #000000;
        position: relative;
        overflow: hidden;
    }

    .header-box::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: repeating-linear-gradient(
            45deg,
            transparent,
            transparent 2px,
            rgba(255, 255, 255, 0.1) 2px,
            rgba(255, 255, 255, 0.1) 4px
        );
        animation: pixelMove 1s linear infinite;
    }

    .main-title {
        font-size: 1.8rem;
        font-weight: normal;
        color: #ffffff;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 
            4px 4px 0px #000000,
            2px 2px 0px #ff0040;
        animation: neonGlow 2s ease-in-out infinite;
        position: relative;
        z-index: 1;
    }

    .subtitle {
        font-size: 0.7rem;
        text-align: center;
        color: #ffff00;
        text-shadow: 2px 2px 0px #000000;
        position: relative;
        z-index: 1;
    }

    /* Toggle Button dengan style arcade */
    .toggle-btn {
        position: fixed;
        top: 20px;
        left: 20px;
        background: #ff0040;
        color: #ffffff;
        font-family: 'Press Start 2P', monospace;
        font-size: 12px;
        font-weight: normal;
        border: 4px solid #ffffff;
        cursor: pointer;
        z-index: 9999;
        padding: 8px 12px;
        border-radius: 0px;
        transition: all 0.2s ease;
        box-shadow: 
            0 0 0 2px #ff0040,
            4px 4px 0 2px #000000;
        text-shadow: 1px 1px 0px #000000;
        animation: buttonBlink 1s ease-in-out infinite;
    }

    .toggle-btn:hover {
        background: #ff4080;
        transform: translate(-2px, -2px);
        box-shadow: 
            0 0 0 2px #ff4080,
            6px 6px 0 2px #000000;
    }

    .toggle-btn:active {
        transform: translate(2px, 2px);
        box-shadow: 
            0 0 0 2px #ff0040,
            2px 2px 0 2px #000000;
    }

    /* Styling untuk input dan button */
    .stTextInput input {
        background: #000000 !important;
        border: 3px solid #00ff00 !important;
        border-radius: 0px !important;
        color: #00ff00 !important;
        font-family: 'Press Start 2P', monospace !important;
        font-size: 12px !important;
        padding: 12px !important;
        box-shadow: inset 2px 2px 0px #004400 !important;
    }

    .stTextInput input:focus {
        border-color: #ffff00 !important;
        box-shadow: 
            inset 2px 2px 0px #666600,
            0 0 10px #ffff00 !important;
    }

    .stButton button {
        background: linear-gradient(180deg, #00ff00, #008800) !important;
        border: 4px solid #ffffff !important;
        border-radius: 0px !important;
        color: #000000 !important;
        font-family: 'Press Start 2P', monospace !important;
        font-size: 12px !important;
        padding: 12px 24px !important;
        text-shadow: 1px 1px 0px #ffffff !important;
        box-shadow: 
            0 0 0 2px #00ff00,
            4px 4px 0 2px #000000 !important;
        transition: all 0.2s ease !important;
    }

    .stButton button:hover {
        background: linear-gradient(180deg, #40ff40, #00aa00) !important;
        transform: translate(-2px, -2px) !important;
        box-shadow: 
            0 0 0 2px #40ff40,
            6px 6px 0 2px #000000 !important;
    }

    .stButton button:active {
        transform: translate(2px, 2px) !important;
        box-shadow: 
            0 0 0 2px #00ff00,
            2px 2px 0 2px #000000 !important;
    }

    /* Progress bar 8-bit */
    .stProgress > div > div {
        background: #ff0040 !important;
        border: 2px solid #ffffff !important;
        border-radius: 0px !important;
        height: 24px !important;
        box-shadow: inset 0 0 0 2px #000000 !important;
    }

    .stProgress > div > div > div {
        background: repeating-linear-gradient(
            90deg,
            #00ff00,
            #00ff00 4px,
            #40ff40 4px,
            #40ff40 8px
        ) !important;
        border-radius: 0px !important;
        animation: pixelMove 0.5s linear infinite !important;
    }

    /* Alert dan info boxes */
    .stAlert {
        border: 3px solid #ffff00 !important;
        border-radius: 0px !important;
        background: #1a1a00 !important;
        color: #ffff00 !important;
        font-family: 'Press Start 2P', monospace !important;
        font-size: 10px !important;
        box-shadow: 4px 4px 0px #000000 !important;
    }

    .stError {
        border-color: #ff0040 !important;
        background: #1a0008 !important;
        color: #ff0040 !important;
    }

    .stSuccess {
        border-color: #00ff00 !important;
        background: #001a00 !important;
        color: #00ff00 !important;
    }

    /* Video player styling */
    .stVideo {
        border: 4px solid #ffffff !important;
        border-radius: 0px !important;
        box-shadow: 
            0 0 0 2px #000000,
            8px 8px 0 2px #666666 !important;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background: #000000 !important;
        border: 2px solid #00ff00 !important;
        border-radius: 0px !important;
        color: #00ff00 !important;
        font-family: 'Press Start 2P', monospace !important;
        font-size: 10px !important;
    }

    .streamlit-expanderContent {
        background: #0a0a0a !important;
        border: 2px solid #004400 !important;
        border-top: none !important;
        border-radius: 0px !important;
    }

    /* Textarea styling */
    .stTextArea textarea {
        background: #000000 !important;
        border: 3px solid #00ff00 !important;
        border-radius: 0px !important;
        color: #00ff00 !important;
        font-family: 'Press Start 2P', monospace !important;
        font-size: 8px !important;
        line-height: 1.4 !important;
    }

    /* Spinner 8-bit */
    .stSpinner {
        border: 4px solid #000000 !important;
        border-top: 4px solid #00ff00 !important;
        border-radius: 0px !important;
        width: 32px !important;
        height: 32px !important;
        animation: spin 1s linear infinite !important;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    /* Scrollbar 8-bit */
    ::-webkit-scrollbar {
        width: 16px;
        background: #000000;
    }

    ::-webkit-scrollbar-track {
        background: #000000;
        border: 2px solid #004400;
    }

    ::-webkit-scrollbar-thumb {
        background: #00ff00;
        border: 2px solid #000000;
        border-radius: 0px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #40ff40;
    }

    /* Markdown styling */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #ffff00 !important;
        text-shadow: 2px 2px 0px #000000 !important;
        font-family: 'Press Start 2P', monospace !important;
    }

    .stMarkdown p, .stMarkdown li {
        color: #00ff00 !important;
        font-family: 'Press Start 2P', monospace !important;
        font-size: 10px !important;
        line-height: 1.6 !important;
    }

    .stMarkdown strong {
        color: #ffff00 !important;
        text-shadow: 1px 1px 0px #000000 !important;
    }

    .stMarkdown hr {
        border: 2px solid #ff0040 !important;
        border-radius: 0px !important;
        margin: 16px 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# Tombol toggle dengan icon dinamis dan style 8-bit
st.markdown("""
<button id="sidebar-toggle" class="toggle-btn">&lt;&lt;</button>
""", unsafe_allow_html=True)

# Script untuk toggle animasi sidebar dengan perubahan icon
components.html("""
<script>
const waitForSidebar = () => {
    const sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
    const toggleBtn = window.parent.document.getElementById("sidebar-toggle");

    if (!sidebar || !toggleBtn) {
        setTimeout(waitForSidebar, 100);
        return;
    }

    // Sembunyikan semua tombol bawaan sidebar
    const hideDefaultButtons = () => {
        const collapseBtn = window.parent.document.querySelector('button[title="Collapse sidebar"]');
        const expandBtn = window.parent.document.querySelector('button[title="Expand sidebar"]');
        const collapsedControl = window.parent.document.querySelector('[data-testid="collapsedControl"]');
        
        if (collapseBtn) collapseBtn.style.display = 'none';
        if (expandBtn) expandBtn.style.display = 'none';
        if (collapsedControl) collapsedControl.style.display = 'none';
        
        const headerButtons = sidebar.querySelectorAll('button[kind="header"], button[data-testid="baseButton-header"]');
        headerButtons.forEach(btn => btn.style.display = 'none');
        
        const sidebarButtons = sidebar.querySelectorAll('button');
        sidebarButtons.forEach(btn => {
            if (btn.getAttribute('title') === 'Collapse sidebar' || 
                btn.getAttribute('title') === 'Expand sidebar' ||
                btn.getAttribute('kind') === 'header' ||
                btn.getAttribute('data-testid') === 'baseButton-header') {
                btn.style.display = 'none';
            }
        });
    };

    hideDefaultButtons();
    setInterval(hideDefaultButtons, 500);

    // State sidebar dengan sound effect simulation
    let sidebarVisible = true;
    
    const updateToggleIcon = () => {
        if (sidebarVisible) {
            toggleBtn.innerHTML = '&lt;&lt;';
            toggleBtn.style.animation = 'buttonBlink 1s ease-in-out infinite';
        } else {
            toggleBtn.innerHTML = '&gt;&gt;';
            toggleBtn.style.animation = 'buttonBlink 0.5s ease-in-out infinite';
        }
    };

    // Arcade-style button click effect
    toggleBtn.addEventListener("click", () => {
        // Visual feedback
        toggleBtn.style.transform = 'translate(2px, 2px)';
        toggleBtn.style.boxShadow = '0 0 0 2px #ff0040, 2px 2px 0 2px #000000';
        
        setTimeout(() => {
            toggleBtn.style.transform = '';
            toggleBtn.style.boxShadow = '';
        }, 100);
        
        // Toggle sidebar
        sidebarVisible = !sidebarVisible;
        sidebar.classList.toggle("sidebar-hidden");
        updateToggleIcon();
    });

    // Set icon awal
    updateToggleIcon();
};

waitForSidebar();
</script>
""", height=0)

# HEADER
st.markdown("""
<div class="header-box">
    <div class="main-title">🎥 Hayu-IT: HSAL Analysis on Youtube Indonesian Transcripts</div>
    <div class="subtitle">Deteksi otomatis ujaran kebencian dan bahasa kasar dari video YouTube berbahasa Indonesia</div>
</div>
""", unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.markdown("## 🔍 Fitur Utama")
    st.markdown("""
    - Ambil transkrip video YouTube berbahasa Indonesia.  
    - Deteksi ujaran kebencian dan bahasa kasar secara otomatis.  
    - Tampilkan hasil analisis lengkap dengan label dan timestamp.  
    - Didukung oleh model IndoBERTweet + BiGRU + MAML.
    """)
    st.markdown("## 🧾 Cara Menggunakan")
    st.markdown("""
    1. Masukkan URL video YouTube.  
    2. Pastikan video memiliki subtitle Bahasa Indonesia.  
    3. Klik "Analisis Video" dan tunggu hasilnya.  
    _Catatan: proses bisa memakan waktu tergantung durasi._
    """)

# ✅ Arsitektur model
class IndoBERTweetBiGRU(nn.Module):
    def __init__(self, bert, hidden_size=512, num_classes=13):
        super().__init__()
        self.bert = bert
        self.gru = nn.GRU(
            input_size=self.bert.config.hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2 + self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        cls_output = sequence_output[:, 0, :]
        gru_output, _ = self.gru(sequence_output)
        pooled_output = gru_output.mean(dim=1)
        combined = torch.cat((pooled_output, cls_output), dim=1)
        logits = self.fc(combined)
        return logits

# 🔠 Label klasifikasi
LABELS = [
    "HS", "Abusive", "HS_Individual", "HS_Group", "HS_Religion",
    "HS_Race", "HS_Physical", "HS_Gender", "HS_Other",
    "HS_Weak", "HS_Moderate", "HS_Strong", "PS" # PS = Konten Positif
]

LABEL_DESCRIPTIONS = {
    "HS": "Ujaran Kebencian",
    "Abusive": "Bahasa Kasar/Ofensif",
    "HS_Individual": "Kebencian terhadap Individu",
    "HS_Group": "Kebencian terhadap Kelompok",
    "HS_Religion": "Kebencian berbasis Agama",
    "HS_Race": "Kebencian berbasis Ras/Etnis",
    "HS_Physical": "Kebencian berbasis Fisik",
    "HS_Gender": "Kebencian berbasis Gender",
    "HS_Other": "Kebencian Kategori Lain",
    "HS_Weak": "Tingkat Kebencian Ringan",
    "HS_Moderate": "Tingkat Kebencian Sedang",
    "HS_Strong": "Tingkat Kebencian Berat",
    "PS": "Ujaran Positif"
}

# 🧼 Preprocessing
def preprocessing(text):
    string = text.lower()
    string = re.sub(r"\n", "", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = string.strip()
    string = re.sub(r'[^A-Za-z\s`"]', " ", string)
    return string

# 🔎 Ambil ID dari URL YouTube
def extract_video_id(url):
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"youtu\.be\/([0-9A-Za-z_-]{11})",
        r"youtube\.com\/embed\/([0-9A-Za-z_-]{11})",
        r"youtube\.com\/watch\?v=([0-9A-Za-z_-]{11})"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

# 🌐 Ambil transcript dari SearchAPI.io
def get_transcript_from_searchapi(video_id: str, api_key: str) -> Optional[Dict]:
    try:
        url = "https://www.searchapi.io/api/v1/search"
        params = {
            "engine": "youtube_transcripts",
            "video_id": video_id,
            "lang": "id",
            "api_key": api_key
        }
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        return data
    except Exception as e:
        st.error(f"Gagal mengambil transcript: {str(e)}")
        return None

# 📦 Download model dan tokenizer
@st.cache_resource
def load_model_tokenizer():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained("indolem/indobertweet-base-uncased")
        bert = AutoModel.from_pretrained("indolem/indobertweet-base-uncased").to(device)

        # model yang dilatih menggunakan pelatihan standar
        #model_safetensors_id = "1U7Z_M4OMosOCD-XEMN1YI19meS4HJ8e8"  
        #safetensors_path = "model_conventional_training.safetensors"         
        #safetensors_url = f"https://drive.google.com/uc?id={model_safetensors_id}"

        # model yang dilatih menggunakan MAML
        model_safetensors_id = "1SfyGkTgRxjx3JEwZ79zJuz5wciOH6d6_"
        safetensors_path = "final_model.safetensors"
        safetensors_url = f"https://drive.google.com/uc?id={model_safetensors_id}"
        
        if not os.path.exists(safetensors_path):
            try:
                gdown.download(safetensors_url, safetensors_path, quiet=False)
            except Exception as e:
                st.error(f"❌ Gagal download model SafeTensors: {str(e)}")
                st.info("💡 Pastikan file dapat diakses publik dan ID benar.")
                return None, None, None

        model = IndoBERTweetBiGRU(bert=bert, hidden_size=512, num_classes=13)

        if os.path.exists(safetensors_path):
            try:
                state_dict = load_file(safetensors_path)
                model.load_state_dict(state_dict)
                model.to(device)
            except Exception as e:
                st.error(f"❌ Gagal memuat model dari SafeTensors: {str(e)}")
                st.info("💡 Pastikan file SafeTensors tidak korup dan arsitektur model cocok.")
                return None, None, None
        else:
            st.error("❌ File model SafeTensors tidak ditemukan. Pastikan telah diunduh.")
            return None, None, None

        model.eval()
        return model, tokenizer, device
    except Exception as e:
        st.error(f"Error loading model/tokenizer: {str(e)}")
        import traceback
        st.exception(traceback.format_exc())
        return None, None, None

# Fungsi untuk memprediksi satu kalimat
def predict_sentence(text, model, tokenizer, device, threshold=0.5):
    cleaned_text = preprocessing(text)
    inputs = tokenizer(
        cleaned_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=192
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.sigmoid(logits)
        predictions = (probs > threshold).int().numpy()[0]

    detected_labels = [LABELS[i] for i, val in enumerate(predictions) if val == 1]

    # Detail probabilitas untuk setiap label
    label_probs = {LABELS[i]: float(probs[0][i]) for i in range(len(LABELS))}

    return detected_labels, label_probs


# 🎯 Main App
def main():
    # SearchAPI key disematkan langsung di sini
    api_key = "Quq35w6JgdtV1fJcnACFK4qF"

    if "is_analyzing" not in st.session_state:
        st.session_state.is_analyzing = False
    if "analysis_done" not in st.session_state:
        st.session_state.analysis_done = False

    youtube_url = st.text_input("🔗 Masukkan URL Video YouTube:")

    if youtube_url:
        video_id = extract_video_id(youtube_url)
        if video_id:
            st.video(f"https://www.youtube.com/watch?v={video_id}")

            with st.spinner("📦 Loading model dan tokenizer..."):
                model, tokenizer, device = load_model_tokenizer()

            if model is None or tokenizer is None or device is None:
                st.error("❌ Gagal memuat model. Periksa koneksi dan file model.")
                return

            if st.button("🚀 Analisis Video", use_container_width=True):
                st.session_state.is_analyzing = True
                st.session_state.analysis_done = False

                with st.spinner("📥 Mengambil transcript dari video..."):
                    transcript_data = get_transcript_from_searchapi(video_id, api_key)

                if not transcript_data or "transcripts" not in transcript_data:
                    st.error("❌ Gagal mengambil transcript. Pastikan video memiliki subtitle bahasa Indonesia.")
                    return

                transcript_entries = transcript_data["transcripts"]
                available_languages = transcript_data.get("available_languages", [])
                is_not_auto_generated = any(lang["name"] == "Indonesian"for lang in available_languages)

                if not is_not_auto_generated:
                    st.warning("⚠️ Transkrip ini merupakan Auto-Generated dan mungkin mengandung kesalahan.")

                full_text = " ".join([entry['text'] for entry in transcript_entries]) # Tidak menambahkan titik di setiap entry
                st.success("✅ Transcript berhasil diambil! ")

                with st.expander("📄 Cuplikan Transcript"):
                    st.text_area("", full_text[:1000] + ("..." if len(full_text) > 1000 else ""), height=200)

                sentences = re.split(r'(?<=[.!?])\s+|\n', full_text)
                clean_sentences = [s.strip() for s in sentences if s.strip()]

                if not clean_sentences:
                    st.warning("Tidak ada kalimat yang dapat dianalisis dari transkrip ini.")
                    return

                problematic_sentences_count = 0
                problematic_sentences_details = []
        
                progress_text = "Analisis kalimat sedang berjalan. Mohon tunggu..."
                my_bar = st.progress(0, text=progress_text)

                start_time = time.time()

                for i, sentence in enumerate(clean_sentences):
                    detected_labels, label_probs = predict_sentence(sentence, model, tokenizer, device)

                    # Logika untuk kalimat "bermasalah": setiap label selain 'PS'
                    is_problematic = any(label != "PS" for label in detected_labels)

                    if is_problematic:
                        problematic_sentences_count += 1
                        
                        # Temukan timestamp
                        matched_entry = next((entry for entry in transcript_entries if sentence.strip().startswith(entry['text'].strip()[:10])), None)
                        timestamp = matched_entry["start"] if matched_entry else None
                        timestamp_str = f"{int(timestamp // 60):02d}:{int(timestamp % 60):02d}" if timestamp is not None else "??:??"

                        # Urutkan hanya label aktif (prob > threshold dan bukan PS)
                        sorted_active = sorted(
                            [(LABEL_DESCRIPTIONS[label], float(prob)) for label, prob in label_probs.items() if prob > 0.5 and label != "PS"],
                            key=lambda x: x[1],
                            reverse=True
                        )

                        # Simpan seluruh probabilitas (untuk dropdown), tapi aktif label saja untuk ringkasan
                        all_probs = {LABEL_DESCRIPTIONS[label]: f"{float(prob):.1%}" for label, prob in label_probs.items()}

                        problematic_sentences_details.append({
                            "kalimat": sentence,
                            "timestamp": timestamp_str,
                            "label_terdeteksi": [label for label, _ in sorted_active],
                            "probabilitas": all_probs
                        })

                    progress_percentage = (i + 1) / len(clean_sentences)
                    my_bar.progress(progress_percentage, text=f"{progress_text} {int(progress_percentage * 100)}%")

                my_bar.empty()

                end_time = time.time()
                elapsed_time = end_time - start_time

                minutes = int(elapsed_time // 60)
                seconds = int(elapsed_time % 60)
                milliseconds = int((elapsed_time - int(elapsed_time)) * 1000)

                st.info(f"⏱️ Waktu proses analisis: {minutes} menit {seconds} detik {milliseconds} milidetik")

                st.session_state.is_analyzing = False
                st.session_state.analysis_done = True

                st.subheader("📊 Ringkasan Hasil Deteksi:")

                total_sentences = len(clean_sentences)
                if total_sentences > 0:
                    percentage_problematic = (problematic_sentences_count / total_sentences) * 100
                    # Tambahan Warning jika lebih dari 50% kalimat tidak positif
                    if percentage_problematic > 10.0:
                        st.error("⚠️ **PERINGATAN**: Lebih dari 10% konten video ini terdeteksi sebagai **konten tidak positif** (ujaran kebencian/abusive) dan **tidak layak dikonsumsi** secara umum.")
                    st.warning(f"Dari **{total_sentences} kalimat**, **{problematic_sentences_count} kalimat ({percentage_problematic:.1f}%)** terklasifikasi sebagai **konten bermasalah** (Ujaran Kebencian / Abusive).")
                else:
                    st.warning("Tidak ada kalimat untuk dianalisis.")

                if problematic_sentences_details:
                    st.info("🚨 Berikut adalah kalimat-kalimat yang terdeteksi bermasalah:")
                    for idx, detail in enumerate(problematic_sentences_details, 1):
                        st.markdown(f"---")
                        st.markdown(f"**Kalimat {idx}** _(pada menit {detail['timestamp']})_: {detail['kalimat']}")
                        # Pastikan hanya menampilkan label yang terdeteksi dan bukan PS
                        display_labels = [label for label in detail['label_terdeteksi'] if label != "Konten Positif"]
                        if display_labels:
                            st.markdown(f"**Label Terdeteksi:** {', '.join(display_labels)}")
                        else:
                            st.markdown(f"**Label Terdeteksi:** (Tidak ada label spesifik yang terdeteksi selain 'Konten Positif')") # Fallback jika hanya PS yang terdeteksi tapi diabaikan

                        with st.expander("Detail Probabilitas:"):
                            for label_desc, prob in detail['probabilitas'].items():
                                st.write(f"- **{label_desc}**: {prob}")
                else:
                    st.success("✅ Tidak terdeteksi adanya hate speech atau konten bermasalah dalam transkrip ini.")

            if st.session_state.is_analyzing and not st.session_state.analysis_done:
                st.subheader("🔍 Menganalisis Konten Video per Kalimat...")

        else:
            st.error("❌ URL tidak valid. Harap masukkan URL video YouTube yang benar.")

if __name__ == "__main__":
    main()