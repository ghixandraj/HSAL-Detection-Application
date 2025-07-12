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
st.set_page_config(page_title="Hayu-IT: HSAL Analysis", page_icon="🎬", layout="wide")

# --- Fungsi untuk memuat CSS eksternal ---
def load_css(file_name):
    """
    Fungsi untuk memuat file CSS lokal dan menyuntikkannya ke dalam aplikasi Streamlit.
    """
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"File CSS '{file_name}' tidak ditemukan. Pastikan file tersebut berada di direktori yang sama dengan app.py.")

# --- Muat file CSS ---
load_css("style.css")

# Font
st.markdown('<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;900&display=swap" rel="stylesheet">', unsafe_allow_html=True)

# Membungkus semua konten utama dalam div 'main-container'
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# --- HEADER ---
st.markdown("""
<div class="header-box">
    <div class="top-pill">Aplikasi Deteksi Ujaran Kebencian dan Bahasa Kasar</div>
    <h1 class="main-title">Hayu-IT: HSAL Analysis on Youtube Indonesian Transcripts</h1>
    <p class="subtitle">Deteksi otomatis ujaran kebencian dan bahasa kasar dari transkrip video YouTube berbahasa Indonesia</p>
</div>
""", unsafe_allow_html=True)

# --- Feature Section ---
st.markdown("""
<div class="feature-section">
    <div class="category-box">
        <h2 class="category-title">Kategori Deteksi</h2>
        <p class="category-text">
            Pendeteksian ini dilakukan terhadap <span class="highlight-1">13 kategori</span> berbeda. Ada kategori <span class="highlight-2">ujaran positif</span>, <span class="highlight-3">bahasa kasar</span>, dan <span class="highlight-4">ujaran kebencian</span> dengan berbagai sub-kategori.
        </p>
    </div>
    <div class="cards-container">
        <div class="info-card">
            <div class="card-icon">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="22" viewBox="0 0 20 22" fill="none">
                    <path d="M10 22C8.61667 22 7.31667 21.7375 6.1 21.2125C4.88333 20.6875 3.825 19.975 2.925 19.075C2.025 18.175 1.3125 17.1167 0.7875 15.9C0.2625 14.6833 0 13.3833 0 12C0 10.1 0.495833 8.34583 1.4875 6.7375C2.47917 5.12917 3.825 3.9 5.525 3.05C5.54167 3.36667 5.575 3.6875 5.625 4.0125C5.675 4.3375 5.75833 4.71667 5.875 5.15C4.675 5.88333 3.72917 6.85417 3.0375 8.0625C2.34583 9.27083 2 10.5833 2 12C2 14.2333 2.775 16.125 4.325 17.675C5.875 19.225 7.76667 20 10 20C12.2333 20 14.125 19.225 15.675 17.675C17.225 16.125 18 14.2333 18 12C18 10.5833 17.6542 9.26667 16.9625 8.05C16.2708 6.83333 15.3167 5.85833 14.1 5.125C14.2167 4.69167 14.3 4.3125 14.35 3.9875C14.4 3.6625 14.4417 3.35 14.475 3.05C16.175 3.9 17.5208 5.125 18.5125 6.725C19.5042 8.325 20 10.0833 20 12C20 13.3833 19.7375 14.6833 19.2125 15.9C18.6875 17.1167 17.975 18.175 17.075 19.075C16.175 19.975 15.1167 20.6875 13.9 21.2125C12.6833 21.7375 11.3833 22 10 22ZM10 18C8.33333 18 6.91667 17.4167 5.75 16.25C4.58333 15.0833 4 13.6667 4 12C4 11.0333 4.2125 10.125 4.6375 9.275C5.0625 8.425 5.66667 7.71667 6.45 7.15C6.53333 7.4 6.625 7.6875 6.725 8.0125C6.825 8.3375 6.95833 8.74167 7.125 9.225C6.75833 9.60833 6.47917 10.0333 6.2875 10.5C6.09583 10.9667 6 11.4667 6 12C6 13.1 6.39167 14.0417 7.175 14.825C7.95833 15.6083 8.9 16 10 16C11.1 16 12.0417 15.6083 12.825 14.825C13.6083 14.0417 14 13.1 14 12C14 11.4667 13.9042 10.9667 13.7125 10.5C13.5208 10.0333 13.2417 9.60833 12.875 9.225C13.0083 8.825 13.1292 8.45417 13.2375 8.1125C13.3458 7.77083 13.4417 7.45 13.525 7.15C14.3083 7.71667 14.9167 8.425 15.35 9.275C15.7833 10.125 16 11.0333 16 12C16 13.6667 15.4167 15.0833 14.25 16.25C13.0833 17.4167 11.6667 18 10 18ZM9.7 8.5C9.48333 8.5 9.29583 8.4375 9.1375 8.3125C8.97917 8.1875 8.86667 8.025 8.8 7.825L7.8 4.575C7.68333 4.14167 7.60417 3.7625 7.5625 3.4375C7.52083 3.1125 7.5 2.8 7.5 2.5C7.5 1.8 7.74167 1.20833 8.225 0.725C8.70833 0.241667 9.3 0 10 0C10.7 0 11.2917 0.241667 11.775 0.725C12.2583 1.20833 12.5 1.8 12.5 2.5C12.5 2.8 12.4792 3.1125 12.4375 3.4375C12.3958 3.7625 12.3167 4.14167 12.2 4.575L11.2 7.825C11.1333 8.025 11.0208 8.1875 10.8625 8.3125C10.7042 8.4375 10.5167 8.5 10.3 8.5H9.7ZM10 14C9.45 14 8.97917 13.8042 8.5875 13.4125C8.19583 13.0208 8 12.55 8 12C8 11.45 8.19583 10.9792 8.5875 10.5875C8.97917 10.1958 9.45 10 10 10C10.55 10 11.0208 10.1958 11.4125 10.5875C11.8042 10.9792 12 11.45 12 12C12 12.55 11.8042 13.0208 11.4125 13.4125C11.0208 13.8042 10.55 14 10 14Z" fill="url(#paint0_linear_3_79)"/>
                    <defs>
                        <linearGradient id="paint0_linear_3_79" x1="10" y1="0" x2="10" y2="22" gradientUnits="userSpaceOnUse">
                            <stop stop-color="#39EDEA"/>
                            <stop offset="1" stop-color="#FBF8FC"/>
                        </linearGradient>
                    </defs>
                </svg>
            </div>
            <h3 class="card-title">Target</h3>
            <p class="card-subtitle">Sasaran Ujaran Kebencian</p>
            <p class="card-content">Individual dan Grup</p>
        </div>
        <div class="info-card">
            <div class="card-icon">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none">
                    <mask id="mask0_3_120" style="mask-type:alpha" maskUnits="userSpaceOnUse" x="0" y="0" width="24" height="24">
                        <rect width="24" height="24" fill="#D9D9D9"/>
                    </mask>
                    <g mask="url(#mask0_3_120)">
                        <path d="M7.425 9.47499L11.15 3.39999C11.25 3.23332 11.375 3.11249 11.525 3.03749C11.675 2.96249 11.8333 2.92499 12 2.92499C12.1667 2.92499 12.325 2.96249 12.475 3.03749C12.625 3.11249 12.75 3.23332 12.85 3.39999L16.575 9.47499C16.675 9.64165 16.725 9.81665 16.725 9.99999C16.725 10.1833 16.6833 10.35 16.6 10.5C16.5167 10.65 16.4 10.7708 16.25 10.8625C16.1 10.9542 15.925 11 15.725 11H8.275C8.075 11 7.9 10.9542 7.75 10.8625C7.6 10.7708 7.48333 10.65 7.4 10.5C7.31667 10.35 7.275 10.1833 7.275 9.99999C7.275 9.81665 7.325 9.64165 7.425 9.47499ZM17.5 22C16.25 22 15.1875 21.5625 14.3125 20.6875C13.4375 19.8125 13 18.75 13 17.5C13 16.25 13.4375 15.1875 14.3125 14.3125C15.1875 13.4375 16.25 13 17.5 13C18.75 13 19.8125 13.4375 20.6875 14.3125C21.5625 15.1875 22 16.25 22 17.5C22 18.75 21.5625 19.8125 20.6875 20.6875C19.8125 21.5625 18.75 22 17.5 22ZM3 20.5V14.5C3 14.2167 3.09583 13.9792 3.2875 13.7875C3.47917 13.5958 3.71667 13.5 4 13.5H10C10.2833 13.5 10.5208 13.5958 10.7125 13.7875C10.9042 13.9792 11 14.2167 11 14.5V20.5C11 20.7833 10.9042 21.0208 10.7125 21.2125C10.5208 21.4042 10.2833 21.5 10 21.5H4C3.71667 21.5 3.47917 21.4042 3.2875 21.2125C3.09583 21.0208 3 20.7833 3 20.5ZM17.5 20C18.2 20 18.7917 19.7583 19.275 19.275C19.7583 18.7917 20 18.2 20 17.5C20 16.8 19.7583 16.2083 19.275 15.725C18.7917 15.2417 18.2 15 17.5 15C16.8 15 16.2083 15.2417 15.725 15.725C15.2417 16.2083 15 16.8 15 17.5C15 18.2 15.2417 18.7917 15.725 19.275C16.2083 19.7583 16.8 20 17.5 20ZM5 19.5H9V15.5H5V19.5ZM10.05 8.99999H13.95L12 5.84999L10.05 8.99999Z" fill="url(#paint0_linear_3_120)"/>
                    </g>
                    <defs>
                        <linearGradient id="paint0_linear_3_120" x1="12.5" y1="2.92499" x2="12.5" y2="22" gradientUnits="userSpaceOnUse">
                            <stop stop-color="#39EDEA"/>
                            <stop offset="1" stop-color="#FBF8FC"/>
                        </linearGradient>
                    </defs>
                </svg>
            </div>
            <h3 class="card-title">Kategori</h3>
            <p class="card-subtitle">Jenis Diskriminasi</p>
            <p class="card-content">Agama, Ras, Fisik, Gender, dan lain-lain</p>
        </div>
        <div class="info-card">
            <div class="card-icon">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none">
                    <mask id="mask0_3_124" style="mask-type:alpha" maskUnits="userSpaceOnUse" x="0" y="0" width="24" height="24">
                        <rect width="24" height="24" fill="#D9D9D9"/>
                    </mask>
                    <g mask="url(#mask0_3_124)">
                        <path d="M5 16C3.9 16 2.95833 15.6083 2.175 14.825C1.39167 14.0417 1 13.1 1 12C1 10.9 1.39167 9.95833 2.175 9.175C2.95833 8.39167 3.9 8 5 8C6.1 8 7.04167 8.39167 7.825 9.175C8.60833 9.95833 9 10.9 9 12C9 13.1 8.60833 14.0417 7.825 14.825C7.04167 15.6083 6.1 16 5 16ZM5 14C5.55 14 6.02083 13.8042 6.4125 13.4125C6.80417 13.0208 7 12.55 7 12C7 11.45 6.80417 10.9792 6.4125 10.5875C6.02083 10.1958 5.55 10 5 10C4.45 10 3.97917 10.1958 3.5875 10.5875C3.19583 10.9792 3 11.45 3 12C3 12.55 3.19583 13.0208 3.5875 13.4125C3.97917 13.8042 4.45 14 5 14ZM17 18C15.3333 18 13.9167 17.4167 12.75 16.25C11.5833 15.0833 11 13.6667 11 12C11 10.3333 11.5833 8.91667 12.75 7.75C13.9167 6.58333 15.3333 6 17 6C18.6667 6 20.0833 6.58333 21.25 7.75C22.4167 8.91667 23 10.3333 23 12C23 13.6667 22.4167 15.0833 21.25 16.25C20.0833 17.4167 18.6667 18 17 18Z" fill="url(#paint0_linear_3_124)"/>
                    </g>
                    <defs>
                        <linearGradient id="paint0_linear_3_124" x1="12" y1="6" x2="12" y2="18" gradientUnits="userSpaceOnUse">
                            <stop stop-color="#39EDEA"/>
                            <stop offset="1" stop-color="#FBF8FC"/>
                        </linearGradient>
                    </defs>
                </svg>
            </div>
            <h3 class="card-title">Intensitas</h3>
            <p class="card-subtitle">Tingkat Keparahan</p>
            <p class="card-content">Ringan, Sedang, dan Berat</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    # 🔍 FITUR UTAMA
    st.markdown("## 🔍 Fitur Utama")
    st.markdown("""
    **Analisis Komprehensif:**
    - Ekstraksi transkrip otomatis dari video YouTube
    - Deteksi ujaran kebencian multi-kategori
    - Identifikasi bahasa kasar dan ofensif
    - Timestamping untuk setiap deteksi
    - Laporan detail probabilitas
    """)

    # 📋 CARA MENGGUNAKAN
    st.markdown("## 📋 Cara Menggunakan")
    st.markdown("""
    **Langkah-langkah:**
    1. **Masukkan URL YouTube** yang valid
    2. **Pastikan video memiliki subtitle** Bahasa Indonesia
    3. **Klik "Analisis Video"** dan tunggu prosesnya
    4. **Lihat hasil analisis** dengan detail timestamp
    5. **Baca laporan lengkap** untuk setiap kalimat bermasalah

    **Persyaratan:**
    - Video harus memiliki subtitle/transkrip bahasa Indonesia
    - Koneksi internet stabil untuk mengunduh model
    - Durasi video optimal: < 30 menit (untuk performa terbaik)
    """)

    # 🤖 DETAIL MODEL
    st.markdown("## 🤖 Detail Model AI")
    st.markdown("""
    **Arsitektur Model:**
    - **Base Model**: IndoBERTweet + BiGRU
    - **Output**: 13 kategori klasifikasi
                
    **Spesifikasi Teknis:**
    - **Hidden Size**: 512 dimensi
    - **Max Sequence Length**: 192 token
    - **Threshold**: 0.5 untuk klasifikasi
                
    **Performa Model:**
    - Dilatih pada dataset Indonesia
    - Multi-label classification
    - Optimized untuk bahasa informal
    """)

    # 💡 TIPS & PANDUAN
    st.markdown("## 💡 Tips & Panduan")
    st.markdown("""
    **Untuk Hasil Optimal:**
    - Hindari analisis video dengan transcript auto-generate
    - Video pendek (< 15 menit) diproses lebih cepat

    **Catatan Penting:**
    - Model dapat menghasilkan false positive/negative
    - Hasil harus diinterpretasi oleh manusia
    - Transkrip auto-generated mungkin kurang akurat
    """)

    # ❗ Disclaimer
    st.markdown("## ❗ Disclaimer")
    st.markdown("""
    - Aplikasi ini untuk penelitian dan edukasi
    - Hasil analisis bukan keputusan final
    - Gunakan dengan bijak dan bertanggung jawab
    """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8rem;">
        <p>🔬 Designed by Ghixandra Julyaneu Irawadi</p>
        <p>v1.0 - Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

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