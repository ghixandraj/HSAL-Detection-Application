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

# CSS Styling dengan tema 8-bit Pac-Man
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

    /* Animasi Pac-Man */
    @keyframes pacmanMove {
        0% { background-position: 0 0; }
        100% { background-position: 40px 40px; }
    }

    @keyframes pacmanGlow {
        0%, 100% { 
            text-shadow: 0 0 5px #ffff00, 0 0 10px #ffff00, 0 0 15px #ffff00, 0 0 20px #ffff00;
            transform: scale(1);
        }
        50% { 
            text-shadow: 0 0 8px #ffff00, 0 0 15px #ffff00, 0 0 25px #ffff00, 0 0 30px #ffff00;
            transform: scale(1.02);
        }
    }

    @keyframes ghostFloat {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-5px); }
    }

    @keyframes powerPellet {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.2); }
    }

    @keyframes dotMove {
        0% { background-position: 0 0; }
        100% { background-position: 20px 20px; }
    }

    /* Background dengan pola maze Pac-Man */
    html, body, [class*="css"] {
        font-family: 'Press Start 2P', monospace !important;
        background: 
            radial-gradient(circle at 10px 10px, #ffff00 2px, transparent 2px),
            radial-gradient(circle at 30px 30px, #ffff00 1px, transparent 1px),
            linear-gradient(90deg, #000080 2px, transparent 2px),
            linear-gradient(180deg, #000080 2px, transparent 2px),
            #000022;
        background-size: 20px 20px, 40px 40px, 100px 100px, 100px 100px;
        animation: dotMove 4s linear infinite;
        color: #ffff00;
        overflow-x: hidden;
    }

    /* Sidebar dengan tema ghost */
    [data-testid="stSidebar"] {
        background: 
            radial-gradient(circle at 50% 30%, rgba(255, 100, 150, 0.1) 30%, transparent 30%),
            linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%) !important;
        border-right: 4px solid #ffff00 !important;
        box-shadow: 
            4px 0 0 #ff6b6b,
            8px 0 0 #4ecdc4,
            12px 0 0 #45b7d1,
            inset -4px 0 0 #ffff00 !important;
        transition: transform 0.3s cubic-bezier(0.68, -0.55, 0.265, 1.55), opacity 0.3s ease;
        transform: translateX(0);
        opacity: 1;
        position: relative;
        z-index: 1;
    }

    [data-testid="stSidebar"]::before {
        content: '👻';
        position: absolute;
        top: 20px;
        right: 20px;
        font-size: 24px;
        animation: ghostFloat 2s ease-in-out infinite;
        opacity: 0.7;
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
        text-shadow: 2px 2px 0px #000080, 0 0 10px #ffff00;
        border-bottom: 3px solid #ffff00;
        padding-bottom: 8px;
        margin-bottom: 16px;
        font-size: 14px !important;
        position: relative;
    }

    [data-testid="stSidebar"] h2::after {
        content: '🍒';
        position: absolute;
        right: 0;
        animation: powerPellet 2s ease-in-out infinite;
    }

    [data-testid="stSidebar"] p, [data-testid="stSidebar"] li {
        color: #ffffff !important;
        font-size: 10px !important;
        line-height: 1.8 !important;
        text-shadow: 1px 1px 0px #000080;
    }

    /* Header box dengan tema Pac-Man */
    .header-box {
        background: 
            radial-gradient(circle at 20% 50%, #ffff00 30px, transparent 30px),
            radial-gradient(circle at 80% 50%, #ff6b6b 20px, transparent 20px),
            radial-gradient(circle at 60% 30%, #4ecdc4 15px, transparent 15px),
            radial-gradient(circle at 40% 70%, #45b7d1 15px, transparent 15px),
            linear-gradient(135deg, #000080, #000022, #000080);
        border: 6px solid #ffff00;
        border-radius: 20px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 
            0 0 0 4px #000080,
            0 0 0 8px #ffff00,
            0 0 20px rgba(255, 255, 0, 0.5),
            inset 0 0 20px rgba(255, 255, 0, 0.1);
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
        background: 
            radial-gradient(circle at 25% 25%, rgba(255, 255, 0, 0.1) 2px, transparent 2px),
            radial-gradient(circle at 75% 75%, rgba(255, 255, 0, 0.1) 2px, transparent 2px);
        background-size: 30px 30px;
        animation: pacmanMove 3s linear infinite;
    }

    .header-box::after {
        content: '🟡';
        position: absolute;
        top: 50%;
        left: 20px;
        transform: translateY(-50%);
        font-size: 2rem;
        animation: pacmanGlow 2s ease-in-out infinite;
    }

    .main-title {
        font-size: 1.8rem;
        font-weight: normal;
        color: #ffff00;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 
            4px 4px 0px #000080,
            2px 2px 0px #ff6b6b,
            0 0 20px #ffff00;
        animation: pacmanGlow 3s ease-in-out infinite;
        position: relative;
        z-index: 1;
    }

    .subtitle {
        font-size: 0.7rem;
        text-align: center;
        color: #ffffff;
        text-shadow: 2px 2px 0px #000080, 0 0 10px #4ecdc4;
        position: relative;
        z-index: 1;
        margin-top: 1rem;
    }

    /* Toggle Button dengan style Pac-Man */
    .toggle-btn {
        position: fixed;
        top: 20px;
        left: 20px;
        background: radial-gradient(circle at 30% 30%, #ffff00, #ffcc00);
        color: #000080;
        font-family: 'Press Start 2P', monospace;
        font-size: 12px;
        font-weight: bold;
        border: 4px solid #000080;
        cursor: pointer;
        z-index: 9999;
        padding: 10px 15px;
        border-radius: 50%;
        width: 60px;
        height: 60px;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.3s ease;
        box-shadow: 
            0 0 0 2px #ffff00,
            0 0 10px rgba(255, 255, 0, 0.5),
            4px 4px 0 2px #000080;
        text-shadow: 1px 1px 0px #ffcc00;
        animation: powerPellet 2s ease-in-out infinite;
    }

    .toggle-btn:hover {
        background: radial-gradient(circle at 30% 30%, #ffff44, #ffdd00);
        transform: scale(1.1);
        box-shadow: 
            0 0 0 2px #ffff00,
            0 0 20px rgba(255, 255, 0, 0.8),
            6px 6px 0 2px #000080;
    }

    .toggle-btn:active {
        transform: scale(0.95);
        box-shadow: 
            0 0 0 2px #ffff00,
            0 0 5px rgba(255, 255, 0, 0.3),
            2px 2px 0 2px #000080;
    }

    /* Styling untuk input dengan tema Pac-Man */
    .stTextInput input {
        background: linear-gradient(45deg, #000080, #000022) !important;
        border: 3px solid #ffff00 !important;
        border-radius: 15px !important;
        color: #ffff00 !important;
        font-family: 'Press Start 2P', monospace !important;
        font-size: 12px !important;
        padding: 15px !important;
        box-shadow: 
            inset 2px 2px 0px #4ecdc4,
            0 0 10px rgba(255, 255, 0, 0.3) !important;
    }

    .stTextInput input:focus {
        border-color: #ffff00 !important;
        box-shadow: 
            inset 2px 2px 0px #4ecdc4,
            0 0 20px rgba(255, 255, 0, 0.6) !important;
    }

    /* Button dengan tema power pellet */
    .stButton button {
        background: radial-gradient(circle at 50% 50%, #ffff00, #ffcc00, #ff6b6b) !important;
        border: 4px solid #000080 !important;
        border-radius: 25px !important;
        color: #000080 !important;
        font-family: 'Press Start 2P', monospace !important;
        font-size: 12px !important;
        padding: 15px 30px !important;
        font-weight: bold !important;
        text-shadow: 1px 1px 0px #ffff00 !important;
        box-shadow: 
            0 0 0 2px #ffff00,
            0 0 15px rgba(255, 255, 0, 0.5),
            4px 4px 0 2px #000080 !important;
        transition: all 0.3s ease !important;
        position: relative;
        overflow: hidden;
    }

    .stButton button::before {
        content: '🍒';
        position: absolute;
        left: 10px;
        top: 50%;
        transform: translateY(-50%);
        animation: powerPellet 1.5s ease-in-out infinite;
    }

    .stButton button:hover {
        background: radial-gradient(circle at 50% 50%, #ffff44, #ffdd00, #ff8b8b) !important;
        transform: scale(1.05) !important;
        box-shadow: 
            0 0 0 2px #ffff00,
            0 0 25px rgba(255, 255, 0, 0.8),
            6px 6px 0 2px #000080 !important;
    }

    .stButton button:active {
        transform: scale(0.95) !important;
        box-shadow: 
            0 0 0 2px #ffff00,
            0 0 10px rgba(255, 255, 0, 0.3),
            2px 2px 0 2px #000080 !important;
    }

    /* Progress bar dengan tema Pac-Man */
    .stProgress > div > div {
        background: linear-gradient(90deg, #000080, #000022) !important;
        border: 3px solid #ffff00 !important;
        border-radius: 15px !important;
        height: 30px !important;
        box-shadow: 
            inset 0 0 0 2px #4ecdc4,
            0 0 10px rgba(255, 255, 0, 0.3) !important;
    }

    .stProgress > div > div > div {
        background: 
            radial-gradient(circle at 10px 15px, #ffff00 3px, transparent 3px),
            radial-gradient(circle at 30px 15px, #ffff00 3px, transparent 3px),
            radial-gradient(circle at 50px 15px, #ffff00 3px, transparent 3px),
            linear-gradient(90deg, #4ecdc4, #45b7d1) !important;
        border-radius: 12px !important;
        background-size: 40px 30px, 100% 100%;
        animation: pacmanMove 1s linear infinite !important;
        position: relative;
    }

    .stProgress > div > div > div::after {
        content: '🟡';
        position: absolute;
        right: 5px;
        top: 50%;
        transform: translateY(-50%);
        animation: pacmanGlow 1s ease-in-out infinite;
    }

    /* Alert dan info boxes dengan tema ghost */
    .stAlert {
        border: 3px solid #4ecdc4 !important;
        border-radius: 15px !important;
        background: linear-gradient(135deg, #000080, #000022) !important;
        color: #ffffff !important;
        font-family: 'Press Start 2P', monospace !important;
        font-size: 10px !important;
        box-shadow: 
            0 0 15px rgba(78, 205, 196, 0.3),
            4px 4px 0px #000080 !important;
        position: relative;
    }

    .stAlert::before {
        content: '👻';
        position: absolute;
        top: 10px;
        right: 15px;
        font-size: 16px;
        animation: ghostFloat 2s ease-in-out infinite;
    }

    .stError {
        border-color: #ff6b6b !important;
        background: linear-gradient(135deg, #4a0000, #220000) !important;
        color: #ff6b6b !important;
        box-shadow: 
            0 0 15px rgba(255, 107, 107, 0.3),
            4px 4px 0px #220000 !important;
    }

    .stError::before {
        content: '💀';
    }

    .stSuccess {
        border-color: #4ecdc4 !important;
        background: linear-gradient(135deg, #004a4a, #002222) !important;
        color: #4ecdc4 !important;
        box-shadow: 
            0 0 15px rgba(78, 205, 196, 0.3),
            4px 4px 0px #002222 !important;
    }

    .stSuccess::before {
        content: '🍒';
    }

    .stWarning {
        border-color: #ffff00 !important;
        background: linear-gradient(135deg, #4a4a00, #222200) !important;
        color: #ffff00 !important;
        box-shadow: 
            0 0 15px rgba(255, 255, 0, 0.3),
            4px 4px 0px #222200 !important;
    }

    .stWarning::before {
        content: '⚠️';
    }

    /* Video player dengan frame arcade */
    .stVideo {
        border: 6px solid #ffff00 !important;
        border-radius: 20px !important;
        box-shadow: 
            0 0 0 4px #000080,
            0 0 20px rgba(255, 255, 0, 0.5),
            8px 8px 0 4px #000080 !important;
        position: relative;
    }

    .stVideo::before {
        content: '🕹️';
        position: absolute;
        top: -15px;
        left: 50%;
        transform: translateX(-50%);
        background: #000080;
        padding: 5px 10px;
        border-radius: 10px;
        font-size: 14px;
        z-index: 1;
    }

    /* Expander dengan tema maze */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, #000080, #000022) !important;
        border: 3px solid #ffff00 !important;
        border-radius: 15px !important;
        color: #ffff00 !important;
        font-family: 'Press Start 2P', monospace !important;
        font-size: 10px !important;
        padding: 15px !important;
        position: relative;
    }

    .streamlit-expanderHeader::before {
        content: '🍎';
        position: absolute;
        right: 15px;
        top: 50%;
        transform: translateY(-50%);
        animation: powerPellet 2s ease-in-out infinite;
    }

    .streamlit-expanderContent {
        background: linear-gradient(135deg, #000022, #000080) !important;
        border: 3px solid #4ecdc4 !important;
        border-top: none !important;
        border-radius: 0 0 15px 15px !important;
        border-top-left-radius: 0 !important;
        border-top-right-radius: 0 !important;
    }

    /* Textarea dengan tema maze */
    .stTextArea textarea {
        background: linear-gradient(45deg, #000080, #000022) !important;
        border: 3px solid #4ecdc4 !important;
        border-radius: 15px !important;
        color: #ffffff !important;
        font-family: 'Press Start 2P', monospace !important;
        font-size: 8px !important;
        line-height: 1.6 !important;
        padding: 15px !important;
        box-shadow: 
            inset 2px 2px 0px #45b7d1,
            0 0 10px rgba(78, 205, 196, 0.3) !important;
    }

    /* Spinner dengan tema power pellet */
    .stSpinner {
        border: 4px solid #000080 !important;
        border-top: 4px solid #ffff00 !important;
        border-radius: 50% !important;
        width: 40px !important;
        height: 40px !important;
        animation: spin 1s linear infinite !important;
        position: relative;
    }

    .stSpinner::after {
        content: '🟡';
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 20px;
        animation: pacmanGlow 1s ease-in-out infinite;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    /* Scrollbar dengan tema Pac-Man */
    ::-webkit-scrollbar {
        width: 20px;
        background: #000080;
    }

    ::-webkit-scrollbar-track {
        background: #000022;
        border: 2px solid #ffff00;
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #ffff00, #ffcc00);
        border: 2px solid #000080;
        border-radius: 10px;
        position: relative;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #ffff44, #ffdd00);
    }

    /* Markdown styling dengan tema Pac-Man */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #ffff00 !important;
        text-shadow: 
            2px 2px 0px #000080,
            0 0 10px #ffff00 !important;
        font-family: 'Press Start 2P', monospace !important;
        position: relative;
    }

    .stMarkdown h1::after, .stMarkdown h2::after, .stMarkdown h3::after {
        content: '🍒';
        margin-left: 10px;
        animation: powerPellet 2s ease-in-out infinite;
    }

    .stMarkdown p, .stMarkdown li {
        color: #ffffff !important;
        font-family: 'Press Start 2P', monospace !important;
        font-size: 10px !important;
        line-height: 1.8 !important;
        text-shadow: 1px 1px 0px #000080 !important;
    }

    .stMarkdown strong {
        color: #ffff00 !important;
        text-shadow: 
            1px 1px 0px #000080,
            0 0 5px #ffff00 !important;
    }

    .stMarkdown hr {
        border: 3px solid #4ecdc4 !important;
        border-radius: 3px !important;
        margin: 20px 0 !important;
        box-shadow: 0 0 10px rgba(78, 205, 196, 0.3) !important;
    }

    /* Efek partikel dots */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: 
            radial-gradient(circle at 5% 5%, rgba(255, 255, 0, 0.1) 1px, transparent 1px),
            radial-gradient(circle at 95% 5%, rgba(255, 107, 107, 0.1) 1px, transparent 1px),
            radial-gradient(circle at 5% 95%, rgba(78, 205, 196, 0.1) 1px, transparent 1px),
            radial-gradient(circle at 95% 95%, rgba(69, 183, 209, 0.1) 1px, transparent 1px);
        background-size: 50px 50px;
        animation: dotMove 5s linear infinite;
        pointer-events: none;
        z-index: -1;
    }

    /* Responsive design */
    @media (max-width: 768px) {
        .main-title {
            font-size: 1.2rem;
        }
        
        .subtitle {
            font-size: 0.6rem;
        }
        
        .toggle-btn {
            width: 50px;
            height: 50px;
            font-size: 10px;
        }
    }
</style>
""", unsafe_allow_html=True)

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