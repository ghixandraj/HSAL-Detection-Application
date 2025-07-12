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

# Tambahkan link Google Font secara terpisah
st.markdown('<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">', unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-container">
    <div class="top-pill">Aplikasi Deteksi Ujaran Kebencian dan Bahasa Kasar</div>
    
    <div class="main-header">
        <h1 class="main-title">
            Hayu-IT: <span>HSAL Analysis on Youtube Indonesian Transcripts</span>
        </h1>
        <p class="subtitle">
            Deteksi otomatis ujaran kebencian dan bahasa kasar dari transkrip video YouTube berbahasa Indonesia
        </p>
    </div>

    <div class="feature-section">
        <div class="section-title-pill">Fitur Utama</div>
        <div class="category-box">
            <h2 class="category-title">Kategori Deteksi</h2>
            <p class="category-text">
                Pendeteksian ini dilakukan terhadap <strong>13 kategori</strong> berbeda. Ada kategori <span class="highlight">ujaran positif</span>, <span class="highlight">bahasa kasar</span>, dan <span class="highlight">ujaran kebencian</span> dengan berbagai sub-kategori.
            </p>
            <div class="cards-container">
                <div class="info-card">
                    <div class="card-icon">🎯</div>
                    <h3 class="card-title">Target</h3>
                    <p class="card-subtitle">Sasaran Ujaran Kebencian</p>
                    <p class="card-content">Individual dan Grup</p>
                </div>
                <div class="info-card">
                    <div class="card-icon">🔲</div>
                    <h3 class="card-title">Kategori</h3>
                    <p class="card-subtitle">Jenis Diskriminasi</p>
                    <p class="card-content">Agama, Ras, Fisik, Gender, dan lain-lain</p>
                </div>
                <div class="info-card">
                    <div class="card-icon">📊</div>
                    <h3 class="card-title">Intensitas</h3>
                    <p class="card-subtitle">Tingkat Keparahan</p>
                    <p class="card-content">Ringan, Sedang, dan Berat</p>
                </div>
            </div>
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

    **Kategori Deteksi:**
    - Ujaran kebencian individual & grup
    - Diskriminasi agama, ras, gender
    - Bahasa kasar dan ofensif
    - Tingkat intensitas (ringan/sedang/berat)
    - Positif atau Netral 
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

    **Interpretasi Hasil:**
    - 🔴 **Merah**: Konten bermasalah terdeteksi
    - 🟡 **Kuning**: Perlu perhatian khusus
    - 🟢 **Hijau**: Konten aman
    - 📊 **Probabilitas > 50%**: Prediksi valid

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

        model_safetensors_id = "1SfyGkTgRxjx3JEwZ79zJuz5wciOH6d6_"
        safetensors_path = "final_model.safetensors"
        safetensors_url = f"https://drive.google.com/uc?id={model_safetensors_id}"
        
        if not os.path.exists(safetensors_path):
            try:
                with st.spinner("Mengunduh model AI (sekitar 400MB)... Ini hanya terjadi sekali."):
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
    label_probs = {LABELS[i]: float(probs[0][i]) for i in range(len(LABELS))}
    return detected_labels, label_probs

# 🎯 Main App
def main():
    api_key = "Quq35w6JgdtV1fJcnACFK4qF"

    if "is_analyzing" not in st.session_state:
        st.session_state.is_analyzing = False
    if "analysis_done" not in st.session_state:
        st.session_state.analysis_done = False

    st.markdown("""
    <p class="input-label">✓ Masukkan URL Video YouTube:</p>
    """, unsafe_allow_html=True)
    
    youtube_url = st.text_input(
        label="URL Input",
        placeholder="Press Enter to apply",
        label_visibility="collapsed"
    )

    if youtube_url:
        video_id = extract_video_id(youtube_url)
        if video_id:
            st.video(f"https://www.youtube.com/watch?v={video_id}")

            with st.spinner("📦 Memuat model dan tokenizer..."):
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
                    st.session_state.is_analyzing = False
                    return

                transcript_entries = transcript_data["transcripts"]
                full_text = " ".join([entry['text'] for entry in transcript_entries])
                st.success("✅ Transcript berhasil diambil!")

                with st.expander("📄 Cuplikan Transcript"):
                    st.text_area("", full_text[:1000] + ("..." if len(full_text) > 1000 else ""), height=200)

                sentences = re.split(r'(?<=[.!?])\s+|\n', full_text)
                clean_sentences = [s.strip() for s in sentences if s.strip()]

                if not clean_sentences:
                    st.warning("Tidak ada kalimat yang dapat dianalisis dari transkrip ini.")
                    st.session_state.is_analyzing = False
                    return

                problematic_sentences_details = []
                progress_bar = st.progress(0, text="Analisis kalimat sedang berjalan...")
                start_time = time.time()

                for i, sentence in enumerate(clean_sentences):
                    detected_labels, label_probs = predict_sentence(sentence, model, tokenizer, device)
                    is_problematic = any(label != "PS" for label in detected_labels)

                    if is_problematic:
                        matched_entry = next((e for e in transcript_entries if sentence.strip().startswith(e['text'].strip()[:10])), None)
                        timestamp = matched_entry["start"] if matched_entry else None
                        timestamp_str = f"{int(timestamp//60):02d}:{int(timestamp%60):02d}" if timestamp is not None else "N/A"
                        
                        problematic_sentences_details.append({
                            "kalimat": sentence,
                            "timestamp": timestamp_str,
                            "probabilitas": {LABEL_DESCRIPTIONS[l]: f"{p:.1%}" for l, p in label_probs.items()},
                            "label_terdeteksi": [LABEL_DESCRIPTIONS[l] for l in detected_labels if l != "PS"]
                        })
                    
                    progress_percentage = (i + 1) / len(clean_sentences)
                    progress_bar.progress(progress_percentage, text=f"Menganalisis... {int(progress_percentage * 100)}%")

                progress_bar.empty()
                elapsed_time = time.time() - start_time
                st.info(f"⏱️ Waktu analisis: {elapsed_time:.2f} detik")

                st.session_state.analysis_done = True
                st.session_state.is_analyzing = False
                
                # Menampilkan hasil
                st.subheader("📊 Ringkasan Hasil Deteksi")
                problematic_count = len(problematic_sentences_details)
                total_count = len(clean_sentences)
                if problematic_count > 0:
                    st.warning(f"Terdeteksi **{problematic_count} dari {total_count} kalimat** mengandung konten bermasalah.")
                    for detail in problematic_sentences_details:
                        st.markdown("---")
                        st.markdown(f"**Kalimat** _(pada menit {detail['timestamp']})_: {detail['kalimat']}")
                        if detail['label_terdeteksi']:
                            st.markdown(f"**Label Terdeteksi:** {', '.join(detail['label_terdeteksi'])}")
                        with st.expander("Lihat Detail Probabilitas"):
                            st.json(detail['probabilitas'])
                else:
                    st.success("✅ Tidak ada konten bermasalah yang terdeteksi dalam video ini.")

        else:
            st.error("❌ URL tidak valid. Harap masukkan URL video YouTube yang benar.")

if __name__ == "__main__":
    main()