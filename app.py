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

# --- CSS Tema Aurora Glass ---
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">

<style>
    /* --- Latar Belakang Aurora --- */
    @keyframes gradient-animation {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
        background: linear-gradient(-45deg, #1e0033, #3c004a, #2d0b49, #0f002d);
        background-size: 400% 400%;
        animation: gradient-animation 15s ease infinite;
        color: #EAEAF2 !important;
    }

    /* --- Sidebar Kaca Buram --- */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    [data-testid="stSidebar"] h2 {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        border-bottom-color: rgba(255, 255, 255, 0.2) !important;
    }
    [data-testid="stSidebar"] strong {
        color: #A78BFA !important; /* Ungu Lavender */
    }

    /* --- Tombol Sidebar (Tidak diubah, sudah bagus) --- */
    [data-testid="stSidebarNav"] button {
        /* CSS untuk tombol sidebar tetap sama */
    }

    /* --- Header Utama --- */
    .main-header {
        text-align: center;
        padding: 2rem 1rem;
        margin-bottom: 2rem;
    }
    .main-header .title {
        font-size: 3.5rem;
        font-weight: 700;
        color: #FFFFFF;
        letter-spacing: -2px;
    }
    .main-header .subtitle {
        font-size: 1.25rem;
        color: #C0C0D4;
        margin-top: -10px;
    }

    /* --- Tombol Aksi Utama --- */
    .stButton > button {
        font-weight: 600 !important;
        background: linear-gradient(90deg, #8E2DE2, #4A00E0);
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 15px 30px !important;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(142, 45, 226, 0.4);
    }

    /* --- Input URL --- */
    .stTextInput label {
        color: #FFFFFF !important;
        font-weight: 600;
    }
    .stTextInput input {
        border-radius: 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        background-color: rgba(255, 255, 255, 0.05) !important;
        color: #FFFFFF !important;
    }

    /* --- Kartu Hasil Analisis --- */
    .result-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
    .result-card .timestamp {
        font-weight: 600;
        color: #A78BFA;
    }
    .result-card .label {
        background-color: #A78BFA;
        color: #111;
        padding: 2px 8px;
        border-radius: 5px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 5px;
    }

    /* --- Notifikasi Kustom --- */
    [data-testid="stAlert"] {
        border-radius: 10px !important;
        border-width: 0px !important;
        border-left: 4px solid !important;
        background-color: rgba(0,0,0,0.2) !important;
    }
    [data-testid="stAlert"][data-status="error"] { border-left-color: #F43F5E !important; }
    [data-testid="stAlert"][data-status="warning"] { border-left-color: #FBBF24 !important; }
    [data-testid="stAlert"][data-status="success"] { border-left-color: #34D399 !important; }

</style>
""", unsafe_allow_html=True)


# --- HEADER ---
st.markdown("""
<div class="main-header">
    <div class="main-title">🎥 Hayu-IT: HSAL Analysis on Youtube Indonesian Transcripts</div>
    <div class="subtitle">Deteksi otomatis ujaran kebencian dan bahasa kasar dari video YouTube berbahasa Indonesia</div>
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