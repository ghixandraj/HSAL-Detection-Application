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
st.set_page_config(
    page_title="Hayu-IT: HSAL Analysis", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS from external file
def load_css():
    with open("styles.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load CSS
load_css()

# ✅ Header Section
def render_header():
    st.markdown("""
    <div class="header-container">
        <div class="header-content">
            <h1 class="main-title">🧠 Hayu-IT: HSAL Analysis</h1>
            <p class="subtitle">Deteksi Otomatis Ujaran Kebencian dan Bahasa Kasar dari Video YouTube Berbahasa Indonesia</p>
            <div class="header-stats">
                <div class="stat-item">
                    <span class="stat-number">13</span>
                    <span class="stat-label">Kategori Label</span>
                </div>
                <div class="stat-item">
                    <span class="stat-number">99.2%</span>
                    <span class="stat-label">Akurasi Model</span>
                </div>
                <div class="stat-item">
                    <span class="stat-number">ID</span>
                    <span class="stat-label">Bahasa Indonesia</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ✅ Sidebar Content
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <h2>📋 Informasi Sistem</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h3>🔍 Fitur Utama</h3>
            <ul>
                <li>Ekstraksi transkrip video YouTube berbahasa Indonesia</li>
                <li>Deteksi ujaran kebencian dan bahasa kasar otomatis</li>
                <li>Analisis sentimen dengan timestamp</li>
                <li>Model IndoBERTweet + BiGRU + MAML</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h3>📖 Cara Penggunaan</h3>
            <ol>
                <li>Masukkan URL video YouTube</li>
                <li>Pastikan video memiliki subtitle Bahasa Indonesia</li>
                <li>Klik tombol "Analisis Video"</li>
                <li>Tunggu proses selesai dan lihat hasilnya</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card warning">
            <h3>⚠️ Catatan Penting</h3>
            <p>Waktu proses tergantung pada durasi video. Video dengan transkrip auto-generated mungkin mengandung kesalahan.</p>
        </div>
        """, unsafe_allow_html=True)

# ✅ Arsitektur model (sama seperti sebelumnya)
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
    "HS_Weak", "HS_Moderate", "HS_Strong", "PS"
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

# Fungsi-fungsi utility (sama seperti sebelumnya)
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
                return None, None, None

        model = IndoBERTweetBiGRU(bert=bert, hidden_size=512, num_classes=13)

        if os.path.exists(safetensors_path):
            try:
                state_dict = load_file(safetensors_path)
                model.load_state_dict(state_dict)
                model.to(device)
            except Exception as e:
                st.error(f"❌ Gagal memuat model dari SafeTensors: {str(e)}")
                return None, None, None

        model.eval()
        return model, tokenizer, device
    except Exception as e:
        st.error(f"Error loading model/tokenizer: {str(e)}")
        return None, None, None

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

def render_results(problematic_sentences_details, total_sentences, problematic_sentences_count):
    """Render hasil analisis dengan styling yang clean"""
    
    # Summary Card
    st.markdown("""
    <div class="results-summary">
        <h2>📊 Ringkasan Hasil Analisis</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_sentences}</div>
            <div class="metric-label">Total Kalimat</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card warning">
            <div class="metric-value">{problematic_sentences_count}</div>
            <div class="metric-label">Kalimat Bermasalah</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        percentage = (problematic_sentences_count / total_sentences) * 100 if total_sentences > 0 else 0
        st.markdown(f"""
        <div class="metric-card {'danger' if percentage > 10 else 'success'}">
            <div class="metric-value">{percentage:.1f}%</div>
            <div class="metric-label">Persentase</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Warning untuk konten bermasalah
    if percentage > 10:
        st.markdown("""
        <div class="alert alert-danger">
            <strong>⚠️ PERINGATAN:</strong> Lebih dari 10% konten video ini terdeteksi sebagai konten tidak positif dan tidak layak dikonsumsi secara umum.
        </div>
        """, unsafe_allow_html=True)
    
    # Detail kalimat bermasalah
    if problematic_sentences_details:
        st.markdown("""
        <div class="section-header">
            <h3>🚨 Kalimat Bermasalah Terdeteksi</h3>
        </div>
        """, unsafe_allow_html=True)
        
        for idx, detail in enumerate(problematic_sentences_details, 1):
            with st.expander(f"Kalimat {idx} - {detail['timestamp']}", expanded=False):
                st.markdown(f"""
                <div class="sentence-detail">
                    <div class="sentence-text">"{detail['kalimat']}"</div>
                    <div class="sentence-meta">
                        <span class="timestamp">⏱️ {detail['timestamp']}</span>
                        <span class="labels">🏷️ {', '.join(detail['label_terdeteksi'])}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Probabilitas detail
                st.markdown("**Detail Probabilitas:**")
                prob_cols = st.columns(2)
                for i, (label, prob) in enumerate(detail['probabilitas'].items()):
                    with prob_cols[i % 2]:
                        st.write(f"• **{label}**: {prob}")
    else:
        st.markdown("""
        <div class="alert alert-success">
            <strong>✅ Hasil Positif:</strong> Tidak terdeteksi adanya hate speech atau konten bermasalah dalam transkrip ini.
        </div>
        """, unsafe_allow_html=True)

# 🎯 Main App
def main():
    # Render header
    render_header()
    
    # Render sidebar
    render_sidebar()
    
    # Main content area
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    
    # SearchAPI key
    api_key = "Quq35w6JgdtV1fJcnACFK4qF"

    # Session state
    if "is_analyzing" not in st.session_state:
        st.session_state.is_analyzing = False
    if "analysis_done" not in st.session_state:
        st.session_state.analysis_done = False

    # Input section
    st.markdown("""
    <div class="input-section">
        <h2>🔗 Masukkan URL Video YouTube</h2>
        <p>Pastikan video memiliki subtitle atau transkrip dalam Bahasa Indonesia</p>
    </div>
    """, unsafe_allow_html=True)
    
    youtube_url = st.text_input("URL Video:", placeholder="https://www.youtube.com/watch?v=...")

    if youtube_url:
        video_id = extract_video_id(youtube_url)
        if video_id:
            # Video preview
            st.markdown("""
            <div class="video-section">
                <h3>📺 Preview Video</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.video(f"https://www.youtube.com/watch?v={video_id}")

            # Load model
            with st.spinner("🔄 Memuat model dan tokenizer..."):
                model, tokenizer, device = load_model_tokenizer()

            if model is None or tokenizer is None or device is None:
                st.error("❌ Gagal memuat model. Periksa koneksi dan file model.")
                return

            # Analysis button
            if st.button("🚀 Mulai Analisis", type="primary", use_container_width=True):
                st.session_state.is_analyzing = True
                st.session_state.analysis_done = False

                # Get transcript
                with st.spinner("📥 Mengambil transkrip video..."):
                    transcript_data = get_transcript_from_searchapi(video_id, api_key)

                if not transcript_data or "transcripts" not in transcript_data:
                    st.error("❌ Gagal mengambil transcript. Pastikan video memiliki subtitle bahasa Indonesia.")
                    return

                transcript_entries = transcript_data["transcripts"]
                available_languages = transcript_data.get("available_languages", [])
                is_not_auto_generated = any(lang["name"] == "Indonesian" for lang in available_languages)

                if not is_not_auto_generated:
                    st.warning("⚠️ Transkrip ini auto-generated dan mungkin mengandung kesalahan.")

                full_text = " ".join([entry['text'] for entry in transcript_entries])
                st.success("✅ Transkrip berhasil diambil!")

                # Transcript preview
                with st.expander("📄 Preview Transkrip"):
                    st.text_area("", full_text[:1000] + ("..." if len(full_text) > 1000 else ""), height=200)

                # Process sentences
                sentences = re.split(r'(?<=[.!?])\s+|\n', full_text)
                clean_sentences = [s.strip() for s in sentences if s.strip()]

                if not clean_sentences:
                    st.warning("Tidak ada kalimat yang dapat dianalisis.")
                    return

                # Analysis progress
                problematic_sentences_count = 0
                problematic_sentences_details = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                start_time = time.time()

                for i, sentence in enumerate(clean_sentences):
                    status_text.text(f"Menganalisis kalimat {i+1}/{len(clean_sentences)}")
                    
                    detected_labels, label_probs = predict_sentence(sentence, model, tokenizer, device)
                    is_problematic = any(label != "PS" for label in detected_labels)

                    if is_problematic:
                        problematic_sentences_count += 1
                        
                        # Find timestamp
                        matched_entry = next((entry for entry in transcript_entries if sentence.strip().startswith(entry['text'].strip()[:10])), None)
                        timestamp = matched_entry["start"] if matched_entry else None
                        timestamp_str = f"{int(timestamp // 60):02d}:{int(timestamp % 60):02d}" if timestamp is not None else "??:??"

                        # Sort active labels
                        sorted_active = sorted(
                            [(LABEL_DESCRIPTIONS[label], float(prob)) for label, prob in label_probs.items() if prob > 0.5 and label != "PS"],
                            key=lambda x: x[1],
                            reverse=True
                        )

                        all_probs = {LABEL_DESCRIPTIONS[label]: f"{float(prob):.1%}" for label, prob in label_probs.items()}

                        problematic_sentences_details.append({
                            "kalimat": sentence,
                            "timestamp": timestamp_str,
                            "label_terdeteksi": [label for label, _ in sorted_active],
                            "probabilitas": all_probs
                        })

                    progress_bar.progress((i + 1) / len(clean_sentences))

                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()

                # Calculate processing time
                end_time = time.time()
                elapsed_time = end_time - start_time
                minutes = int(elapsed_time // 60)
                seconds = int(elapsed_time % 60)

                st.info(f"⏱️ Waktu pemrosesan: {minutes} menit {seconds} detik")

                # Render results
                render_results(problematic_sentences_details, len(clean_sentences), problematic_sentences_count)

                st.session_state.is_analyzing = False
                st.session_state.analysis_done = True

        else:
            st.error("❌ URL tidak valid. Masukkan URL YouTube yang benar.")
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()