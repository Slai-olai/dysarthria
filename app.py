import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import librosa
import librosa.display
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image
from huggingface_hub import hf_hub_download
from audio_recorder_streamlit import audio_recorder
import io

# ──────────────────────────────────────────────────────────
# KONFIGURASI
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Deteksi Dysarthria - ViT",
    page_icon="🎙️",
    layout="centered"
)

HF_REPO_ID = "ariif-rahmaan/torgo-dysarthria-vit"
HF_FILENAME = "best_vit_model.pth"
SR = 16000
N_MELS = 128
DURATION = 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ──────────────────────────────────────────────────────────
# LOAD MODEL
# ──────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
    model = models.vit_b_16(weights=None)
    model.heads.head = nn.Linear(model.heads.head.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# ──────────────────────────────────────────────────────────
# PREPROCESSING
# ──────────────────────────────────────────────────────────
def audio_to_melspectrogram(file_obj, sr=SR, n_mels=N_MELS, duration=DURATION):
    y, sr = librosa.load(file_obj, sr=sr, duration=duration)
    target_length = sr * duration
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db, y, sr


def mel_to_tensor(mel_db):
    mel_norm = ((mel_db - mel_db.min()) / (mel_db.max() - mel_db.min()) * 255).astype(np.uint8)
    img = Image.fromarray(mel_norm).convert('RGB').resize((224, 224))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)


def predict(model, img_tensor):
    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        pred = int(np.argmax(probs))
    return pred, probs


def show_results(audio_source):
    with st.spinner("Memproses audio dan menjalankan model..."):
        try:
            mel_db, y_audio, sr = audio_to_melspectrogram(audio_source)
            img_tensor = mel_to_tensor(mel_db)
            model = load_model()
            pred, probs = predict(model, img_tensor)
        except Exception as e:
            st.error(f"Gagal memproses audio: {e}")
            st.stop()

    label_map = {0: "Control (Sehat)", 1: "Dysarthric"}
    color_map = {0: "🟢", 1: "🔴"}

    st.divider()
    st.subheader("Hasil Prediksi")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Prediksi", f"{color_map[pred]} {label_map[pred]}")
    with col2:
        st.metric("Confidence", f"{probs[pred]*100:.2f}%")

    # Spektrogram
    st.subheader("Mel Spectrogram")
    fig1, ax1 = plt.subplots(figsize=(8, 3.5))
    img_disp = librosa.display.specshow(
        mel_db, sr=sr, x_axis='time', y_axis='mel', ax=ax1, cmap='magma'
    )
    fig1.colorbar(img_disp, ax=ax1, format='%+2.0f dB')
    ax1.set_title("Mel Spectrogram dari Audio")
    st.pyplot(fig1)

    # Confidence Score Graph
    st.subheader("Confidence Score per Kelas")
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    classes = ["Control", "Dysarthric"]
    bar_colors = ['#2ecc71' if i == pred else '#95a5a6' for i in range(2)]
    bars = ax2.bar(classes, probs * 100, color=bar_colors)
    for bar, prob in zip(bars, probs):
        ax2.text(bar.get_x() + bar.get_width()/2, prob*100 + 2,
                 f"{prob*100:.2f}%", ha='center', fontweight='bold')
    ax2.set_ylabel("Confidence (%)")
    ax2.set_ylim(0, 110)
    ax2.set_title("Distribusi Probabilitas Prediksi")
    st.pyplot(fig2)

    st.divider()
    st.caption(
        "⚠️ Aplikasi ini dibuat untuk keperluan tugas akademik (Mata Kuliah Pengenalan Pola) "
        "dan tidak boleh digunakan sebagai alat diagnosis medis."
    )


# ──────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────
st.title("🎙️ Deteksi Dysarthria pada Suara")
st.markdown(
    "Aplikasi ini menggunakan **Vision Transformer (ViT-B/16)** yang di-fine-tune "
    "pada representasi **Mel Spectrogram** dari dataset **TORGO** untuk membedakan "
    "suara penderita dysarthria dengan suara normal (control)."
)

st.divider()

# Tab: Upload vs Rekam
tab1, tab2 = st.tabs(["📂 Upload File WAV", "🎙️ Rekam Suara"])

with tab1:
    uploaded_file = st.file_uploader(
        "Upload file audio (.wav)",
        type=["wav"],
        help="Durasi ideal 1-3 detik. File lebih pendek akan otomatis di-padding."
    )
    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        show_results(uploaded_file)
    else:
        st.info("Silakan upload file audio berformat .wav untuk memulai prediksi.")

with tab2:
    st.markdown("Klik tombol mikrofon di bawah untuk mulai merekam, klik lagi untuk berhenti.")
    audio_bytes = audio_recorder(
        text="",
        recording_color="#e74c3c",
        neutral_color="#2c3e50",
        icon_name="microphone",
        icon_size="3x",
        pause_threshold=3.0
    )
    if audio_bytes is not None:
        st.audio(audio_bytes, format="audio/wav")
        audio_file = io.BytesIO(audio_bytes)
        show_results(audio_file)
    else:
        st.info("Tekan tombol mikrofon untuk mulai merekam suara.")

st.divider()
st.caption("Model: ViT-B/16 fine-tuned • Dataset: TORGO • Dibuat oleh Muh. Arif Rahman Gani")
