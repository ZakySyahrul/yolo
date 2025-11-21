import streamlit as st
from model_loader import load_model
from detect_video import process_video
from utils import get_linechart_data, get_barchart_data
import pandas as pd
import tempfile
import cv2
import os
import altair as alt
import numpy as np
from sklearn.cluster import KMeans

# ---------------------------------------------------------
# 1. PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="YOLO Traffic Analyzer",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# 2. CUSTOM CSS (NEON EFFECT)
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    .stApp, .block-container, .stMarkdown { font-family: 'Inter', sans-serif; color: #e6f7f5; }
    .stApp { background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%); }
    .metric-container { text-align: center; padding: 10px; margin-bottom: 20px; }
    .metric-value { 
        font-size: 48px; font-weight: 900; color: #ffffff; 
        text-shadow: 0 0 10px rgba(126, 231, 214, 0.8), 0 0 20px rgba(126, 231, 214, 0.6);
    }
    .metric-label { font-size: 14px; color: #94a3b8; text-transform: uppercase; letter-spacing: 2px; font-weight: 600; margin-top: -10px; }
    h1, h2, h3, h4 { color: #7ee7d6 !important; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# 3. HEADER & SETTINGS
# ---------------------------------------------------------
st.markdown('<h1 style="text-align: center; margin-bottom: 10px;">🚦 YOLO Traffic Analyzer</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #94a3b8; margin-bottom: 40px;">Upload video CCTV/Lalu lintas untuk mendeteksi kendaraan dan menganalisis kepadatan secara otomatis.</p>', unsafe_allow_html=True)

st.markdown("### ⚙️ Konfigurasi Analisis")
col_s1, col_s2 = st.columns(2)
with col_s1:
    model_choice = st.selectbox("🔍 Pilih Versi Model YOLO", ["YOLO v8 Final", "YOLO v12 Final"])
with col_s2:
    analysis_choice = st.selectbox("⏱️ Granularitas Waktu Analisis", ["Per Frame", "Per Detik", "Per Menit", "Per Jam"])

st.markdown("---")

model_path_map = {"YOLO v8 Final": "best_v8.pt", "YOLO v12 Final": "best_v12.pt"}
group_key_map = {"Per Frame": None, "Per Detik": "second", "Per Menit": "minute", "Per Jam": "hour"}
selected_model_path = model_path_map[model_choice]
group_key = group_key_map[analysis_choice]

@st.cache_resource
def load_yolo(path):
    return load_model(path)

model = load_yolo(selected_model_path)

# ---------------------------------------------------------
# 4. UPLOAD SECTION
# ---------------------------------------------------------
st.markdown("### 📂 Upload Video")
video_file = st.file_uploader("", type=["mp4", "avi", "mov"])

# ---------------------------------------------------------
# 5. MAIN PROCESSING LOGIC
# ---------------------------------------------------------
if video_file:
    # Buat temp file untuk INPUT
    tfile_input = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile_input.write(video_file.read())
    
    # Buat temp file untuk OUTPUT (Fix Error Path)
    tfile_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_path_full = tfile_output.name 

    with st.status("🔄 Sedang memproses video...", expanded=True) as status:
        st.write(f"Menggunakan model: **{model_choice}**")
        st.write("Deteksi objek sedang berjalan...")
        
        # PASSING output_path_full KE FUNGSI
        df, final_vid_path, augmented, model_names = process_video(model, tfile_input.name, output_path=output_path_full)
        
        st.write("Menghitung statistik kepadatan...")
        status.update(label="✅ Analisis Selesai!", state="complete", expanded=False)

    # Data Aggregation
    df_processed = df.copy()
    if group_key is not None:
        df_processed[group_key] = df_processed[group_key].round().astype(int)
        type_cols = [c for c in df.columns if c.startswith("type_")]
        agg = {c: "sum" for c in type_cols}
        agg["total"] = "sum"
        df_processed = df_processed.groupby(group_key).agg(agg).reset_index()

    st.markdown("---")

    # A. DASHBOARD METRICS
    total_vehicles = df_processed["total"].sum()
    peak_traffic = df_processed["total"].max()
    avg_traffic = round(df_processed["total"].mean(), 1)

    st.markdown("### 📊 Ringkasan Statistik")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="metric-container"><div class="metric-value">{total_vehicles:,}</div><div class="metric-label">Total Kendaraan</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-container"><div class="metric-value">{peak_traffic}</div><div class="metric-label">Puncak ({analysis_choice})</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-container"><div class="metric-value">{avg_traffic}</div><div class="metric-label">Rata-rata Volume</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # B. CHARTS
    row2_col1, row2_col2 = st.columns([1.5, 1])
    with row2_col1:
        st.markdown(f'#### 📈 Tren Lalu Lintas ({analysis_choice})')
        line_data = get_linechart_data(df_processed)
        chart = alt.Chart(line_data).mark_line(point=True, strokeWidth=3).encode(
            x=alt.X(line_data.columns[0] + ":Q", title="Waktu"),
            y=alt.Y("count:Q", title="Jumlah"),
            color=alt.value("#7ee7d6"), tooltip=[line_data.columns[0], "count"]
        ).properties(height=300).interactive()
        st.altair_chart(chart, use_container_width=True)

    with row2_col2:
        st.markdown('#### 🏆 Komposisi Kendaraan')
        bar_data = get_barchart_data(df_processed)
        bar_df = bar_data.reset_index()
        bar_df.columns = ["type", "count"]
        bar = alt.Chart(bar_df).mark_bar(cornerRadius=4).encode(
            x=alt.X("count:Q", title="Jumlah"), y=alt.Y("type:N", sort='-x', title="Jenis"),
            color=alt.Color("type:N", legend=None, scale=alt.Scale(scheme="tealblues")), tooltip=["type", "count"]
        ).properties(height=300)
        st.altair_chart(bar, use_container_width=True)

    st.markdown("---")

    # C. CLUSTERING
    st.markdown('#### 🔵 Analisis Kepadatan (K-Means Clustering)')
    if len(df_processed) > 2: 
        X = df_processed["total"].values.reshape(-1, 1)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
        raw_labels = kmeans.fit_predict(X)
        centers = kmeans.cluster_centers_.flatten()
        order = np.argsort(centers)
        centroid_to_label = {order[0]: "Lancar", order[1]: "Ramai", order[2]: "Padat"}
        mapped_labels = [centroid_to_label[c] for c in raw_labels]
        scatter_df = pd.DataFrame({"index": df_processed[df_processed.columns[0]], "total": df_processed["total"], "cluster": mapped_labels})
        cluster_colors = alt.Scale(domain=['Lancar', 'Ramai', 'Padat'], range=['#4ade80', '#facc15', '#f87171'])
        scatter_chart = alt.Chart(scatter_df).mark_circle(size=80).encode(
            x=alt.X("index:Q", title=analysis_choice), y=alt.Y("total:Q", title="Total Kendaraan"),
            color=alt.Color("cluster:N", scale=cluster_colors, title="Status"), tooltip=["index", "total", "cluster"]
        ).properties(height=350).interactive()
        st.altair_chart(scatter_chart, use_container_width=True)
    else:
        st.warning("Data terlalu sedikit untuk analisis clustering.")
    
    st.markdown("---")

    # D. TABS (VISUALISASI & VIDEO HANDLING)
    tab1, tab2, tab3 = st.tabs(["🎥 Visualisasi Hasil", "📋 Data Lengkap", "🧪 Augmentasi"])

    with tab1:
        st.markdown("### Perbandingan Video")
        col_vid1, col_vid2 = st.columns(2)
        
        with col_vid1:
            st.markdown("**📹 Video Asli (Input)**")
            st.video(tfile_input.name)
            
        with col_vid2:
            st.markdown("**🤖 Video Hasil Deteksi (Output)**")
            
            # --- ERROR HANDLING VIDEO ---
            try:
                # Cek apakah file ada dan tidak kosong (0 bytes)
                if os.path.exists(final_vid_path) and os.path.getsize(final_vid_path) > 1000:
                    # Baca sebagai binary agar Streamlit tidak bingung dengan Path
                    with open(final_vid_path, 'rb') as v_file:
                        video_bytes = v_file.read()
                        st.video(video_bytes)
                else:
                    st.error("Video output gagal digenerate atau file rusak (Codec Issue). Tapi statistik di atas Valid!")
            except Exception as e:
                st.error(f"Gagal memuat video: {e}")

        st.markdown("---")
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.download_button("📄 Download Laporan CSV", df_processed.to_csv(index=False).encode("utf-8"), "traffic_report.csv", use_container_width=True)
        with col_d2:
            try:
                with open(final_vid_path, "rb") as f:
                    st.download_button("🎥 Download Video Hasil", f, file_name="video_output_yolo.mp4", use_container_width=True)
            except:
                st.write("Video belum tersedia untuk download.")

    with tab2:
        st.dataframe(df_processed.style.background_gradient(cmap="YlGnBu", subset=["total"]), use_container_width=True)

    with tab3:
        st.markdown("Sample frame hasil augmentasi:")
        cols = st.columns(3)
        for idx, img in enumerate(augmented):
            with cols[idx % 3]:
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"Aug #{idx+1}", use_column_width=True)

st.markdown("---")
st.caption("Developed by Team 5 FGD • Gunadarma University")
