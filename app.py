import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import logic

st.set_page_config(page_title="AmbiguityShield | Phase-Aware Auditor", layout="wide")

st.markdown("""
    <style>
    .log-container {
        height: 150px;
        overflow-y: auto;
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 5px;
        padding: 10px;
        font-family: monospace;
        font-size: 0.8rem;
        color: #8b949e;
        margin-bottom: 20px;
    }
    .stMetric { background-color: #1d2127; border-radius: 8px; padding: 10px; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

st.title("AmbiguityShield: VLA Data Quality Auditor")
st.markdown("*Auditing the 'Action Zone' to filter out startup and shutdown noise.*")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    threshold = st.slider("Action Zone Threshold", 0.0, 5.0, 3.0, help="Lower = stricter quality control.")
    num_samples = st.select_slider("Frame Density", options=[10, 20, 50, 100], value=50)
    
    st.divider()
    if 'model' not in st.session_state:
        if st.button("Initialize OpenVLA", use_container_width=True):
            with st.status("Loading 7B Model...") as status:
                st.session_state.model, st.session_state.processor = logic.load_vla_model()
                status.update(label="System Ready!", state="complete")

# Main Interface
instr = st.text_input("Human Instruction", value="pick up the duster")
video_file = st.file_uploader("Upload Trajectory (MP4)", type=['mp4'])

if video_file and instr and 'model' in st.session_state:
    if st.button("Run Phase-Aware Audit", use_container_width=True):
        with open("temp.mp4", "wb") as f: f.write(video_file.read())
        cap = cv2.VideoCapture("temp.mp4")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total_frames - 1, num_samples).astype(int)
        
        results, frames_cache = [], []
        
        # Logging window
        st.write("### Live Analysis")
        log_placeholder = st.empty() 
        log_history = []
        progress_bar = st.progress(0)
        
        for idx, f_idx in enumerate(indices):
            log_msg = f"Frame {f_idx}: Calculating Entropy..."
            log_history.append(log_msg)
            log_html = f"<div class='log-container'>{'<br>'.join(log_history[-10:])}</div>"
            log_placeholder.markdown(log_html, unsafe_allow_html=True)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret: break
            
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            entropy = logic.process_video_step(st.session_state.model, st.session_state.processor, img, instr)
            
            results.append({"Frame": f_idx, "Entropy": entropy})
            frames_cache.append(img)
            progress_bar.progress((idx + 1) / num_samples)
        
        log_placeholder.empty()
        df = pd.DataFrame(results)
        
        # Phase Aware Calculation
        # We define the 'Action Zone' as the middle 60% of the video
        start_idx = int(len(df) * 0.2)
        end_idx = int(len(df) * 0.8)
        action_zone_df = df.iloc[start_idx:end_idx]
        
        az_avg = action_zone_df['Entropy'].mean()
        overall_peak = df['Entropy'].max()
        
        # Results dashboard
        st.divider()
        m1, m2, m3 = st.columns(3)
        
        with m1:
            st.metric("Action Zone Avg", f"{az_avg:.2f}")
        with m2:
            st.metric("Overall Peak", f"{overall_peak:.2f}")
        with m3:
            verdict = "PASS" if az_avg < threshold else "REJECT"
            st.metric("Final Verdict", verdict)

        # Visual Chart with Highlighted Action Zone
        st.subheader("Entropy Profile")
        df['Smoothing'] = df['Entropy'].rolling(window=3, center=True).mean().fillna(df['Entropy'])
        st.line_chart(df.set_index('Frame')['Smoothing'])
        st.caption("Note: The verdict is based on the average of the middle 'Action Zone' (20% to 80% mark).")

        # Critical failure points
        st.subheader("Deep Dive: Top 3 High-Uncertainty Moments")
        top_indices = df.nlargest(3, 'Entropy').index.tolist()
        cols = st.columns(3)
        for i, r_idx in enumerate(top_indices):
            with cols[i]:
                st.image(frames_cache[r_idx], use_container_width=True)
                st.error(f"**Frame {df.iloc[r_idx]['Frame']} | Entropy: {df.iloc[r_idx]['Entropy']:.2f}**")