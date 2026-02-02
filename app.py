import streamlit as st
import cv2
import numpy as np
from PIL import Image
from logic import load_vla_model, process_video_step

st.set_page_config(page_title="AmbiguityShield")

st.title("AmbiguityShield: VLA Data Auditor")
st.markdown("Detecting instruction ambiguity via Action Token Entropy.")

# 1. Sidebar - Model Loading
with st.sidebar:
    st.header("Model Settings")
    if 'model' not in st.session_state:
        if st.button("Initialize OpenVLA (7B)"):
            with st.spinner("Loading 4-bit model... (takes ~2 mins)"):
                st.session_state.model, st.session_state.processor = load_vla_model()
                st.success("Model Loaded!")

# 2. Main UI - Inputs
uploaded_video = st.file_uploader("Upload Trajectory Video", type=["mp4", "mov"])
instruction = st.text_input("Human Instruction", placeholder="e.g., 'Pick up the block'")

if uploaded_video and instruction and 'model' in st.session_state:
    if st.button("Audit Quality"):
        # Temporary file handling
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_video.read())
            
        cap = cv2.VideoCapture("temp_video.mp4")
        entropy_values = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 3. Processing Loop (Simplified for demo)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(0, frame_count, 10): # Process every 10th frame for speed
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret: break
            
            # Convert frame for model
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Get Entropy from logic.py
            e_score = process_video_step(st.session_state.model, st.session_state.processor, img, instruction)
            entropy_values.append(e_score)
            
            progress_bar.progress(i / frame_count)
            status_text.text(f"Processing Frame {i}/{frame_count}...")

        # 4. Final Verdict Display
        max_e = max(entropy_values)
        st.subheader("Analysis Results")
        
        # Threshold for Ambiguity
        if max_e > 2.5: 
            st.error(f"AMBIGUITY DETECTED (Max Entropy: {max_e:.2f})")
            st.write("The model is confused. Instruction needs more detail.")
        else:
            st.success(f"HIGH SIGNAL (Max Entropy: {max_e:.2f})")
            st.write("The instruction is clear for the VLA model.")
            
        st.line_chart(entropy_values)
        cap.release()