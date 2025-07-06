import streamlit as st
from datetime import datetime
from PIL import Image
import numpy as np
import cv2
import os
from roboflow import Roboflow


st.set_page_config(page_title="AI Insect Count", layout="wide", page_icon="🦟")

st.markdown(
    """
    <div style="
        background: linear-gradient(to right, #6ab04c, #badc58);
        border-radius: 12px;
        padding: 10px 0;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        color: white;
        margin-bottom: 20px;
    ">
        AI Insect Count
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("""
<style>
.stSelectbox > div > div, .stDateInput > div > div, .stTextInput > div > div > input {
    background-color: #f0f8ff !important;
    border-radius: 8px !important;
}
div.stButton > button {
    background-color: #27ae60;
    color: white;
    font-weight: bold;
    border-radius: 25px;
    padding: 10px 40px;
    font-size: 18px;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_insect_model():
    """โหลดโมเดลตรวจจับแมลงจาก Roboflow"""
    try:
        API_KEY = "3ZQFofNJkviVJdyAb4mG"
        rf = Roboflow(api_key=API_KEY)
        project = rf.workspace("aiinsect").project("ai-insect")
        model = project.version(1).model
        return model
    except Exception as e:
        st.error(f"❌ ไม่สามารถเชื่อมต่อ AI ได้: {e}")
        return None

model = load_insect_model()
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None


col_left, col_right = st.columns([1, 1.5])

with col_left:
    st.subheader("กรอกข้อมูลการตรวจ")
    
    factory = st.selectbox(
        "โรงงาน",
        ["", "SB", "MDC", "MPK", "MPV", "MKS"],
        help="เลือกโรงงานที่ทำการตรวจ"
    )
    inspection_date = st.date_input("วันที่ตรวจ", datetime.today())
    department = st.text_input("หน่วยงาน/แผนก")
    location = st.text_input("พื้นที่ติดตั้ง")
    
    st.subheader("อัปโหลดรูปภาพ")
    source_option = st.radio("เลือกแหล่งที่มาของรูป:", ["อัปโหลดไฟล์", "ถ่ายภาพจากกล้อง"], horizontal=True)

    uploaded_image = None
    if source_option == "อัปโหลดไฟล์":
        uploaded_image = st.file_uploader("เลือกไฟล์รูปภาพ...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    else:
        uploaded_image = st.camera_input("ถ่ายภาพโดยตรง", label_visibility="collapsed")

    if st.button("Analysis"):
        if uploaded_image is None:
            st.warning("⚠️ กรุณาอัปโหลดหรือถ่ายภาพก่อน")
        elif not model:
            st.error("❌ โมเดล AI ยังไม่พร้อมใช้งาน")
        else:
            try:
                with st.spinner("🧠 กำลังวิเคราะห์ภาพ..."):
                    image_pil = Image.open(uploaded_image).convert("RGB")
                    
                    temp_path = "temp_insect_image.jpg"
                    image_pil.save(temp_path)
                    
                    results_json = model.predict(temp_path, confidence=40, overlap=30).json()
                    os.remove(temp_path) 
                    
                    predictions = results_json.get('predictions', [])
                    
                    image_cv2 = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
                    for pred in predictions:
                        x, y, w, h = int(pred['x']), int(pred['y']), int(pred['width']), int(pred['height'])
                        x1, y1 = x - w // 2, y - h // 2
                        x2, y2 = x + w // 2, y + h // 2
                        
                        cv2.rectangle(image_cv2, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{pred['class']} ({pred['confidence']:.2f})"
                        (lw, lh), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        cv2.rectangle(image_cv2, (x1, y1 - lh - 10), (x1 + lw, y1), (0, 255, 0), -1)
                        cv2.putText(image_cv2, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                    annotated_image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

                    st.session_state.analysis_results = {
                        "total_insects": len(predictions),
                        "annotated_image": annotated_image_rgb,
                        "original_image": image_pil
                    }
                
                st.success("✅ วิเคราะห์สำเร็จ!")
                st.rerun()

            except Exception as e:
                st.error("😭 เกิดข้อผิดพลาดร้ายแรงระหว่างการประมวลผล!")
                st.exception(e)

with col_right:
    st.subheader("ผลการวิเคราะห์")
    
    results = st.session_state.get('analysis_results')

    if results:
        st.metric("จำนวนแมลงทั้งหมด (Total Insects)", f"{results.get('total_insects', 0)} ตัว")
        st.markdown("---")
        st.markdown("#### รูปกระดานกาวที่ Label (Picture Label)")
        st.image(results.get("annotated_image"), use_container_width=True)
    else:
        st.info("อัปโหลดหรือถ่ายภาพ จากนั้นกดปุ่ม 'Analysis' เพื่อดูผลลัพธ์ที่นี่")

        placeholder_image = np.full((400, 600, 3), 240, dtype=np.uint8)
        cv2.putText(placeholder_image, "Analysis Result Here", (100, 200), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (180, 180, 180), 2)
        st.image(placeholder_image, use_container_width=True)
