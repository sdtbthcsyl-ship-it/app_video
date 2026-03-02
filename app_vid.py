import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
import time

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN
# ==========================================
st.set_page_config(page_title="AI Crack Detection - Offline Fix", page_icon="🎥", layout="wide")

st.markdown("""
    <style>
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border: 1px solid #e0e0e0; }
    .stButton>button { width: 100%; background-color: #28a745; color: white; font-weight: bold; height: 3em; }
    </style>
""", unsafe_allow_html=True)

# Tải mô hình an toàn
@st.cache_resource
def load_model():
    model_path = 'best.pt'
    try:
        # Nếu gặp lỗi 'Conv' object has no attribute 'bn', lệnh này sẽ cố gắng tải lại
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"⚠️ Lỗi xung đột phiên bản thư viện: {e}")
        st.info("💡 Cách sửa: Hãy chạy lệnh cập nhật thư viện ở hướng dẫn bên dưới.")
        return None

model = load_model()

st.title("🎥 Hệ thống AI Phân tích Video (Bản sửa lỗi Windows)")
st.write("Xử lý video Offline - Tự động đếm vết nứt không trùng lặp.")

# ==========================================
# 2. XỬ LÝ VIDEO
# ==========================================
uploaded_video = st.file_uploader("Chọn video từ máy tính (.mp4, .avi)...", type=['mp4', 'avi', 'mov'])

if uploaded_video is not None and model is not None:
    conf_thresh = st.sidebar.slider("Độ nhạy AI", 0.1, 1.0, 0.25, 0.05)
    
    if st.button("🚀 BẮT ĐẦU PHÂN TÍCH"):
        # SỬA LỖI PermissionError (WinError 32):
        # Tạo file tạm nhưng KHÔNG dùng chế độ tự động xóa của tempfile
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        tfile_path = tfile.name
        tfile.close() # Đóng luồng ghi ngay để Windows không khóa file
        
        cap = cv2.VideoCapture(tfile_path)
        st_frame = st.empty()
        st_metric = st.empty()
        unique_ids = set()
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Xử lý khung hình
                frame = cv2.resize(frame, (854, 480))
                
                # Sử dụng Tracking để đếm vết nứt duy nhất
                results = model.track(frame, conf=conf_thresh, persist=True, verbose=False)
                
                annotated_frame = results[0].plot()
                
                if results[0].boxes.id is not None:
                    ids = results[0].boxes.id.cpu().numpy().astype(int)
                    unique_ids.update(ids)
                
                # Hiển thị
                st_frame.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                with st_metric.container():
                    st.metric("Tổng số vết nứt độc lập", f"{len(unique_ids)} vị trí")
        
        except Exception as e:
            st.error(f"Lỗi khi đang quét: {e}")
            
        finally:
            # GIẢI QUYẾT TRIỆT ĐỂ LỖI PermissionError:
            cap.release() # Giải phóng file khỏi OpenCV
            cv2.destroyAllWindows()
            
            # Chờ 1 giây để Windows nhả tệp hoàn toàn
            time.sleep(1) 
            
            try:
                if os.path.exists(tfile_path):
                    os.remove(tfile_path) # Bây giờ mới xóa file tạm
            except:
                # Nếu vẫn lỗi quyền, bỏ qua để chương trình không bị văng
                pass
                
        st.success("✅ Đã hoàn thành phân tích!")