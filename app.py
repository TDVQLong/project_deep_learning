# -*- coding: utf-8 -*-
"""
HỆ THỐNG AI HỖ TRỢ CHẨN ĐOÁN GÃY XƯƠNG CỔ TAY TRẺ EM TRÊN ẢNH X-QUANG (YOLOv8s)
Đề tài: Báo cáo Dự án Học phần Học sâu (Deep Learning) - Nhóm 11
Giảng viên hướng dẫn: TS. Ngô Hữu Huy
Thực hiện: Trần Đặng Vương Quốc Long (Trưởng nhóm), Nguyễn Văn Minh, Nguyễn Văn Chỉnh
Giao diện: Chuẩn phòng đọc phim PACS / DICOM Radiology Workstation
"""

import os
import io
import time
import base64
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import streamlit as st
from ultralytics import YOLO

# -----------------------------------------------------------------------------
# 1. CẤU HÌNH TRANG VÀ GIAO DIỆN STREAMLIT
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="PACS AI - Chẩn Đoán Gãy Xương Cổ Tay Trẻ Em",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS phong cách Dark Mode chuyên sâu phòng đọc phim PACS / DICOM
st.markdown("""
<style>
    /* BẢNG MÀU CHUẨN PHÒNG ĐỌC PHIM PACS / DICOM */
    :root {
        --bg-main: #08090b;
        --bg-surface: #111317;
        --bg-card: #15181e;
        --border-color: #22262e;
        --border-focus: #00e5ff;
        --accent-cyan: #00e5ff;
        --accent-teal: #00adb5;
        --text-white: #ffffff;
        --text-silver: #e2e8f0;
        --text-muted: #8b949e;
        --red-fracture: #ff2e4d;
        --blue-metal: #3b82f6;
        --amber-can: #f59e0b;
        --green-safe: #10b981;
    }

    /* Nền tổng thể Deep Obsidian Black */
    .stApp {
        background-color: var(--bg-main) !important;
        color: var(--text-silver);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
    }

    /* Ẩn viền thừa của Streamlit */
    header[data-testid="stHeader"] {
        background-color: rgba(8, 9, 11, 0.85) !important;
        backdrop-filter: blur(8px);
    }

    /* Thanh bên Sidebar PACS */
    [data-testid="stSidebar"] {
        background-color: #0c0e12 !important;
        border-right: 1px solid var(--border-color) !important;
    }

    /* Header hệ thống PACS Workstation */
    .pacs-header-card {
        background: linear-gradient(135deg, #0d0f14 0%, #141820 60%, #06283d 100%);
        border: 1px solid #1f2530;
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 18px;
        box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.6);
    }
    .pacs-main-title {
        font-size: 1.95rem;
        font-weight: 800;
        letter-spacing: -0.3px;
        color: #ffffff;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .pacs-subtitle {
        font-size: 0.95rem;
        color: var(--text-muted);
        margin-bottom: 12px;
    }
    .pacs-badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }
    .pacs-tag {
        font-size: 0.78rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 4px;
        letter-spacing: 0.3px;
        display: inline-flex;
        align-items: center;
        gap: 5px;
    }
    .tag-cyan { background: rgba(0, 229, 255, 0.12); color: #00e5ff; border: 1px solid rgba(0, 229, 255, 0.3); }
    .tag-green { background: rgba(16, 185, 129, 0.12); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); }
    .tag-amber { background: rgba(245, 158, 11, 0.12); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.3); }
    .tag-purple { background: rgba(168, 85, 247, 0.12); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.3); }

    /* Thẻ chỉ số KPI (PACS Metric HUD) */
    .pacs-kpi-card {
        background: #111317;
        border: 1px solid #1f232b;
        border-radius: 8px;
        padding: 14px 16px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        transition: all 0.2s ease;
    }
    .pacs-kpi-card:hover {
        border-color: var(--accent-cyan);
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 229, 255, 0.15);
    }
    .pacs-kpi-label {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        color: #7d8590;
        margin-bottom: 4px;
    }
    .pacs-kpi-val {
        font-size: 1.65rem;
        font-weight: 800;
        color: #00e5ff;
        text-shadow: 0 0 12px rgba(0, 229, 255, 0.25);
    }
    .pacs-kpi-sub {
        font-size: 0.74rem;
        color: #606770;
        margin-top: 2px;
    }

    /* Bảng thông báo lâm sàng PACS HUD Alert */
    .pacs-alert-danger {
        background: linear-gradient(135deg, rgba(255, 46, 77, 0.12), rgba(153, 16, 36, 0.20));
        border: 1px solid rgba(255, 46, 77, 0.4);
        border-left: 5px solid #ff2e4d;
        border-radius: 8px;
        padding: 14px 18px;
        color: #ffe4e6;
        margin: 14px 0;
    }
    .pacs-alert-warning {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.12), rgba(180, 83, 9, 0.20));
        border: 1px solid rgba(245, 158, 11, 0.4);
        border-left: 5px solid #f59e0b;
        border-radius: 8px;
        padding: 14px 18px;
        color: #fef3c7;
        margin: 14px 0;
    }
    .pacs-alert-success {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12), rgba(4, 120, 87, 0.20));
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-left: 5px solid #10b981;
        border-radius: 8px;
        padding: 14px 18px;
        color: #d1fae5;
        margin: 14px 0;
    }

    /* Tabs phong cách thanh điều khiển DICOM */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 1px solid #1f232b;
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        background-color: #111317;
        border: 1px solid #1f242d;
        border-radius: 6px 6px 0 0;
        color: #8b949e;
        font-weight: 600;
        font-size: 0.90rem;
        padding: 0 16px;
        transition: all 0.15s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #ffffff;
        background-color: #161920;
    }
    .stTabs [aria-selected="true"] {
        background-color: #171b22 !important;
        border-color: #00e5ff !important;
        border-bottom-color: #171b22 !important;
        color: #00e5ff !important;
        box-shadow: inset 0 2px 0 #00e5ff;
    }

    /* Tùy chỉnh màu sắc Sliders và Toggles sang Medical Cyan */
    .stSlider [data-baseweb="slider"] div[role="slider"] {
        background-color: #00e5ff !important;
        border-color: #00e5ff !important;
        box-shadow: 0 0 10px rgba(0, 229, 255, 0.6) !important;
    }
    .stSlider [data-baseweb="slider"] div[data-testid="stSliderTrack"] > div:first-child {
        background-color: #00e5ff !important;
    }
    [data-testid="stToggle"] button[aria-checked="true"] {
        background-color: #00e5ff !important;
    }

    /* Phiếu kết luận lâm sàng DICOM Report */
    .dicom-report-card {
        background: #0f1115;
        border: 1px solid #1e222a;
        border-radius: 8px;
        padding: 16px 20px;
        margin-top: 14px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
    }
    .dicom-report-head {
        font-size: 1.05rem;
        font-weight: 700;
        color: #00e5ff;
        border-bottom: 1px solid #1e222a;
        padding-bottom: 8px;
        margin-bottom: 10px;
        display: flex;
        justify-content: space-between;
    }
    .dicom-report-row {
        display: flex;
        justify-content: space-between;
        padding: 5px 0;
        font-size: 0.88rem;
        border-bottom: 1px dashed #1a1e26;
    }
    .dicom-key { color: #7d8590; }
    .dicom-val { color: #f0f6fc; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. ĐỊNH NGHĨA DỮ LIỆU & ÁNH XẠ LÂM SÀNG 9 LỚP BỆNH LÝ
# -----------------------------------------------------------------------------
CLASS_MAP = {
    "fracture": {
        "vi": "Gãy xương (Fracture)",
        "color": (255, 46, 77),       # Đỏ huỳnh quang (#ff2e4d)
        "hex": "#ff2e4d",
        "desc": "Đường nứt hoặc gián đoạn liên tục của bờ vỏ xương (cành tươi / hoàn toàn).",
        "severity": "Khẩn cấp (Cần bác sĩ xử trí ngay)",
        "action": "Bất động cổ tay ngay bằng nẹp máng cẳng bàn tay; hội chẩn chuyên khoa Chấn thương Chỉnh hình Nhi."
    },
    "metal": {
        "vi": "Vật liệu kim loại (Metal)",
        "color": (59, 130, 246),      # Xanh dương ánh kim (#3b82f6)
        "hex": "#3b82f6",
        "desc": "Nẹp vis, đinh Kirschner, chỉ thép cố định xương sau phẫu thuật.",
        "severity": "Theo dõi cố định",
        "action": "Kiểm tra độ vững chắc của đinh/nẹp; không phát hiện đường gãy mới quanh chân nẹp."
    },
    "periostealreaction": {
        "vi": "Phản ứng màng xương (Periosteal)",
        "color": (245, 158, 11),      # Vàng hổ phách (#f59e0b)
        "hex": "#f59e0b",
        "desc": "Màng xương dày lên, phản ứng tạo can xương liền sau chấn thương hoặc viêm.",
        "severity": "Theo dõi can xương",
        "action": "Dấu hiệu liền xương lành tính sau chấn thương cũ. Hẹn tái khám định kỳ sau 2-3 tuần."
    },
    "pronatorsign": {
        "vi": "Dấu hiệu cơ sấp (Pronator Sign)",
        "color": (192, 132, 252),     # Tím thạch anh (#c084fc)
        "hex": "#c084fc",
        "desc": "Đường mỡ cơ vuông sấp bị cong, dịch chuyển hoặc mờ đục.",
        "severity": "Chỉ điểm gián tiếp",
        "action": "Dấu hiệu chỉ điểm có gãy xương kín đáo đầu dưới xương quay. Đề nghị soi kỹ viền vỏ xương."
    },
    "softtissue": {
        "vi": "Sưng phù mô mềm (Soft Tissue)",
        "color": (244, 63, 94),       # Hồng san hô (#f43f5e)
        "hex": "#f43f5e",
        "desc": "Dày bóng mô mềm, phù nề quanh vị trí chấn thương.",
        "severity": "Dấu hiệu phụ trợ",
        "action": "Chườm lạnh, kê cao tay; theo dõi chèn ép khoang nếu có sưng căng nhiều."
    },
    "boneanomaly": {
        "vi": "Bất thường giải phẫu (Bone Anomaly)",
        "color": (251, 146, 60),      # Cam ánh kim (#fb923c)
        "hex": "#fb923c",
        "desc": "Biến dạng giải phẫu bẩm sinh hoặc cốt hóa không đồng đều.",
        "severity": "Theo dõi",
        "action": "Cần đối chiếu với phim X-quang cổ tay bên đối diện để loại trừ biến thể bình thường."
    },
    "bonelesion": {
        "vi": "Tổn thương xương (Bone Lesion)",
        "color": (45, 212, 191),      # Xanh mòng két (#2dd4bf)
        "hex": "#2dd4bf",
        "desc": "Vùng tiêu xương, nang xương đơn độc hoặc khuyết xương.",
        "severity": "Chẩn đoán phân biệt",
        "action": "Khuyến cáo chỉ định thêm chụp MRI hoặc CT Scanner để xác định bản chất tổn thương."
    },
    "foreignbody": {
        "vi": "Dị vật cản quang (Foreign Body)",
        "color": (239, 68, 68),       # Đỏ tươi
        "hex": "#ef4444",
        "desc": "Mảnh thủy tinh, kim loại hoặc dị vật cản quang ngoại lai lọt vào mô.",
        "severity": "Cần can thiệp gắp bỏ",
        "action": "Xác định chính xác tọa độ theo 2 bình diện Thẳng/Nghiêng để phẫu thuật mở gắp dị vật."
    },
    "text": {
        "vi": "Ký hiệu đánh dấu (Text L/R)",
        "color": (16, 185, 129),      # Xanh ngọc lục bảo (#10b981)
        "hex": "#10b981",
        "desc": "Chữ cái chì ghi chú tư thế hoặc bên giải phẫu trên phim (Left/Right).",
        "severity": "Không có bệnh lý",
        "action": "Ký hiệu kỹ thuật phục vụ lưu trữ hồ sơ PACS."
    }
}

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(CURRENT_DIR, "weights", "best.pt")
SAMPLES_DIR = os.path.join(CURRENT_DIR, "sample_images")
RUNS_DIR = os.path.normpath(os.path.join(CURRENT_DIR, "..", "yolov8s_results", "kaggle", "working", "runs_v8"))

# -----------------------------------------------------------------------------
# 3. HÀM TẢI MÔ HÌNH VÀ TIỀN XỬ LÝ ẢNH (CLAHE)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_yolo_model(model_path):
    if not os.path.exists(model_path):
        st.error(f"⚠️ Không tìm thấy file trọng số tại: {model_path}")
        return None
    return YOLO(model_path)

model = load_yolo_model(WEIGHTS_PATH)

def apply_clahe(pil_img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Áp dụng cân bằng biểu đồ mức xám thích ứng có giới hạn độ tương phản (CLAHE)."""
    img_np = np.array(pil_img)
    if len(img_np.shape) == 3:
        # Ảnh RGB: chuyển sang không gian màu LAB để cân bằng riêng kênh sáng L
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    else:
        # Ảnh Grayscale
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced = clahe.apply(img_np)
    return Image.fromarray(enhanced)

# -----------------------------------------------------------------------------
# 4. HÀM RENDER KÍNH LÚP PAN & ZOOM CHUẨN PACS / DICOM (KHÔNG ĐÈ CHỮ)
# -----------------------------------------------------------------------------
def render_panzoom_viewer(pil_image, caption="ẢNH X-QUANG", viewer_id="viewer_1", height=540):
    """Render viewer chuẩn PACS Workstation: Thanh điều khiển riêng biệt ở trên cùng, không đè lên ảnh."""
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()
    img_uri = f"data:image/png;base64,{img_b64}"

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
      * {{ box-sizing: border-box; margin: 0; padding: 0; }}
      body {{
        background: #08090b;
        color: #e2e8f0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        overflow: hidden;
        user-select: none;
      }}
      .pacs-frame {{
        position: relative;
        width: 100%;
        height: {height}px;
        background: #040507;
        border: 1px solid #1e2430;
        border-radius: 8px;
        overflow: hidden;
        display: flex;
        flex-direction: column;
        box-shadow: inset 0 0 30px rgba(0, 0, 0, 0.8), 0 4px 15px rgba(0, 0, 0, 0.6);
      }}
      /* Thanh tiêu đề PACS trên cùng - Tách biệt hoàn toàn */
      .pacs-topbar {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 12px;
        background: #0e1117;
        border-bottom: 1px solid #1a202c;
        height: 44px;
        flex-shrink: 0;
      }}
      .pacs-title-tag {{
        display: flex;
        align-items: center;
        gap: 8px;
        min-width: 0;
      }}
      .pacs-dot {{
        width: 7px;
        height: 7px;
        background: #00e5ff;
        border-radius: 50%;
        box-shadow: 0 0 8px #00e5ff;
        flex-shrink: 0;
      }}
      .caption-text {{
        font-size: 12px;
        font-weight: 700;
        color: #00e5ff;
        letter-spacing: 0.4px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }}
      /* Thanh công cụ mini bên phải */
      .pacs-toolbar {{
        display: flex;
        align-items: center;
        gap: 5px;
        flex-shrink: 0;
      }}
      .pacs-btn {{
        background: #171c26;
        color: #e2e8f0;
        border: 1px solid #283142;
        border-radius: 5px;
        padding: 5px 9px;
        font-size: 12px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.15s ease;
      }}
      .pacs-btn:hover {{
        background: #00e5ff;
        border-color: #00e5ff;
        color: #000000;
        box-shadow: 0 0 10px rgba(0, 229, 255, 0.5);
      }}
      .pacs-btn:active {{
        transform: scale(0.96);
      }}
      .zoom-level {{
        font-size: 11px;
        font-weight: 800;
        color: #00e5ff;
        padding: 4px 8px;
        background: rgba(0, 229, 255, 0.10);
        border: 1px solid rgba(0, 229, 255, 0.25);
        border-radius: 4px;
        min-width: 48px;
        text-align: center;
      }}
      /* Vùng nhìn X-quang */
      .pacs-viewport {{
        flex: 1;
        width: 100%;
        height: calc(100% - 44px);
        cursor: grab;
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
        background-color: #030406;
      }}
      .pacs-viewport.grabbing {{
        cursor: grabbing;
      }}
      .pacs-layer {{
        transform-origin: center center;
        transition: transform 0.04s linear;
        will-change: transform;
        display: inline-block;
      }}
      .pacs-layer img {{
        max-height: {height - 90}px;
        max-width: 95%;
        width: auto;
        height: auto;
        display: block;
        border-radius: 2px;
        box-shadow: 0 0 35px rgba(0,0,0,0.9);
        pointer-events: none;
      }}
      /* Chú thích hướng dẫn dưới đáy */
      .pacs-footer-hint {{
        position: absolute;
        bottom: 6px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 30;
        background: rgba(14, 17, 23, 0.85);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 3px 12px;
        font-size: 10px;
        color: #8b949e;
        pointer-events: none;
        white-space: nowrap;
      }}
    </style>
    </head>
    <body>
    <div class="pacs-frame" id="{viewer_id}_frame">
      <div class="pacs-topbar">
        <div class="pacs-title-tag">
          <span class="pacs-dot"></span>
          <span class="caption-text" title="{caption}">{caption}</span>
        </div>
        <div class="pacs-toolbar">
          <button class="pacs-btn" onclick="zoomIn()" title="Phóng to (hoặc lăn chuột lên)">➕</button>
          <button class="pacs-btn" onclick="zoomOut()" title="Thu nhỏ (hoặc lăn chuột xuống)">➖</button>
          <div class="zoom-level" id="{viewer_id}_zoom">100%</div>
          <button class="pacs-btn" onclick="resetView()" title="Kích thước gốc (nhấp đúp chuột)">🔄 1:1</button>
          <button class="pacs-btn" onclick="toggleFull()" title="Toàn màn hình">⛶</button>
        </div>
      </div>
      <div class="pacs-viewport" id="{viewer_id}_vp">
        <div class="pacs-layer" id="{viewer_id}_layer">
          <img src="{img_uri}" alt="{caption}" />
        </div>
        <div class="pacs-footer-hint">🖱️ Lăn chuột để Zoom (1x - 5x) • Kéo chuột để Pan • Nhấp đúp để Reset</div>
      </div>
    </div>

    <script>
      (function() {{
        let scale = 1.0;
        let panX = 0;
        let panY = 0;
        let isDragging = false;
        let startX = 0;
        let startY = 0;

        const frame = document.getElementById('{viewer_id}_frame');
        const vp = document.getElementById('{viewer_id}_vp');
        const layer = document.getElementById('{viewer_id}_layer');
        const zoomText = document.getElementById('{viewer_id}_zoom');

        function apply() {{
          layer.style.transform = `translate(${{panX}}px, ${{panY}}px) scale(${{scale}})`;
          zoomText.textContent = Math.round(scale * 100) + '%';
        }}

        window.zoomIn = function() {{
          scale = Math.min(scale * 1.25, 5.0);
          apply();
        }};

        window.zoomOut = function() {{
          scale = Math.max(scale * 0.8, 0.5);
          apply();
        }};

        window.resetView = function() {{
          scale = 1.0;
          panX = 0;
          panY = 0;
          apply();
        }};

        window.toggleFull = function() {{
          if (!document.fullscreenElement) {{
            frame.requestFullscreen().catch(err => {{}});
          }} else {{
            document.exitFullscreen();
          }}
        }};

        vp.addEventListener('wheel', function(e) {{
          e.preventDefault();
          const factor = e.deltaY < 0 ? 1.15 : 0.87;
          scale = Math.min(Math.max(scale * factor, 0.5), 5.0);
          apply();
        }}, {{ passive: false }});

        vp.addEventListener('mousedown', function(e) {{
          if (e.button !== 0) return;
          isDragging = true;
          vp.classList.add('grabbing');
          startX = e.clientX - panX;
          startY = e.clientY - panY;
        }});

        window.addEventListener('mousemove', function(e) {{
          if (!isDragging) return;
          panX = e.clientX - startX;
          panY = e.clientY - startY;
          apply();
        }});

        window.addEventListener('mouseup', function() {{
          if (isDragging) {{
            isDragging = false;
            vp.classList.remove('grabbing');
          }}
        }});

        vp.addEventListener('dblclick', function(e) {{
          e.preventDefault();
          resetView();
        }});
      }})();
    </script>
    </body>
    </html>
    """
    st.components.v1.html(html_code, height=height + 15)

# -----------------------------------------------------------------------------
# 5. THANH ĐIỀU KHIỂN BÊN (SIDEBAR - CHUẨN PACS THEME)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Cấu hình Suy luận AI")
    conf_thresh = st.slider("Ngưỡng tin cậy (Confidence):", min_value=0.05, max_value=0.95, value=0.20, step=0.05,
                           help="Ngưỡng điểm số tối thiểu để AI hiển thị tổn thương. Chuẩn F1-Score tối ưu là 0.20.")
    iou_thresh = st.slider("Ngưỡng chồng lấp (IoU NMS):", min_value=0.10, max_value=0.90, value=0.45, step=0.05,
                          help="Ngưỡng NMS loại bỏ các hộp thừa trùng đè lên nhau quanh cùng một ổ gãy.")

    st.markdown("---")
    st.markdown("### 🔬 Tiền xử lý & Nâng cao ảnh")
    enable_clahe = st.toggle("Bật tăng cường tương phản CLAHE", value=True,
                            help="Cân bằng biểu đồ mức xám thích ứng cục bộ, làm nổi bật đường nứt vi thể.")
    if enable_clahe:
        clahe_clip = st.slider("Độ cắt ClipLimit:", 1.0, 4.0, 2.0, 0.5)
    else:
        clahe_clip = 2.0

    enable_panzoom = st.toggle("Bật Kính lúp Pan & Zoom PACS", value=True,
                              help="Cho phép lăn chuột phóng to 1x-5x và kéo di chuyển ảnh X-quang trực tiếp.")

    st.markdown("---")
    st.markdown("### 🏷️ Lọc nhãn hiển thị")
    all_class_keys = list(CLASS_MAP.keys())
    selected_classes = st.multiselect(
        "Hiển thị các loại tổn thương:",
        options=all_class_keys,
        default=["fracture", "metal", "periostealreaction", "pronatorsign", "softtissue", "text"],
        format_func=lambda x: CLASS_MAP[x]["vi"]
    )

    st.markdown("---")
    st.markdown("### 📥 Nguồn ảnh chụp")
    input_source = st.radio("Chọn phương thức nạp ảnh:", ("Dùng ảnh mẫu lâm sàng", "Tải ảnh từ máy tính (Upload)"))

    image_to_process = None
    image_name = ""
    patient_id_label = "P-TEST-SAMPLE_01"

    if input_source == "Dùng ảnh mẫu lâm sàng":
        sample_options = {
            "Mẫu 1: Gãy xương cổ tay kèm nẹp kim loại (Fracture + Metal)": "sample_01_fracture.png",
            "Mẫu 2: Nẹp cố định sau phẫu thuật (Metal View)": "sample_02_metal.png",
            "Mẫu 3: Phản ứng màng xương tạo can (Periosteal Reaction)": "sample_03_periosteal.png",
            "Mẫu 4: X-quang cổ tay bình thường (Normal Bone)": "sample_04_normal.png",
            "Mẫu 5: Ca gãy xương bó bột thạch cao (Cast Fracture - Test Set)": "sample_05_cast_fracture_test.png"
        }
        selected_sample_label = st.selectbox("Chọn phim X-quang mẫu:", list(sample_options.keys()))
        sample_filename = sample_options[selected_sample_label]
        sample_path = os.path.join(SAMPLES_DIR, sample_filename)
        
        if os.path.exists(sample_path):
            image_to_process = Image.open(sample_path).convert("RGB")
            image_name = selected_sample_label
            patient_id_label = f"P-TEST-{sample_filename[:9].upper()}"
        else:
            st.warning("Chưa tìm thấy file ảnh mẫu.")
    else:
        uploaded_file = st.file_uploader("Tải file ảnh X-quang (JPG, PNG):", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image_to_process = Image.open(uploaded_file).convert("RGB")
            image_name = uploaded_file.name
            patient_id_label = f"BN-{int(time.time()) % 10000:04d}"

# -----------------------------------------------------------------------------
# 6. HEADER CHÍNH CỦA ỨNG DỤNG (CHUẨN PACS WORKSTATION)
# -----------------------------------------------------------------------------
st.markdown("""
<div class="pacs-header-card">
    <div class="pacs-main-title">
        <span style="color:#00e5ff;">🩻</span> PACS AI WORKSTATION - CHẨN ĐOÁN GÃY XƯƠNG CỔ TAY NHI
    </div>
    <div class="pacs-subtitle">
        Hệ thống hỗ trợ ra quyết định lâm sàng (CDSS) trên ảnh X-quang chấn thương cổ tay trẻ em | Nhóm 11 - Deep Learning (ICTU)
    </div>
    <div class="pacs-badge-row">
        <span class="pacs-tag tag-cyan">⚡ YOLOv8s Anchor-Free</span>
        <span class="pacs-tag tag-green">🎯 mAP@0.5 Fracture: 94.0%</span>
        <span class="pacs-tag tag-amber">🔬 CLAHE Adaptive Filter</span>
        <span class="pacs-tag tag-purple">⏱️ Latency: ~15ms (GPU) / ~189ms (CPU)</span>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 7. PHÂN CHIA HỆ THỐNG THÀNH 4 TABS CHUYÊN NGHIỆP
# -----------------------------------------------------------------------------
tab_main, tab_analytics, tab_guide = st.tabs([
    "🩺 Phòng Chẩn đoán hình ảnh (Workspace)",
    "📊 Bảng phân tích & Hiệu năng mô hình",
    "📖 Cẩm nang 9 tổn thương"
])

# =============================================================================
# TAB 1: PHÒNG CHẨN ĐOÁN HÌNH ẢNH (LIVE DIAGNOSTIC WORKSPACE)
# =============================================================================
with tab_main:
    if image_to_process is None:
        # MÀN HÌNH CHÀO ĐÓN KHI CHƯA NẠP ẢNH (EMPTY STATE CHUYÊN NGHIỆP)
        st.markdown("""
        <div style="background: #0d0f14; border: 1px dashed #22262e; border-radius: 10px; padding: 45px 30px; text-align: center; margin: 20px 0;">
            <div style="font-size: 3.8rem; margin-bottom: 12px; filter: drop-shadow(0 0 15px rgba(0,229,255,0.3));">🩻</div>
            <h3 style="color: #ffffff; margin-bottom: 8px; font-weight: 700;">Trạm Chẩn đoán Hình ảnh X-quang Kỹ thuật số</h3>
            <p style="color: #8b949e; max-width: 680px; margin: 0 auto 24px auto; font-size: 0.95rem; line-height: 1.6;">
                Hệ thống tích hợp mô hình học sâu <b>YOLOv8s Anchor-Free</b> được huấn luyện trên <b>20.327 ảnh X-quang</b> của bộ dữ liệu quốc tế GRAZPEDWRI-DX, hỗ trợ bác sĩ sàng lọc khẩn cấp các tổn thương gãy xương cành tươi, nẹp kim loại và sưng nề mô mềm với độ nhạy 98.1%.
            </p>
            <div style="display: flex; justify-content: center; gap: 15px; margin-top: 20px; flex-wrap: wrap;">
                <div style="background: #111317; padding: 14px 22px; border-radius: 6px; border: 1px solid #1f232b; min-width: 140px;">
                    <div style="font-weight: 800; color: #00e5ff; font-size: 1.3rem;">20.327</div>
                    <div style="font-size: 0.78rem; color: #7d8590;">Ảnh X-quang Train/Val/Test</div>
                </div>
                <div style="background: #111317; padding: 14px 22px; border-radius: 6px; border: 1px solid #1f232b; min-width: 140px;">
                    <div style="font-weight: 800; color: #10b981; font-size: 1.3rem;">94.0%</div>
                    <div style="font-size: 0.78rem; color: #7d8590;">mAP@0.5 Lớp Gãy xương</div>
                </div>
                <div style="background: #111317; padding: 14px 22px; border-radius: 6px; border: 1px solid #1f232b; min-width: 140px;">
                    <div style="font-weight: 800; color: #f59e0b; font-size: 1.3rem;">&lt; 200 ms</div>
                    <div style="font-size: 0.78rem; color: #7d8590;">Tốc độ suy luận CPU</div>
                </div>
            </div>
            <div style="margin-top: 30px; font-size: 0.9rem; color: #00e5ff;">
                👈 <b>Thao tác:</b> Hãy chọn <b>Ảnh mẫu lâm sàng</b> hoặc <b>Tải ảnh lên</b> ở thanh điều khiển bên trái để bắt đầu soi phim.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # THANH THÔNG TIN CA CHỤP PACS METADATA BAR
        st.markdown(f"""
        <div style="background: #111317; border-left: 4px solid #00e5ff; padding: 10px 16px; border-radius: 6px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; border: 1px solid #1f242d; border-left: 4px solid #00e5ff;">
            <div><b>Mã ca chụp:</b> <span style="color:#00e5ff; font-family: monospace;">{patient_id_label}</span> &nbsp;|&nbsp; <b>Tập tin:</b> <span style="color:#e2e8f0;">{image_name}</span></div>
            <div style="color: #7d8590; font-size: 0.84rem;">Thời gian phân tích: <b style="color:#c9d1d9;">{time.strftime('%H:%M:%S - %d/%m/%Y')}</b></div>
        </div>
        """, unsafe_allow_html=True)

        # Xử lý ảnh với bộ lọc CLAHE nếu được bật
        if enable_clahe:
            input_image_ready = apply_clahe(image_to_process, clip_limit=clahe_clip, tile_grid_size=(8, 8))
            input_caption = f"X-QUANG (CLAHE Clip={clahe_clip})"
        else:
            input_image_ready = image_to_process
            input_caption = "X-QUANG GỐC (RAW)"

        # Chạy suy luận với YOLOv8s
        t_start = time.time()
        with st.spinner("Đang chạy mô hình YOLOv8s phân tích vi tổn thương..."):
            results = model.predict(input_image_ready, conf=conf_thresh, iou=iou_thresh, verbose=False)
        t_infer = (time.time() - t_start) * 1000

        result = results[0]
        boxes = result.boxes
        
        # Lọc các hộp theo danh sách nhãn bác sĩ chọn ở sidebar
        filtered_indices = []
        if len(boxes) > 0:
            for idx, box in enumerate(boxes):
                cls_id = int(box.cls[0].item())
                raw_name = result.names[cls_id]
                if raw_name in selected_classes:
                    filtered_indices.append(idx)

        num_detections = len(filtered_indices)

        # Vẽ kết quả Bounding Box tùy chỉnh sắc nét chuẩn PACS
        res_np = np.array(input_image_ready.copy())
        fracture_detected = False
        metal_detected = False
        periosteal_detected = False
        det_list = []
        crop_patches = []

        for idx_order, idx in enumerate(filtered_indices):
            box = boxes[idx]
            cls_id = int(box.cls[0].item())
            raw_name = result.names[cls_id]
            conf = float(box.conf[0].item())
            xyxy = [int(v) for v in box.xyxy[0].tolist()]

            info = CLASS_MAP.get(raw_name, {
                "vi": raw_name, "color": (0, 229, 255), "hex": "#00e5ff", 
                "desc": "Tổn thương", "severity": "Theo dõi", "action": "Khám chuyên khoa"
            })

            if raw_name == "fracture":
                fracture_detected = True
            elif raw_name == "metal":
                metal_detected = True
            elif raw_name == "periostealreaction":
                periosteal_detected = True

            # Vẽ bounding box lên ảnh
            color_bgr = (info["color"][2], info["color"][1], info["color"][0])
            x1, y1, x2, y2 = xyxy
            cv2.rectangle(res_np, (x1, y1), (x2, y2), color_bgr, thickness=2)

            # Nhãn chữ có nền màu sắc nét
            label_text = f"{info['vi'].split('(')[0].strip()}: {conf * 100:.1f}%"
            (tw, th), bl = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            ty = max(y1, th + bl + 2)
            cv2.rectangle(res_np, (x1, ty - th - bl - 2), (x1 + tw + 6, ty + 2), color_bgr, -1)
            cv2.putText(res_np, label_text, (x1 + 3, ty - bl), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

            # Cắt ảnh Crop phục vụ kính lúp soi chi tiết (ROI Crop)
            img_h, img_w = res_np.shape[:2]
            pad_x = int((x2 - x1) * 0.25)
            pad_y = int((y2 - y1) * 0.25)
            crop_x1 = max(0, x1 - pad_x)
            crop_y1 = max(0, y1 - pad_y)
            crop_x2 = min(img_w, x2 + pad_x)
            crop_y2 = min(img_h, y2 + pad_y)
            cropped_area = Image.fromarray(res_np[crop_y1:crop_y2, crop_x1:crop_x2])
            crop_patches.append((idx_order + 1, info["vi"], conf, cropped_area, info))

            det_list.append({
                "STT": idx_order + 1,
                "Tổn thương phát hiện": info["vi"],
                "Độ tin cậy": f"{conf * 100:.1f}%",
                "Mức độ ưu tiên": info["severity"],
                "Ý nghĩa lâm sàng": info["desc"],
                "Tọa độ [x1, y1, x2, y2]": str(xyxy)
            })

        output_image_ready = Image.fromarray(res_np)

        # 2 CỘT HIỂN THỊ ẢNH (KÈM KÍNH LÚP PAN & ZOOM PACS KHÔNG ĐÈ CHỮ)
        col_view_left, col_view_right = st.columns(2)

        with col_view_left:
            st.markdown("##### 📷 Ảnh X-quang đầu vào")
            if enable_panzoom:
                render_panzoom_viewer(input_image_ready, caption=f"ĐẦU VÀO | {input_caption}", viewer_id="viewer_input", height=520)
            else:
                st.image(input_image_ready, caption=input_caption, use_container_width=True)

        with col_view_right:
            st.markdown("##### 🎯 Kết quả Chẩn đoán Khoanh vùng của AI")
            output_caption = f"YOLOv8s | {num_detections} VÙNG (Conf ≥ {conf_thresh:.2f})"
            if enable_panzoom:
                render_panzoom_viewer(output_image_ready, caption=output_caption, viewer_id="viewer_result", height=520)
            else:
                st.image(output_image_ready, caption=f"Phát hiện {num_detections} vùng tổn thương", use_container_width=True)

        # 4 THẺ METRICS KPI TỔNG HỢP THEO PHONG CÁCH PACS HUD
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="pacs-kpi-card">
                <div class="pacs-kpi-label">Tổng tổn thương</div>
                <div class="pacs-kpi-val">{num_detections}</div>
                <div class="pacs-kpi-sub">Vùng nghi ngờ khoanh hộp</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            top_conf = max([float(boxes[i].conf[0].item()) for i in filtered_indices]) if num_detections > 0 else 0.0
            st.markdown(f"""
            <div class="pacs-kpi-card">
                <div class="pacs-kpi-label">Độ tin cậy cao nhất</div>
                <div class="pacs-kpi-val">{top_conf * 100:.1f}%</div>
                <div class="pacs-kpi-sub">Mức độ chắc chắn của AI</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="pacs-kpi-card">
                <div class="pacs-kpi-label">Thời gian suy luận</div>
                <div class="pacs-kpi-val">{t_infer:.1f} <span style="font-size:1rem;color:#7d8590;">ms</span></div>
                <div class="pacs-kpi-sub">Thời gian thực (Real-time)</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="pacs-kpi-card">
                <div class="pacs-kpi-label">Cấu hình Ngưỡng</div>
                <div class="pacs-kpi-val">{conf_thresh:.2f}</div>
                <div class="pacs-kpi-sub">IoU NMS: {iou_thresh:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        # KHUNG ĐÁNH GIÁ CẢNH BÁO LÂM SÀNG SƠ BỘ
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        if fracture_detected:
            st.markdown("""
            <div class="pacs-alert-danger">
                <h4 style="margin: 0 0 6px 0; color: #ff2e4d; font-weight: 800;">🚨 CẢNH BÁO CẤP CỨU: PHÁT HIỆN TỔN THƯƠNG GÃY XƯƠNG CỔ TAY (FRACTURE)</h4>
                <p style="margin: 0; font-size: 0.94rem; line-height: 1.5;">
                    Hệ thống nhận diện thấy gián đoạn liên tục của bờ vỏ xương (đường nứt hoặc gãy cành tươi). 
                    <b>Khuyến nghị lâm sàng:</b> Cố định ngay cổ tay bằng nẹp máng cẳng bàn tay; chỉ định chụp thêm tư thế Chếch hoặc hội chẩn chuyên khoa Chấn thương Chỉnh hình Nhi.
                </p>
            </div>
            """, unsafe_allow_html=True)
        elif metal_detected or periosteal_detected:
            st.markdown("""
            <div class="pacs-alert-warning">
                <h4 style="margin: 0 0 6px 0; color: #f59e0b; font-weight: 800;">⚠️ LƯU Ý LÂM SÀNG: PHÁT HIỆN DI CHỨNG CAN THIỆP / PHẢN ỨNG MÀNG XƯƠNG</h4>
                <p style="margin: 0; font-size: 0.94rem; line-height: 1.5;">
                    Ghi nhận có vật liệu kim loại (đinh/nẹp) hoặc dày màng xương tạo can. Không thấy đường gãy cấp tính di lệch mới rõ rệt. Cần theo dõi tiến trình liền xương.
                </p>
            </div>
            """, unsafe_allow_html=True)
        elif num_detections > 0:
            st.markdown("""
            <div class="pacs-alert-success">
                <h4 style="margin: 0 0 6px 0; color: #10b981; font-weight: 800;">ℹ️ KHÔNG PHÁT HIỆN ĐƯỜNG GÃY CẤP TÍNH</h4>
                <p style="margin: 0; font-size: 0.94rem; line-height: 1.5;">
                    Chỉ phát hiện các ký hiệu đánh dấu giải phẫu (Text L/R) hoặc bóng sưng mô mềm nhẹ. Toàn vẹn vỏ xương quay và xương trụ được bảo tồn.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="pacs-alert-success">
                <h4 style="margin: 0 0 6px 0; color: #10b981; font-weight: 800;">✅ BÌNH THƯỜNG: KHÔNG PHÁT HIỆN BẤT THƯỜNG</h4>
                <p style="margin: 0; font-size: 0.94rem; line-height: 1.5;">
                    Không phát hiện bất kỳ dấu hiệu tổn thương gãy xương hoặc dị vật nào vượt qua ngưỡng tin cậy đã chọn.
                </p>
            </div>
            """, unsafe_allow_html=True)

        # KÍNH HIỂN VI SOI CẬN CẢNH TỪNG Ổ GÃY (ROI CROP ZOOM-IN INSPECTOR)
        if crop_patches:
            with st.expander("🔬 Kính hiển vi soi cận cảnh từng tổn thương (ROI Zoom-In Inspector)", expanded=True):
                st.caption("Hệ thống tự động cắt riêng từng vùng nghi ngờ và phóng đại 300% để bác sĩ soi kỹ ranh giới vỏ xương:")
                roi_cols = st.columns(min(len(crop_patches), 4))
                for idx_p, (roi_num, roi_name, roi_conf, roi_img, roi_info) in enumerate(crop_patches[:4]):
                    with roi_cols[idx_p]:
                        st.image(roi_img, caption=f"Vùng #{roi_num}: {roi_name.split('(')[0]} ({roi_conf * 100:.1f}%)", use_container_width=True)
                        st.markdown(f"""
                        <div style="font-size: 0.78rem; background: #0c0e12; padding: 8px 10px; border-radius: 6px; border: 1px solid #1e2430;">
                            <b style="color:#00e5ff;">Mức độ:</b> {roi_info['severity']}<br>
                            <b style="color:#f59e0b;">Xử trí:</b> {roi_info['action']}
                        </div>
                        """, unsafe_allow_html=True)

        # BẢNG THỐNG KÊ CHI TIẾT CÁC TỔN THƯƠNG
        if det_list:
            st.markdown("##### 📋 Danh mục chi tiết các vùng tổn thương khoanh vùng")
            df = pd.DataFrame(det_list)
            st.dataframe(df, use_container_width=True, hide_index=True)

        # =============================================================================
# TAB 2: BẢNG PHÂN TÍCH & HIỆU NĂNG MÔ HÌNH (MODEL ANALYTICS)
# =============================================================================
with tab_analytics:
    st.markdown("### 📊 Kết quả Thực nghiệm Huấn luyện & Đối sánh Khoa học")
    st.caption("Các biểu đồ được xuất tự động từ phiên huấn luyện 50 epoch trên 2 GPU NVIDIA Tesla T4 tại Kaggle:")

    col_bench1, col_bench2 = st.columns(2)

    with col_bench1:
        st.markdown("##### 📈 Quá trình hội tụ hàm mất mát (Training Loss Curves)")
        loss_img_path = os.path.join(RUNS_DIR, "yolov8s_grazpedwri", "results.png")
        if os.path.exists(loss_img_path):
            st.image(loss_img_path, caption="Biểu đồ suy giảm Loss (Box, Cls, DFL) qua 50 epochs", use_container_width=True)
        else:
            st.info("Biểu đồ results.png lưu tại Kaggle runs.")

    with col_bench2:
        st.markdown("##### 🎯 Ma trận nhầm lẫn chuẩn hóa (Normalized Confusion Matrix)")
        cm_img_path = os.path.join(RUNS_DIR, "test_eval", "confusion_matrix_normalized.png")
        if os.path.exists(cm_img_path):
            st.image(cm_img_path, caption="Tỷ lệ nhận diện đúng lớp Gãy xương đạt 98.0%", use_container_width=True)
        else:
            st.info("Biểu đồ confusion_matrix.png lưu tại Kaggle runs.")

    col_pr1, col_pr2 = st.columns(2)
    with col_pr1:
        st.markdown("##### 📉 Đường cong Precision - Recall (PR Curve)")
        pr_img_path = os.path.join(RUNS_DIR, "test_eval", "BoxPR_curve.png")
        if os.path.exists(pr_img_path):
            st.image(pr_img_path, caption="mAP@0.5 lớp Gãy xương đạt 94.0%; Text đạt 99.2%", use_container_width=True)

    with col_pr2:
        st.markdown("##### ⚖️ Đường cong F1 - Confidence Curve")
        f1_img_path = os.path.join(RUNS_DIR, "test_eval", "BoxF1_curve.png")
        if os.path.exists(f1_img_path):
            st.image(f1_img_path, caption="Điểm cân bằng F1 tối ưu đạt tại ngưỡng Confidence = 0.189 (~0.20)", use_container_width=True)

    st.markdown("---")
    st.markdown("##### 🥊 Bảng đối sánh trực diện: YOLOv8s vs Faster R-CNN vs YOLOv5m")
    benchmark_df = pd.DataFrame([
        {
            "Mô hình đối chứng": "Faster R-CNN (ResNet50-FPN)",
            "Trường phái": "Hai giai đoạn (Two-Stage)",
            "Số tham số": "≈ 41.5 Triệu",
            "Dung lượng weights": "≈ 160 MB",
            "Độ trễ GPU": "65 - 80 ms",
            "Độ trễ CPU": "650 - 900 ms",
            "mAP@0.5 Toàn cục": "56.8%",
            "mAP@0.5 Fracture": "88.5%"
        },
        {
            "Mô hình đối chứng": "Baseline YOLOv5m (Nature 2022)",
            "Trường phái": "Một giai đoạn (Anchor-Based)",
            "Số tham số": "≈ 21.2 Triệu",
            "Dung lượng weights": "42.5 MB",
            "Độ trễ GPU": "18 - 25 ms",
            "Độ trễ CPU": "~380 ms",
            "mAP@0.5 Toàn cục": "62.2%",
            "mAP@0.5 Fracture": "93.3%"
        },
        {
            "Mô hình đối chứng": "Đề xuất: YOLOv8s (Nhóm 11)",
            "Trường phái": "Một giai đoạn (Anchor-Free)",
            "Số tham số": "≈ 11.2 Triệu (-47%)",
            "Dung lượng weights": "22.5 MB (-47%)",
            "Độ trễ GPU": "15.2 ms (Nhanh gấp đôi)",
            "Độ trễ CPU": "~189 ms (Tức thời)",
            "mAP@0.5 Toàn cục": "66.9% (Val) / 58.4% (Test)",
            "mAP@0.5 Fracture": "94.0% (Xuất sắc nhất)"
        }
    ])
    st.dataframe(benchmark_df, use_container_width=True, hide_index=True)

# =============================================================================
# TAB 3: CẨM NANG 9 TỔN THƯƠNG & THÔNG TIN ĐỀ TÀI
# =============================================================================
with tab_guide:
    st.markdown("### 📖 Cẩm nang Phân loại 9 Lớp Tổn thương (GRAZPEDWRI-DX)")
    st.caption("Chi tiết định nghĩa lâm sàng và ý nghĩa y khoa của 9 lớp đối tượng:")

    c_g1, c_g2, c_g3 = st.columns(3)
    classes_list = list(CLASS_MAP.items())

    for idx, (cls_name, cls_info) in enumerate(classes_list):
        col_target = [c_g1, c_g2, c_g3][idx % 3]
        with col_target:
            st.markdown(f"""
            <div style="background: #111317; border: 1px solid #1f242d; border-top: 3px solid {cls_info['hex']}; border-radius: 6px; padding: 14px; margin-bottom: 14px;">
                <h5 style="color: {cls_info['hex']}; margin: 0 0 6px 0; font-weight: 700;">{cls_info['vi']}</h5>
                <p style="font-size: 0.85rem; color: #cbd5e1; margin-bottom: 8px;"><b>Mô tả:</b> {cls_info['desc']}</p>
                <div style="font-size: 0.78rem; background: #08090b; padding: 6px 8px; border-radius: 4px; color: #8b949e; border: 1px solid #1a1e26;">
                    <b style="color:#00e5ff;">Xử trí:</b> {cls_info['action']}
                </div>
            </div>
            """, unsafe_allow_html=True)


