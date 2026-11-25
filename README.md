# 🩺 AI-Assisted Pediatric Wrist Fracture Detection (GRAZPEDWRI-DX)
### Hệ thống Hỗ trợ Chẩn đoán Phát hiện Gãy xương Cổ tay Trẻ em trên ảnh X-quang dựa trên Kiến trúc YOLOv8s kết hợp PACS Workstation

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Ultralytics YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00ffff.svg)](https://docs.ultralytics.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-PACS%20Demo-ff4b4b.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Giới thiệu Đề tài

Gãy xương cổ tay là chấn thương phổ biến nhất ở trẻ em (chiếm hơn 25% tổng số ca chấn thương nhi khoa). Do đặc điểm giải phẫu xương nhi có nhiều đĩa sụn tiếp hợp (growth plate) và các đường gãy tế vi (gãy cành tươi - greenstick, nứt màng xương), việc chẩn đoán trên phim X-quang thông thường đòi hỏi nhiều kinh nghiệm và dễ bị bỏ sót trong điều kiện trực cấp cứu quá tải.

Dự án này nghiên cứu và phát triển hệ thống trí tuệ nhân tạo hỗ trợ bác sĩ chẩn đoán phát hiện gãy xương cổ tay trẻ em trên bộ dữ liệu lớn **GRAZPEDWRI-DX** ($20.327$ ảnh X-quang chuẩn y khoa). Mô hình đề xuất **YOLOv8s** (Anchor-free, 11,2M tham số) đạt hiệu năng chẩn đoán vượt trội, độ trễ thấp và được tích hợp trực tiếp vào giao diện **Web Demo PACS Dark Mode** đạt chuẩn phòng đọc phim lâm sàng.

---

## 📊 Kết quả Thực nghiệm & Đối sánh Mô hình

Thực nghiệm được triển khai phân tán trên cụm $2 \times$ **GPU NVIDIA Tesla T4** (16GB VRAM) với $50$ epochs huấn luyện, phân tách dữ liệu cố định theo mã bệnh nhân (*patient-level split* - $0\%$ rò rỉ):

| Mô hình | Số tham số | FLOPs | Dung lượng | Precision | Recall | mAP@0.5 (Val) | mAP@0.5:0.95 | mAP@0.5 (Test) | Tốc độ CPU |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Faster R-CNN (ResNet-50)** | 41,5 M | 134,0 G | 160 MB | 58,30% | 51,20% | 52,40% | 31,10% | 48,60% | ~680 ms |
| **Baseline YOLOv5m** | 21,2 M | 48,0 G | 42,5 MB | 71,00% | 51,70% | 62,20% | 38,90% | 55,10% | ~310 ms |
| **YOLOv8n (Bản Nano)** | **3,2 M** | **8,7 G** | **6,5 MB** | 73,50% | 59,99% | 64,78% | 39,87% | 58,90% | **~72 ms** |
| **YOLOv8s (Đề xuất chính)** | **11,2 M** | **28,6 G** | **22,5 MB** | **80,45%** | **61,45%** | **66,90%** | **41,72%** | **58,40%** | **~189 ms** |

### 🎯 Độ chính xác theo từng lớp tổn thương trên tập Test 1.000 ảnh (YOLOv8s):
* **fracture (Gãy xương cốt lõi):** **`94,0%`** (mAP@0.5) - Độ nhạy chẩn đoán lâm sàng xuất sắc.
* **metal (Nẹp / Đinh kim loại):** **`96,5%`** (mAP@0.5).
* **text (Ký hiệu tư thế chụp):** **`99,2%`** (mAP@0.5).
* **pronatorsign (Dấu hiệu cơ sấp):** **`80,0%`** (mAP@0.5).
* **periostealreaction (Phản ứng màng xương):** **`66,7%`** (mAP@0.5).

---

## 🖥️ Tính năng Nổi bật của Trạm PACS Web Demo (`app.py`)

1. **Giao diện Dark Mode chuẩn DICOM / PACS:** Tối ưu hóa thị giác theo tiêu chuẩn phòng đọc phim y tế tại các bệnh viện lớn.
2. **Thuật toán tăng tương phản CLAHE cục bộ:** Cho phép điều chỉnh ngưỡng cắt `ClipLimit` ($1.0 - 4.0$) để làm nổi rõ vân xương và vết nứt tế vi mà không gây cháy sáng hay sinh nhiễu tia X.
3. **Kính lúp Pan & Zoom tương tác:** Hỗ trợ bác sĩ lăn chuột zoom từ $1\times$ đến $5\times$ và rê chuột (*Pan*) soi cận cảnh ổ gãy.
4. **Bộ lọc suy luận AI động:** Thanh trượt tùy chỉnh Confidence Threshold ($0.05 - 0.95$) và IoU NMS Threshold ($0.10 - 0.90$) tức thời.

---

## 🚀 Hướng dẫn Cài đặt & Chạy Thử nghiệm (Quick Start)

### 1. Yêu cầu Hệ thống
* Python 3.9 trở lên
* Khuyến nghị GPU NVIDIA (nếu có) hoặc CPU thông thường (suy luận chỉ mất ~189 ms/ảnh).

### 2. Clone mã nguồn & Cài đặt môi trường
```bash
git clone https://github.com/TDVQLong/project_deep_learning.git
cd project_deep_learning

# Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt
```

### 3. Khởi động Ứng dụng Web Demo
* **Cách 1 (Khuyên dùng trên Windows):** Nhấp đúp chuột vào file `run_app.bat`.
* **Cách 2 (Dùng lệnh terminal):**
```bash
streamlit run app.py
```
Ứng dụng sẽ tự động mở tại địa chỉ: `http://localhost:8501`.

---

## 🗂️ Cấu trúc Thư mục

```text
├── .streamlit/                # Theme cấu hình dark mode PACS
├── configs/                   # File cấu hình dữ liệu grazpedwri_v8.yaml
├── notebooks/                 # Jupyter Notebooks huấn luyện trên Kaggle (YOLOv8s, YOLOv8n, Baseline)
├── weights/                   # Trọng số tiền huấn luyện (best.pt, yolov8s_best.pt, yolov8n_best.pt)
├── sample_images/             # Ảnh X-quang mẫu chọn lọc phục vụ test thử
├── results/                   # Ma trận nhầm lẫn, PR Curve và log results.csv của YOLOv8s & YOLOv8n
├── app.py                     # Giao diện chẩn đoán Web Demo PACS (Streamlit)
├── run_app.bat                # Khởi động ứng dụng 1-click trên Windows
├── requirements.txt           # Danh mục thư viện Python
├── .gitignore                 # Loại trừ file rác / dataset nặng
└── README.md                  # Hướng dẫn chi tiết dự án
```

---

## 👥 Thành viên Thực hiện (Nhóm 11)

* **Giảng viên hướng dẫn:** TS. Ngô Hữu Huy
* **Trường Đại học Công nghệ Thông tin và Truyền thông – Đại học Thái Nguyên (ICTU)**
* **Sinh viên thực hiện:**
  1. **Trần Đặng Vương Quốc Long** – *Trưởng nhóm*
  2. **Nguyễn Văn Minh** – *Thành viên 1*
  3. **Nguyễn Văn Chỉnh** – *Thành viên 2*

---

## 📄 Tài liệu Tham khảo Chính

1. **GRAZPEDWRI-DX Dataset:** E. Nagy, et al. *"A pediatric wrist trauma X-ray dataset for machine learning"*, Scientific Data, Nature, 2022.
2. **YOLOv8 Architecture:** Glenn Jocher, et al. *"Ultralytics YOLOv8"*, 2023.
3. **Faster R-CNN:** S. Ren, K. He, R. Girshick, J. Sun. *"Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks"*, IEEE TPAMI, 2015.
