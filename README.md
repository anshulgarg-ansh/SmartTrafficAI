# 🚦 SmartTraffic AI

**AI-Powered Intelligent Traffic Monitoring and Violation Detection System**

SmartTraffic AI is an AI-powered traffic monitoring platform that uses Computer Vision, Deep Learning, and OCR to detect vehicles, identify potential traffic violations, and recognize vehicle registration numbers from traffic videos.

## ✨ Features

- 🚗 **Vehicle Detection:** Detects cars, motorcycles, buses, trucks, and persons using YOLOv8.
- 🪖 **Helmet Detection:** Identifies riders wearing or not wearing helmets.
- 🔢 **Number Plate Detection:** Detects vehicle registration plates.
- 🔍 **Automatic Number Plate Recognition (ANPR):** Extracts registration numbers using EasyOCR.
- 🧠 **OCR Voting & Deduplication:** Combines repeated OCR readings to improve candidate ranking.
- 🚨 **Violation Evidence:** Saves images of potential no-helmet violations.
- 📊 **Traffic Dashboard:** Displays vehicle statistics, violations, pending challans, and fines.
- 📋 **E-Challan Management:** Manages prototype challan records.
- 🔎 **No Helmet Event Review:** Displays related evidence and possible number plate candidates for manual verification.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Backend programming |
| Flask | Web application framework |
| YOLOv8 | Object detection |
| OpenCV | Video and image processing |
| EasyOCR | Number plate text recognition |
| SQLite | Database |
| HTML, CSS, JavaScript | Frontend |

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/anshulgarg-ansh/SmartTrafficAI.git
cd SmartTrafficAI
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

**Windows PowerShell:**

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use:

```cmd
venv\Scripts\activate.bat
```

### 4. Install dependencies

```bash
pip install flask ultralytics opencv-python easyocr
```

Install any additional dependencies listed in `requirements.txt`, if present.

### 5. Run the application

```bash
python app.py
```

Open your browser:

`http://127.0.0.1:5000`

## 📁 Project Modules

- `app.py` – Flask application and routing.
- `ai_engine.py` – AI video processing and detection.
- `database.py` – Database operations.
- `models/` – Trained AI model weights.
- `static/` – CSS, JavaScript, uploads and evidence.
- `templates/` – HTML pages.

## 🔬 How It Works

1. User uploads a traffic video.
2. OpenCV extracts video frames.
3. YOLO models detect vehicles and helmet-related classes.
4. Number plate detection identifies plate regions.
5. EasyOCR extracts registration text.
6. OCR voting consolidates repeated readings.
7. Potential violations and evidence are displayed on the dashboard.
8. Human verification is required before official action.

## ⚠️ Important Disclaimer

This project is an academic prototype. AI predictions and OCR results may contain errors. Vehicle-to-rider associations are not guaranteed to be correct. All potential violations and registration numbers must be manually verified before issuing an official challan.

## 🚀 Future Enhancements

- Live CCTV camera integration.
- Real-time traffic monitoring.
- Improved vehicle tracking.
- More accurate number plate recognition.
- Automated violation review workflow.
- Improved rider-to-vehicle association.

---

**Developed as an AI-based Smart Traffic Monitoring Project.**
