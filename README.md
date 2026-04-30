# 🚶 Kişi Sayma ve Takip Sistemi (KSTS)

> **Real-time people counting and tracking system using YOLOv3 and centroid tracking.**
> Developed during internship at **Forte Bilgi İletişim Teknolojileri A.Ş.** · July 2023

---

## 📖 Overview

This project implements a **people counting and tracking system** that works on video streams or camera feeds. A configurable reference line is drawn on the frame; when a person's centroid crosses this line, they are counted as entering or exiting. The system uses **YOLOv3** for person detection and a custom **Centroid Tracker** for persistent ID assignment across frames.

**Use cases:** crowd control, capacity management, visitor analytics, marketing insights, security monitoring.

---

## 🏗️ How It Works

```
Video / Camera Input
        │
        ▼
YOLOv3 Person Detection
(bounding boxes → person class only)
        │
        ▼
Centroid Calculation
X_cen = (xmin + xmax) // 2
Y_cen = (ymin + ymax) // 2
        │
        ▼
Centroid Tracker
(Euclidean distance → assign unique IDs)
        │
        ▼
Reference Line Crossing Detection
(track centroid direction: UP ↑ or DOWN ↓)
        │
        ▼
Counter Update + Annotated Output
```

---

## 📁 Project Structure

```
ksts/
│
├── utils/
│   ├── centroidtracker.py      # Centroid-based multi-object tracker
│   └── object_trackable.py     # TrackableObject class (ID + centroid history)
│
├── weights/                    # YOLOv3 weights (not included, see Setup)
│   ├── yolov3.weights
│   ├── yolov3.cfg
│   └── coco.names
│
├── output/                     # Output videos saved here
│
├── counting_people.py          # Main detection + counting script
└── README.md
```

---

## ⚙️ Setup

### 1. Clone & install dependencies

```bash
git clone https://github.com/busesaricayir/ksts.git
cd ksts
pip install -r requirements.txt
```

### 2. Download YOLOv3 weights

```bash
wget https://pjreddie.com/media/files/yolov3.weights -P weights/
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg -P weights/
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names -P weights/
```

---

## 🚀 Usage

```bash
# Run on a video file
python counting_people.py --input path/to/video.mp4 --output output/result.mp4

# Run on webcam (real-time)
python counting_people.py --input 0 --output output/result.mp4
```

---

## 📦 Requirements

```
numpy
opencv-python
scipy
```

---

## 📄 References

- [YOLOv3 — Ultralytics](https://github.com/ultralytics/yolov3)
- Ren, P. et al. (2020). *A novel squeeze YOLO-based real-time people counting approach.* IJBIC, 16(2).

---

<div align="center"></div>
