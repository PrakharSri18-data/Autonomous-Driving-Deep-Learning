# 🚗 Autonomous Driving using Deep Learning & Data Analysis

An end-to-end AI/ML project that combines **Deep Learning (Object Detection)** and **Data Science (EDA)** to simulate key components of autonomous driving systems.

This project is divided into two major parts:
- **Part 1 → Vehicle Detection using Deep Learning**
- **Part 2 → Tesla Autopilot Incident Analysis**

---

## 📌 Problem Statement

This project is based on a real-world autonomous driving scenario where:

- Vehicles must be **detected and localized in images**
- Autopilot systems must be analyzed for **road safety impact**

For the complete problem statement open the **Problem Statement.pdf** inside the **Datasets & Problem Statement** folder.

---

## 🎯 Project Objectives

### 🔹 Part 1: Object Detection
- Detect vehicles in images
- Localize them using bounding boxes
- Train a deep learning model for detection

### 🔹 Part 2: Data Analysis
- Analyze Tesla autopilot accident data
- Perform EDA on fatalities, locations, and trends
- Understand safety implications

---

## 🏗️ Project Structure
- **Datasets & Problem Statement**
     - Part 1
          - labels.csv
     - Part 2
          - Tesla-Deaths.csv
     - Problem Statement.pdf
- **Notebook**
     - reports/figures
          - Data Cleaning.ipynb
          - EDA Analysis.ipynb
- **src**
     - part1DeepLearning
          - __init__.py
          - data ingestion.py
          - evaluation.py
          - inference.py
          - model.py
          - train.py
     - __init__.py
- .gitignore
- LICENSE
- README.md
- output.jpg
- requirements.txt

---

## 🧠 Part 1: Vehicle Detection (Deep Learning)

### 🔹 Pipeline

1. **Data Ingestion**
   - Reads annotations CSV
   - Splits into train/validation sets  
   👉 Implemented in `data_ingestion.py`

2. **Model Architecture**
   - Faster R-CNN with MobileNet backbone
   - Pretrained model fine-tuned for vehicle detection  
   👉 Defined in `model.py`

3. **Training**
   - Custom PyTorch dataset
   - Bounding box + label training
   - GPU support  
   👉 `train.py` 

4. **Evaluation**
   - IoU-based matching
   - Precision & Recall calculation  
   👉 `evaluation.py`

5. **Inference**
   - Predict bounding boxes on new images
   - Draw detections and save outputs  
   👉 `inference.py` :contentReference[oaicite:5]{index=5}

  
