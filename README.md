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
- outputs
- requirements.txt

---

## 🧠 Part 1: Vehicle Detection (Deep Learning)

### 🔹 Pipeline

1. **Data Ingestion**
   - Reads annotations CSV
   - Splits into train/validation sets  
   👉 `data_ingestion.py`

2. **Model Architecture**
   - Faster R-CNN with MobileNet backbone
   - Pretrained model fine-tuned for vehicle detection  
   👉 `model.py`

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
   👉 `inference.py` 

---

### ⚙️ Model Details

- Architecture: **Faster R-CNN**
- Backbone: **MobileNet V3**
- Task: **Object Detection (Bounding Boxes + Labels)**

---

## 📊 Part 2: Tesla Autopilot Data Analysis

### 🔹 Tasks Performed

- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- Trend analysis of:
  - Accidents over time
  - Death distribution
  - Driver vs pedestrian impact
  - Autopilot-related fatalities

### 📁 Notebooks

- `Data Cleaning.ipynb`
- `EDA Analysis.ipynb`

---

## 📦 Dataset

### Part 1
- Image dataset with bounding box annotations
- Format:
     - image_id
     - label
     - xmin
     - ymin
     - xmax
     - ymax

### Part 2
- Tesla accident dataset including:
     - Date, location, deaths
     - Autopilot involvement
     - Vehicle & collision details

---

## 🚀 How to Run

### 1️⃣ Clone Repository
- git clone https://github.com/PrakharSri18-data/Autonomous-Driving-Deep-Learning.git
- cd Autonomous-Driving-Deep-Learning

### 2️⃣ Setup Environment
- python -m venv venv
- venv\Scripts\activate

### 3️⃣ Install Dependencies
- pip install -r requirements.txt

### 4️⃣ Data Ingestion
- python src/part1DeepLearning/data_ingestion.py

### 5️⃣ Train Model
- python src/part1DeepLearning/train.py

### 6️⃣ Evaluate Model
- python src/part1DeepLearning/evaluation.py

### 7️⃣ Run Inference
- python src/part1DeepLearning/inference.py

---

## 📈 Results
- Successfully detects vehicles with bounding boxes
- Model evaluated using **Precision & Recall**
- Visual outputs saved in `outputs` folder

---

## ⚠️ Challenges
- Dataset size & preprocessing time
- Bounding box accuracy
- Class imbalance
- Real-time performance constraints

---

## 🧠 Key Learnings
- End-to-end ML pipeline design
- Object detection using PyTorch
- Evaluation using IoU metrics
- Real-world data analysis (Tesla autopilot)
- Combining Deep Learning + Data Science in one project

---

## 👤 Author
- Prakhar Srivastava
- Aspiring Data Scientist & Business Analyst | Machine Learning, Deep Learning & Generative AI Enthusiast
