# 🌊 Ocean Noise Pollution Analyzer (Enhanced)

A Machine Learning system to analyze and classify ocean acoustic signals and detect noise pollution patterns using supervised learning models.

---
## 📌 Problem Statement

Ocean environments are affected by increasing noise pollution from ships, submarines, and industrial activities. This project aims to classify underwater acoustic signals and detect different noise patterns using machine learning techniques.

## 🚀 Features
- Audio data preprocessing and cleaning  
- Ocean noise pattern analysis  
- Feature extraction using MFCC and Spectrograms  
- Machine Learning model evaluation  
- Data visualization using Matplotlib & Seaborn  
- Performance tracking using leaderboard system  

---

## 📂 Project Structure

data/ → Dataset  
src/ → Source code (model + leaderboard)  
outputs/ → Graphs & results  
leaderboard.csv → Model performance tracking  
test_model.py → Model evaluation  
train_model.py → Model training  
---

---

## 📊 Dataset Source
- Hydrophone audio recordings (marine sound dataset)  
- Public ocean acoustic datasets (NOAA / Kaggle / custom collected samples)  

--- 
## ⚙️ Workflow

Hydrophone Audio Data  
→ Data Preprocessing  
→ Train-Test Split  
→ Feature Extraction  
→ Model Training (Logistic Regression, Random Forest, Decision Tree)  
→ Prediction  
→ Evaluation (Accuracy, Precision, Recall, F1 Score)  
→ Baseline Comparison  
→ Visualization (Baseline vs Model)
---

## 🧠 Model Explanation
- MFCC (Mel Frequency Cepstral Coefficients) used for audio feature extraction  
- Spectrograms converted into image format  
- CNN model trained for underwater sound classification  

---
## 🤖 Models Used

- Logistic Regression  
- Random Forest Classifier  
- Decision Tree Classifier  

Each model is evaluated using:
- Accuracy  
- Precision  
- Recall  
- F1 Score

## 🏆 Leaderboard System

A dynamic leaderboard tracks model performance based on:
- Accuracy  
- Precision  
- Recall  
- F1 Score  

Best performing model is automatically ranked at the top.

### ⚠️ Contribution Rules
- Do NOT delete existing entries  
- Only append new results  
- Maintain consistent formatting  

---

## 📊 Results

### Model Performance:

| Model | Accuracy |
|------|----------|
| Logistic Regression | 0.76 |
| Random Forest | 0.77 |
| Decision Tree | 0.76 |

### Baseline Comparison:
- Baseline Accuracy: ~0.62  
- Best Model Accuracy: ~0.77  
- Improvement: Significant improvement over baseline

### Visualization:
- Baseline vs Model accuracy graph included in outputs

### Model Performance
- Accuracy: 85% – 92% (based on tuning)

### Visual Outputs
- Confusion Matrix  
- Accuracy/Loss Graphs  
- Spectrogram Analysis  


---
## 🌍 Real-World Applications

- Monitoring marine traffic noise pollution  
- Detecting industrial underwater disturbances  
- Supporting marine ecosystem research  
- Assisting environmental protection agencies  

## 📌 Tech Stack
- Python  
- NumPy, Pandas  
- Librosa (audio processing)  
- Matplotlib, Seaborn  
- TensorFlow / PyTorch  

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python main.py

## 📈 Results

### Accuracy Graph
![Accuracy](results/accuracy.png)

### Confusion Matrix
![Confusion Matrix](results/confusion_matrix.png)

### Spectrogram Output
![Spectrogram](results/spectrogram.png)

