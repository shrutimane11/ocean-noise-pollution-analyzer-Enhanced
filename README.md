# 🌊 Ocean Noise Pollution Analyzer (Enhanced)

A Machine Learning-based system for analyzing and classifying marine (ocean) noise pollution using audio signal processing, feature extraction, and model evaluation techniques.

---

## 🚀 Features
- Audio data preprocessing and cleaning  
- Ocean noise pattern analysis  
- Feature extraction using MFCC and Spectrograms  
- Machine Learning model evaluation  
- Data visualization using Matplotlib & Seaborn  
- Performance tracking using leaderboard system  

---

## 📂 Project Structure
data/ → Dataset files
src/ → Source code
outputs/ → Generated plots and results
leaderboard.csv → Model performance results
README.md → Project documentation

---

---

## 📊 Dataset Source
- Hydrophone audio recordings (marine sound dataset)  
- Public ocean acoustic datasets (NOAA / Kaggle / custom collected samples)  

---

## ⚙️ Workflow

Hydrophone Audio  
→ Preprocessing (Noise filtering, normalization)  
→ Feature Extraction (MFCC / Spectrogram)  
→ Machine Learning Model (CNN / LSTM)  
→ Classification (Ship / Marine life / Background noise)  
→ Visualization (Graphs & Audio analysis)  

---

## 🧠 Model Explanation
- MFCC (Mel Frequency Cepstral Coefficients) used for audio feature extraction  
- Spectrograms converted into image format  
- CNN model trained for underwater sound classification  

---

## 🏆 Leaderboard System
The `leaderboard.csv` file tracks model performance using:

- Accuracy  
- Precision  
- Recall  
- F1 Score  

### ⚠️ Contribution Rules
- Do NOT delete existing entries  
- Only append new results  
- Maintain consistent formatting  

---

## 📈 Results

### Model Performance
- Accuracy: 85% – 92% (based on tuning)

### Visual Outputs
- Confusion Matrix  
- Accuracy/Loss Graphs  
- Spectrogram Analysis  

## 🌍 Real-World Application
This system can help:

- Monitor marine traffic noise  
- Detect harmful underwater industrial activity  
- Support marine biodiversity conservation  
- Assist environmental research organizations  

---

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

