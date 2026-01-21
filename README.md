# 🎗️ Breast Cancer Prediction System

**CSC334 - Python II | Project 5**

A machine learning-based web application that predicts whether a breast tumor is benign or malignant using the K-Nearest Neighbors (KNN) algorithm.

---

## 📋 Project Information

| Field | Details |
|-------|---------|
| **Student Name** | Oluwalase Soboyejo |
| **Matric Number** | 23CD034363 |
| **Course** | CSC334 - Python II |
| **Project** | Breast Cancer Prediction System |

---

## ⚠️ Disclaimer

**IMPORTANT:** This system is strictly for **EDUCATIONAL PURPOSES ONLY** and must **NOT** be presented as a medical diagnostic tool. Always consult with qualified healthcare professionals for medical diagnosis.

---

## 🎯 Project Overview

This project develops a Breast Cancer Prediction System using the Breast Cancer Wisconsin (Diagnostic) dataset. The system uses a K-Nearest Neighbors (KNN) classifier to predict whether a tumor is:
- **Benign** (non-cancerous)
- **Malignant** (cancerous)

### Selected Features (5)

1. **Radius Mean** - Mean of distances from center to points on the perimeter
2. **Texture Mean** - Standard deviation of gray-scale values
3. **Perimeter Mean** - Mean size of the core tumor
4. **Area Mean** - Mean area of the tumor
5. **Concavity Mean** - Severity of concave portions of the contour

---

## 🗂️ Project Structure

```
BreastCancer_Project_OluwalaseSoboyejo_23CD034363/
├── app.py                              # Flask web application
├── requirements.txt                    # Python dependencies
├── Procfile                            # Deployment configuration
├── render.yaml                         # Render.com configuration
├── README.md                           # Project documentation
├── BreastCancer_hosted_webGUI_link.txt # Submission information
├── model/
│   ├── model_building.ipynb            # Jupyter notebook for model development
│   ├── train_model.py                  # Python script to train the model
│   └── breast_cancer_model.pkl         # Trained model file (generated)
├── static/
│   └── style.css                       # Stylesheet for web interface
└── templates/
    └── index.html                      # HTML template for web interface
```

---

## 🛠️ Technologies Used

| Category | Technology |
|----------|------------|
| **Programming Language** | Python 3.11+ |
| **Machine Learning** | scikit-learn (KNN) |
| **Web Framework** | Flask |
| **Model Persistence** | Joblib |
| **Data Processing** | NumPy, Pandas |
| **Visualization** | Matplotlib, Seaborn |
| **Production Server** | Gunicorn |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/BreastCancer_Project_OluwalaseSoboyejo_23CD034363.git
   cd BreastCancer_Project_OluwalaseSoboyejo_23CD034363
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model** (generates breast_cancer_model.pkl)
   ```bash
   python model/train_model.py
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Open in browser**
   ```
   http://localhost:5000
   ```

---

## 📊 Model Development (Part A)

The model development process is documented in `model/model_building.ipynb`. Key steps include:

### 1. Data Loading
- Loaded the Breast Cancer Wisconsin (Diagnostic) dataset from scikit-learn
- 569 samples with 30 features

### 2. Data Preprocessing
- ✅ Checked for missing values (none found)
- ✅ Selected 5 features from the recommended list
- ✅ Target variable encoding (0 = Malignant, 1 = Benign)
- ✅ Feature scaling using StandardScaler (mandatory for KNN)

### 3. Model Training
- Algorithm: **K-Nearest Neighbors (KNN)**
- Optimal K value determined through cross-validation
- Distance metric: Euclidean

### 4. Model Evaluation
- **Accuracy**: ~95%+
- **Precision**: High precision for both classes
- **Recall**: High recall for both classes
- **F1-Score**: Balanced F1-score

### 5. Model Persistence
- Saved using **Joblib**
- Model file: `breast_cancer_model.pkl`
- Includes: trained model, scaler, feature names

---

## 🌐 Web Application (Part B)

The web GUI is built using Flask and provides:

- Clean, responsive user interface
- Input form for tumor feature values
- Real-time prediction display
- Confidence percentage (when available)
- Sample values for testing

### Features
- Mobile-responsive design
- Input validation
- Error handling
- Health check endpoint (`/health`)

---

## ☁️ Deployment (Part D)

### Deploying to Render.com

1. Push code to GitHub
2. Create a new Web Service on Render.com
3. Connect your GitHub repository
4. Render will automatically detect the configuration

### Deploying to PythonAnywhere

1. Create a free account at pythonanywhere.com
2. Upload project files
3. Create a new web app with Flask
4. Configure WSGI file to point to your app

### Environment Variables

| Variable | Description |
|----------|-------------|
| `PORT` | Port number (default: 5000) |

---

## 📝 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Home page with prediction form |
| `/predict` | POST | Make a prediction |
| `/api/predict` | POST | JSON API endpoint |
| `/health` | GET | Health check |

### Example API Request

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "radius_mean": 14.5,
    "texture_mean": 19.0,
    "perimeter_mean": 92.0,
    "area_mean": 655.0,
    "concavity_mean": 0.09
  }'
```

---

## 📚 References

- [Breast Cancer Wisconsin Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
- [K-Nearest Neighbors Algorithm](https://scikit-learn.org/stable/modules/neighbors.html)
- [Flask Documentation](https://flask.palletsprojects.com/)

---

## 📄 License

This project is created for educational purposes as part of the CSC334 - Python II course.

---

## 👨‍💻 Author

**Oluwalase Soboyejo**
- Matric Number: 23CD034363
- Course: CSC334 - Python II

---

*Submission Deadline: Friday, January 22, 2026, on or before 11:59 PM*