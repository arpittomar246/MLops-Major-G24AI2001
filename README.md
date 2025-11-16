# Olivetti Face Classifier – MLOps End-to-End Pipeline

## Overview
This project implements a complete MLOps pipeline for training, containerizing, and deploying a face classification model using the Olivetti Faces Dataset. It covers all essential components of modern MLOps workflows:

- Machine Learning model training and evaluation
- Flask-based inference API and interactive web UI
- Docker containerization
- GitHub Actions CI/CD pipeline
- Model artifact handling
- Kubernetes deployment with multiple replicas
- Automated rollout updates

The system allows users to upload face images, processes them into the required format, runs inference using a trained Decision Tree Classifier, and displays predictions with clear explanations.

---

## Project Architecture
```
┌────────────┐      ┌──────────────┐      ┌──────────────┐      ┌─────────────┐
│  Training  │ ---> │  Artifact     │ ---> │  Docker Build │ ---> │ Kubernetes  │
│ (train.py) │      │  Storage      │      │  and Push     │      │ Deployment  │
└────────────┘      └──────────────┘      └──────────────┘      └─────────────┘
                                                       │
                                                       ▼
                                              Flask Web App (API and UI)
```

---

## Project Structure
```
MLops-Major-Project/
│
├── app.py                 # Flask web application
├── train.py               # Train DecisionTreeClassifier model
├── test.py                # Evaluate model accuracy
├── Dockerfile             # Application container
├── requirements.txt       # Python dependencies
├── models/                # Contains savedmodel.pth
│
├── static/
│   └── style.css          # Custom UI styling
│
├── templates/
│   └── index.html         # Main web UI template
│
└── k8s/                   # Kubernetes manifests
    ├── deployment.yaml
    ├── service.yaml
    ├── hpa.yaml (optional)
    └── ingress.yaml (optional)
```

---

## Features
### Machine Learning Component
- Uses Olivetti Faces Dataset (400 grayscale face images)
- Preprocessing includes image resizing and flattening
- Classifies images into one of 40 individuals (class 0 to 39)
- Displays top-k probabilities
- Provides detailed interpretation of output

### Web Application
- Built using Flask
- Dark-theme UI using Bootstrap
- Includes image preview before prediction
- Displays prediction with ranked probabilities
- Includes model description and usage instructions

### MLOps Pipeline
- Fully automated CI/CD pipeline using GitHub Actions
- Two-stage process:
  1. Train model and upload artifact
  2. Build Docker image with the model and push to DockerHub
- Automated validation, error checking, and consistency

### Kubernetes Deployment
- Multi-replica Deployment for scalability
- NodePort Service for exposure
- Liveness and readiness probes
- Optional Horizontal Pod Autoscaler (HPA)

---

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/MLops-Major-Project.git
cd MLops-Major-Project
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Model
```bash
python train.py
```
The trained model will be saved in `models/savedmodel.pth`.

### 5. Test the Model
```bash
python test.py
```

---

## Docker Setup
### Build the Image
```bash
docker build -t <yourdockerhub>/olivetti:latest .
```
### Run the Container
```bash
docker run -p 5000:5000 <yourdockerhub>/olivetti:latest
```
Open the application at:
```
http://127.0.0.1:5000
```

---

## Kubernetes Deployment
Ensure Kubernetes is enabled in Docker Desktop or Minikube.

### Apply Manifests
```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

### Check Resource Status
```bash
kubectl get pods
kubectl get svc
```

### Access Application
Open the NodePort URL:
```
http://localhost:<NodePort>
```

---

## CI/CD Pipeline
This project includes a complete GitHub Actions pipeline that:

1. Sets up Python environment
2. Installs dependencies
3. Trains the model
4. Uploads the trained model as an artifact
5. Downloads the artifact in the build stage
6. Builds Docker image with model embedded
7. Pushes image to DockerHub

---

## Prediction Workflow
- User uploads a face image
- Image is converted to 64x64 grayscale
- Model predicts class label (0 to 39)
- Web UI displays predictions, probabilities, and explanation

---

## Results
- Model accuracy displayed on UI
- Predictions with ranked probability scores
- Kubernetes ensures high availability and load balancing

---

## Model Artifacts
Model is saved as:
```
models/savedmodel.pth
```
It is automatically included inside the Docker image during CI.

---

## License
This project is intended for academic and learning purposes.

---

## Support
For help with deployment, debugging, Kubernetes, or documentation improvements, feel free to request assistance.

