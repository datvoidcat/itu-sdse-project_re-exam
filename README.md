# Data Science in Production: MLOps and Software Engineering (Autumn 2025) – Exam Project

## Group Members
- **Aske Neye** — @askeneye  
- **Alex Soo** — @datvoidcat

---

## Project Overview

This repository contains our solution to the *Data Science in Production: MLOps and Software Engineering* exam project.

The goal of the project is to restructure a monolithic Jupyter notebook into a production-ready MLOps pipeline that:

- Pulls raw data via DVC  
- Cleans and processes features  
- Trains multiple machine learning models  
- Selects the best model  
- Produces a validated model artifact  
- Runs fully automated CI/CD via GitHub Actions  
- Uses **Dagger (Go SDK)** for containerized pipeline orchestration  

The pipeline identifies potential new customers based on user behavior data.

---

# Project Structure
itu-sdse-project_re-exam/
├── .github/workflows/      # GitHub Actions CI pipeline
├── dagger/                 # Dagger pipeline (Go SDK)
│ └── main.go
├── MLOps_Project/          # Python ML code
│ ├── pipeline.py
│ ├── loaders.py
│ ├── cleaners.py
│ └── config.py
├── data/
│ ├── raw/                  # Raw data (tracked via DVC)
│ └── processed/artifacts/  # Training artifacts
├── models/                 # Final exported model artifact (validator)
├── tests/                  # Inference validation script
├── requirements.txt
└── README.md

---

# Machine Learning Models

The training pipeline evaluates multiple candidate models:

- **XGBoost**
- **Logistic Regression**


## Important Design Decision

- The **best model (often XGBoost)** is saved in:
```text 
data/processed/artifacts/lead_model_xgboost.json
```

- A **validator-compatible model (Logistic Regression)** is always exported to:
```text 
models/model.pkl
```

This separation ensures:

- We retain the best-performing model.
- The GitHub validator (which does not install xgboost) can successfully load the artifact.
- The CI pipeline passes under constrained dependency environments.


---

# Dagger Pipeline

The pipeline is implemented in:
```text 
dagger/main.go
```
It defines:

- `train` → Pull data, run pipeline, export `models/model.pkl`
- `test` → Run inference validation script



---

## Github Actions Workflow

```text 
.github/workflows/action.yml
```

Triggered on:
- Push to main
- Pull request to main
- Manual dispatch

**Pipeline Steps**
- Checkout repository
- Setup Go
- Run Dagger training
- Verify models/model.pkl exists
- Run Dagger inference tests
- Upload artifact named model
- Run ITU model validator action
- The artifact uploaded is:

```text 
models/model.pkl
```


---

## Running Locally

### Prerequisites
- Docker running
- Dagger CLI installed
- Go installed

### Train the Model

From the project root:

```bash
```text 
cd dagger
go run main.go train
```

This will:
- Pull raw data via DVC
- Train candidate models
- Select the best model
- Export models/model.pkl (Logistic Regression for validator)

```text 
go run main.go test
```

This executes:
```text 
tests/model_inference.py
```
Which:
- Loads models/model.pkl
- Loads test data
- Runs predictions
- Verifies output consistency


