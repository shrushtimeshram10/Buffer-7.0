# 404Found

# Hybrid Decision Engine

## Overview
Hybrid Decision Engine is a loan approval system built as a **DSA-focused project** enhanced with **Machine Learning** for smarter and more accurate decisions.

The system combines **Data Structures and Algorithms** for fast processing, sorting, filtering, and rule-based evaluation, along with a **Random Forest model** that predicts whether a loan should be **Approved** or **Denied** based on applicant details.

This project demonstrates how traditional DSA concepts and modern AI techniques can work together in real-world financial systems.

---

# Video Link : 
https://drive.google.com/drive/folders/1turLjxJeEoh6zzDZwEoIF8h1VmE20HfI?usp=drive_link

## Core DSA Concepts Used

### Data Structures
- **Arrays / Lists** – Store applicant records and input values  
- **Hash Maps / Dictionaries** – Fast access to applicant fields  
- **Queues** – Handle multiple loan requests efficiently  
- **Trees** – Decision path logic and tree traversal  
- **Stacks** – Maintain decision trace steps  

### Algorithms
- Searching algorithms for validation  
- Sorting based on score / priority  
- Rule-based filtering  
- Threshold comparison logic  
- Efficient traversal for decision flow  

---

## Machine Learning Concepts Used

### Model Used
- **Random Forest Classifier**

### Why Random Forest?
- Uses multiple decision trees  
- Reduces overfitting  
- Gives better accuracy than a single tree  
- Works well on tabular loan datasets  

### ML Workflow
1. Dataset collection  
2. Data preprocessing  
3. Feature selection  
4. Train-test split  
5. Model training  
6. Prediction generation  
7. Confidence score output  

### Libraries Used
- Pandas  
- NumPy  
- Scikit-learn  
- Joblib  

---

## Features

✅ Loan approval / rejection prediction  
✅ DSA-based fast decision logic  
✅ Machine learning based accuracy  
✅ Confidence percentage display  
✅ Decision trace output  
✅ Frontend + Backend integration  
✅ Real-world finance use case  

---

## Tech Stack

### Frontend
- HTML  
- CSS  
- JavaScript  

### Backend
- Python  
- Flask  
- Flask-CORS  

### ML Tools
- Scikit-learn  
- Pandas  
- NumPy  

---

## Project Structure

```bash
project/
│── app.py
│── trainModel.py
│── loan_model.pkl
│── decisionSystem.html
│── loan_approval_dataset.csv
│── README.md


How to Run
1. Install Dependencies
pip install flask flask-cors pandas numpy scikit-learn joblib
2. Train Model
python trainModel.py
3. Run Backend
python app.py
4. Open Frontend

Open:

decisionSystem.html

in browser.

How It Works
User enters loan applicant details
Frontend sends request to backend
Backend organizes data using structures
ML model predicts approval result
Confidence score is calculated
Decision shown on dashboard
