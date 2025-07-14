# 🎓 College Student Placement Predictor

A full-stack machine learning web application that predicts whether a college student is likely to be placed based on their academic and skill profile.

![Streamlit App](https://img.shields.io/badge/Streamlit-Deployment-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![Scikit-Learn](https://img.shields.io/badge/ML-ScikitLearn-orange)

---

## 🚀 Project Overview

This application allows users to input student data — such as academic performance, IQ, internships, communication skills, and more — and uses a trained machine learning model to predict whether the student is likely to be placed.

### ✅ Features

- 🔍 Real-time prediction using a trained **Random Forest Classifier**
- 💡 Intuitive and responsive **Streamlit user interface**
- 🧠 Takes key academic and personal performance inputs
- ☁️ Fully deployable on **Streamlit Cloud** — no additional server required

---

## 🧠 Technologies Used

| Tool          | Purpose                             |
|---------------|-------------------------------------|
| Python        | Core programming language           |
| Pandas        | Data processing & transformation    |
| Scikit-learn  | Machine Learning model training     |
| Joblib        | Model serialization & loading       |
| Streamlit     | Frontend web application framework  |
| GitHub        | Version control and hosting         |

---

## 🛠️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/MYounus-Codes/Full-Stack-Machine-Learning-WebApp.git
cd Full-Stack-Machine-Learning-WebApp
````

### 2. Install the required packages

```bash
pip install -r requirements.txt
```

### 3. Run the application locally

```bash
streamlit run app.py
```

---

## 🌐 Live Demo

You can try out the live version here:
👉 **[Visit App](https://your-app-name.streamlit.app)**
---

## 📂 Project Structure

```
Full-Stack-Machine-Learning-WebApp/
├── app.py                       # Main Streamlit application
├── random_forest_model.joblib  # Pre-trained ML model file
├── requirements.txt            # Project dependencies
└── README.md                   # You're reading it!
```

---

## 🤖 Model Details

* **Algorithm**: Random Forest Classifier
* **Target Variable**: `Placement` (0 = Not Placed, 1 = Placed)
* **Accuracy**: 98.5%

### Input Features:

* IQ
* Previous Semester Result
* Academic Performance
* Internship Experience
* Communication Skills
* Projects Completed

---

## 📬 Contact

Built with 💻 by **Younus**
Feel free to reach out on [GitHub](https://github.com/MYounus-Codes)
(LinkedIn link can be added if you'd like)

---

## 🏁 Future Improvements

* [ ] Add CSV upload support for batch predictions
* [ ] Display prediction confidence score or probability
* [ ] Deploy with custom domain
* [ ] Add SHAP explainability for model interpretation

---

## 📜 License

This project is licensed under the **MIT License** — see the `LICENSE` file for full details.
