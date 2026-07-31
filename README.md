# 🏀 HoopIQ

### Predict NBA Player Performance Using Machine Learning

An end-to-end machine learning application that predicts an NBA player's next-game **points, rebounds, and assists** using historical NBA data, custom feature engineering, and predictive modeling.

---

## 🌐 Live Demo

**Application:** https://hoopiq-nba.streamlit.app/

**GitHub Repository:** https://github.com/devan-jariwala/HoopIQ

---

## 📸 Application Preview

### Homepage

![Homepage](images/homepage.png)

### Example Prediction

![Prediction](images/prediction.png)

---

# 🎯 Project Overview

HoopIQ is an end-to-end machine learning application that predicts an NBA player's next-game **points, rebounds, and assists** using historical NBA game data.

The project combines data engineering, feature engineering, machine learning, and web deployment into a complete analytics workflow. Rather than relying on simple season averages, HoopIQ learns from each player's recent performance, long-term trends, opportunity metrics, and consistency to generate predictions for a hypothetical next-game scenario.

The application was built entirely in Python using official NBA API data and deployed as an interactive Streamlit web application.

---

# ❓ Business Problem

Player performance prediction is one of the most challenging problems in sports analytics because every game is influenced by countless factors including recent performance, playing time, shooting volume, rest, and natural game-to-game variability.

The objective of HoopIQ is to answer the following business question:

> **Given everything known about a player before a game begins, can we accurately predict how many points, rebounds, and assists they will record?**

To answer this question, the project builds predictive models using only information that would have been available before tipoff, preventing data leakage while simulating a real-world prediction environment.