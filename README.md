# 📧 Email Spam Classifier

A machine learning-powered web application that classifies emails as spam or ham (non-spam) using Logistic Regression with TF-IDF feature extraction.

## 🌟 Overview

This project provides an interactive interface to classify email messages as either spam or legitimate (ham). It includes:
- A trained machine learning model
- Data visualization tools
- Model performance metrics
- Explanations of how the classification works

## ✨ Features

- **Interactive Classification**: Enter email text and get instant spam/ham prediction
- **Visual Analytics**:
  - Word clouds for spam/ham messages
  - Message length distributions
  - Model performance metrics
- **Detailed Insights**:
  - Classification confidence scores
  - Top spam/ham indicator words
  - Feature importance visualization
- **User-Friendly Interface**: Clean, intuitive web interface with navigation

## 🛠️ How It Works

1. **Data Processing**:
   - Loads email dataset (spam/ham labeled)
   - Cleans and preprocesses text data
   - Converts labels to numerical values (0=spam, 1=ham)

2. **Feature Extraction**:
   - Uses TF-IDF (Term Frequency-Inverse Document Frequency) to convert text to numerical features
   - Removes stop words and applies lowercase conversion

3. **Model Training**:
   - Logistic Regression classifier
   - Trained on 80% of the data (20% held out for testing)

4. **Prediction**:
   - User inputs email text
   - Text is transformed using the same TF-IDF process
   - Model predicts probability of being spam/ham
   - Results displayed with confidence metrics

## 📊 Methods & Technologies

- **Machine Learning Algorithm**: Logistic Regression
- **Feature Extraction**: TF-IDF with Scikit-learn
- **Web Interface**: Streamlit
- **Data Visualization**:
  - Matplotlib
  - Seaborn
  - WordCloud
- **Data Processing**: Pandas, NumPy

