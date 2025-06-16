import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter

# Set page config
st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="✉️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load and process data
@st.cache_data
def load_data():
    raw_mail_data = pd.read_csv('mail_data.csv')
    mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)), '')
    mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
    mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1
    return mail_data

mail_data = load_data()

# Sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Select a page:", 
                          ["Classifier", "Data Analysis", "Model Details", "About"])

if options == "Classifier":
    st.title("📧 Email Spam Classifier")
    st.markdown("""
    This app uses a Logistic Regression model to classify emails as **spam (0)** or **ham (1)**.
    """)
    
    # Input section
    st.subheader("Try it out!")
    msg = st.text_area("Enter your email text here:", height=200)
    
    if st.button("Classify"):
        if not msg.strip():
            st.warning("Please enter some text to classify.")
        else:
            # Load model (cached)
            @st.cache_resource
            def get_model():
                # Split data
                X = mail_data['Message']
                Y = mail_data['Category'].astype('int')
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
                
                # Feature extraction
                feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
                X_train_features = feature_extraction.fit_transform(X_train)
                
                # Train model
                model = LogisticRegression()
                model.fit(X_train_features, Y_train)
                return model, feature_extraction
            
            model, feature_extraction = get_model()
            
            # Process input
            input_mail = [msg]
            input_data_features = feature_extraction.transform(input_mail)
            prediction = model.predict(input_data_features)
            prediction_proba = model.predict_proba(input_data_features)
            
            # Display results
            st.subheader("Classification Result")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Prediction", value="Spam 🚫" if prediction[0] == 0 else "Ham ✅")
            
            with col2:
                confidence = prediction_proba[0][prediction[0]] * 100
                st.metric("Confidence", f"{confidence:.2f}%")
            
            # Show probabilities
            st.write("Detailed probabilities:")
            proba_df = pd.DataFrame({
                'Class': ['Spam (0)', 'Ham (1)'],
                'Probability': [prediction_proba[0][0]*100, prediction_proba[0][1]*100]
            })
            st.bar_chart(proba_df.set_index('Class'))
            
            # Show most important words
            st.subheader("Important Words in Prediction")
            coef = model.coef_[0]
            feature_names = feature_extraction.get_feature_names_out()
            top_spam_words = sorted(zip(feature_names, coef), key=lambda x: x[1], reverse=False)[:10]
            top_ham_words = sorted(zip(feature_names, coef), key=lambda x: x[1], reverse=True)[:10]
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("Top Spam Indicators:")
                for word, score in top_spam_words:
                    st.write(f"{word}: {score:.4f}")
            
            with col2:
                st.write("Top Ham Indicators:")
                for word, score in top_ham_words:
                    st.write(f"{word}: {score:.4f}")

elif options == "Data Analysis":
    st.title("📊 Data Analysis")
    st.subheader("Dataset Overview")
    
    st.write(f"Total emails: {len(mail_data)}")
    st.write(f"Spam emails: {len(mail_data[mail_data['Category'] == 0])}")
    st.write(f"Ham emails: {len(mail_data[mail_data['Category'] == 1])}")
    
    # Distribution plot
    fig, ax = plt.subplots()
    sns.countplot(x='Category', data=mail_data, ax=ax)
    ax.set_title('Distribution of Spam vs Ham')
    ax.set_xticklabels(['Spam (0)', 'Ham (1)'])
    st.pyplot(fig)
    
    # Word clouds
    st.subheader("Word Clouds")
    
    spam_text = " ".join(mail_data[mail_data['Category'] == 0]['Message'])
    ham_text = " ".join(mail_data[mail_data['Category'] == 1]['Message'])
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("Spam Words")
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(spam_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
    
    with col2:
        st.write("Ham Words")
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(ham_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
    
    # Message length analysis
    st.subheader("Message Length Analysis")
    mail_data['Message_Length'] = mail_data['Message'].apply(len)
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    sns.histplot(mail_data[mail_data['Category'] == 0]['Message_Length'], bins=50, ax=ax[0])
    ax[0].set_title('Spam Message Length')
    sns.histplot(mail_data[mail_data['Category'] == 1]['Message_Length'], bins=50, ax=ax[1])
    ax[1].set_title('Ham Message Length')
    st.pyplot(fig)

elif options == "Model Details":
    st.title("🤖 Model Information")
    
    st.subheader("Model Performance")
    
    # Split data
    X = mail_data['Message']
    Y = mail_data['Category'].astype('int')
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
    
    # Feature extraction
    feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
    X_train_features = feature_extraction.fit_transform(X_train)
    X_test_features = feature_extraction.transform(X_test)
    
    # Train model
    model = LogisticRegression()
    model.fit(X_train_features, Y_train)
    
    # Predictions
    train_pred = model.predict(X_train_features)
    test_pred = model.predict(X_test_features)
    
    # Metrics
    st.write(f"Training Accuracy: {accuracy_score(Y_train, train_pred):.4f}")
    st.write(f"Test Accuracy: {accuracy_score(Y_test, test_pred):.4f}")
    
    # Confusion matrix
    st.subheader("Confusion Matrix (Test Data)")
    cm = confusion_matrix(Y_test, test_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Spam', 'Predicted Ham'], 
                yticklabels=['Actual Spam', 'Actual Ham'])
    st.pyplot(fig)
    
    # Classification report
    st.subheader("Classification Report (Test Data)")
    report = classification_report(Y_test, test_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.highlight_max(axis=0))
    
    # Feature importance
    st.subheader("Top 20 Important Features")
    coef = model.coef_[0]
    feature_names = feature_extraction.get_feature_names_out()
    top_features = sorted(zip(feature_names, coef), key=lambda x: abs(x[1]), reverse=True)[:20]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x=[x[1] for x in top_features], y=[x[0] for x in top_features], ax=ax)
    ax.set_title('Top 20 Important Features')
    ax.set_xlabel('Coefficient Value')
    st.pyplot(fig)

elif options == "About":
    st.title("ℹ️ About This App")
    
    st.markdown("""
    ### Email Spam Classifier
    
    This application uses machine learning to classify emails as spam or ham (non-spam).
    
    **Features:**
    - Interactive classification of email text
    - Data visualization and analysis
    - Model performance metrics
    - Explanation of important features
    
    **Technical Details:**
    - Algorithm: Logistic Regression
    - Feature Extraction: TF-IDF (Term Frequency-Inverse Document Frequency)
    - Python Libraries:
        - Streamlit for the web interface
        - Scikit-learn for machine learning
        - Pandas for data processing
        - Matplotlib/Seaborn for visualization
        - WordCloud for text visualization
    
    **How to Use:**
    1. Go to the "Classifier" page
    2. Enter your email text in the text box
    3. Click "Classify" to see the prediction
    4. Explore other pages for data analysis and model details
    """)
    
    st.markdown("---")
    st.write("Created with ❤️ using Streamlit")
