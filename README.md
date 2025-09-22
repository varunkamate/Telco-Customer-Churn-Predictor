📱 Telco Customer Churn Predictor
A comprehensive machine learning application that predicts customer churn for telecommunications companies using advanced analytics and an interactive web interface built with Streamlit.

=======================================================================

Data_Set_Link:=> 

=======================================================================

App_Live_Link:=> https://telco-customer-churn-predictor-h73gpj3rvqwgixpxc8rnmo.streamlit.app/

=======================================================================

🌟 Features
🤖 Machine Learning Models

Multiple Algorithm Support: Compare performance across Decision Tree, AdaBoost, XGBoost, and CatBoost classifiers
Advanced Preprocessing: Automated data cleaning, feature encoding, and standardization
Class Imbalance Handling: Uses RandomOverSampler to address imbalanced datasets
Model Persistence: Automatic saving of the best-performing model

📊 Interactive Web Application

Multi-page Interface: Clean navigation between Home, Prediction, Analytics, and About sections
Real-time Predictions: Input customer data and get instant churn probability scores
Risk Assessment: Visual indicators and actionable recommendations based on prediction results
Interactive Visualizations: Dynamic charts and graphs using Plotly for data exploration

📈 Analytics Dashboard

Performance Metrics: Track model accuracy, precision, recall, and F1-scores
Customer Insights: Analyze churn patterns by demographics, services, and billing information
Trend Analysis: Visualize customer retention and churn rates over time
Feature Importance: Understand which factors most influence customer churn

🛠️ Technical Stack

Backend: Python with scikit-learn, XGBoost, CatBoost
Frontend: Streamlit with custom CSS styling
Data Processing: Pandas, NumPy, imbalanced-learn
Visualizations: Plotly, Matplotlib, Seaborn
Model Management: Pickle for serialization

📋 Requirements
numpy==2.3.3
pandas==2.3.2
python-dateutil==2.9.0.post0
pytz==2025.2
six==1.17.0
tzdata==2025.2


🚀 Quick Start

Clone the repository

bashgit clone <repository-url>
cd telco-churn-predictor

Install dependencies

bashpip install -r requirements.txt
pip install streamlit plotly scikit-learn xgboost catboost imbalanced-learn

Run the Jupyter notebook to train models

bashjupyter notebook XG_BOOST.ipynb

Launch the web application

bashstreamlit run app.py

📖 Usage
Training Models
The XG_BOOST.ipynb notebook contains the complete machine learning pipeline:

Data exploration and visualization
Feature engineering and preprocessing
Model training and comparison
Performance evaluation and model selection

Web Application
Navigate through different sections:

Home: Overview and key performance metrics
Prediction: Input customer data for churn prediction
Analytics: Explore data patterns and trends

🎯 Model Performance
The application automatically selects the best-performing model based on ROC-AUC scores:

Accuracy: Up to 88.2%
F1-Score: Up to 89.1%
ROC-AUC: Up to 92.3%

🔍 Key Insights
The analysis reveals important churn predictors:

Contract type (month-to-month customers at higher risk)
Payment method (electronic check users show higher churn)
Tenure (newer customers more likely to churn)
Monthly charges and service combinations

streamlit run app.py
