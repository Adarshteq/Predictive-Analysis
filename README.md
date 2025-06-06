# Diabetes Prediction Using Machine Learning

## Project Details

- **Company**: Codtech IT Solutions  

- **Name**: Addanki Adarsh  

- **Intern ID**: CT06DN221  

- **Domain**: Data Analytics  

- **Duration**: 6 weeks  

- **Mentor**: Neela Santhosh Kumar  

## Project Description

This project focuses on developing a machine learning model to predict the likelihood of diabetes in patients based on various health metrics.

Diabetes is a chronic disease that affects millions worldwide, and early detection can significantly improve treatment outcomes. 

The goal of this project is to create an accurate predictive model that can assist healthcare professionals in identifying at-risk patients.

![image](https://github.com/user-attachments/assets/d881c706-39a4-4331-bfec-56b43f6fd579)


### Dataset Overview

The dataset used in this project contains health information from 768 patients, with 8 clinical features and a binary outcome variable indicating the presence (1) or absence (0) of 

diabetes. The features include:

- Pregnancies: Number of times pregnant

- Glucose: Plasma glucose concentration

- BloodPressure: Diastolic blood pressure (mm Hg)

- SkinThickness: Triceps skin fold thickness (mm)

- Insulin: 2-Hour serum insulin (mu U/ml)

- BMI: Body mass index (weight in kg/(height in m)^2)

- DiabetesPedigreeFunction: Diabetes pedigree function

- Age: Age in years

### Methodology

The project follows a comprehensive data science workflow:

1. **Data Exploration**: Initial analysis of the dataset to understand distributions, correlations, and data quality

2. **Data Preprocessing**: Handling missing values, feature scaling, and train-test splitting

3. **Feature Selection**: Identifying the most predictive features using statistical methods

4. **Model Development**: Implementing and comparing multiple classification algorithms

5. **Model Evaluation**: Assessing performance using various metrics

6. **Hyperparameter Tuning**: Optimizing model performance through grid search

7. **Model Deployment**: Saving the final model for future use

### Technical Implementation

The project was implemented in Python using the following key libraries:

- Pandas and NumPy for data manipulation

- Scikit-learn for machine learning algorithms

- Matplotlib and Seaborn for data visualization

- Joblib for model serialization

Three classification models were developed and compared:

1. **Logistic Regression**: A linear model providing good interpretability

2. **Random Forest**: An ensemble method that handles non-linear relationships well

3. **Support Vector Machine**: Effective for high-dimensional data

### Key Findings

- The Random Forest model achieved the best performance with 79% accuracy and 0.87 ROC AUC score

- The most important predictive features were Glucose level, BMI, and Age

- The model shows good recall (ability to identify true positive cases), which is crucial for medical applications

- Hyperparameter tuning improved the model's performance by about 3-5%

### Business Impact

This predictive model can provide significant value to healthcare providers by:

- Enabling early intervention for at-risk patients

- Reducing healthcare costs through preventive care

- Supporting clinical decision-making with data-driven insights

- Potentially being integrated into health screening systems

### Challenges and Solutions

1. **Class Imbalance**: The dataset had more negative cases than positive. Addressed through stratified sampling during train-test split.

2. **Feature Correlation**: Some features showed multicollinearity. Handled through feature selection techniques.

3. **Model Interpretability**: Important for healthcare applications. Addressed by using Random Forest feature importance and logistic regression coefficients.

### Future Enhancements

1. Collect more data to improve model robustness

2. Experiment with advanced techniques like neural networks

3. Develop a web application for easy model deployment

4. Incorporate additional clinical features for better accuracy

5. Implement explainable AI techniques for better model interpretability

### How to Use This Repository

1. Clone the repository

2. Install required packages (`pip install -r requirements.txt`)

3. Run the Jupyter notebook `Diabetes_Prediction.ipynb`




### Acknowledgments

Special thanks to Codtech IT Solutions for providing this internship opportunity and to my mentor Neela Santhosh Kumar for guidance throughout this project. This experience has been invaluable in developing my data science skills and understanding real-world applications of machine learning in healthcare.
