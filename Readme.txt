Here I took Two datasets of Bank churns to demonstrate the utilizing multiple datasets to create a scalable ml model. I used Google Colab instead Jupiter Notebook.
After exploring datasets, I did few things as below

----------------------------------------------------------------------------------------------------------------------------------------------------------

>>>>Removing columns that doesn't contribute much to final ML model and changing Column names of dataset-2 to make dataset consistent when training.

import pandas as pd

# Load both datasets
df1 = pd.read_csv('/content/data2/Bank_churn_Dataset1.csv')
df2 = pd.read_csv('/content/data2/Bank_churn_Dataset2.csv')

# Select and rename columns in df2 to match df1
df2_cleaned = df2[['clientnum', 'churn', 'customer_age', 'gender', 'months_on_book', 'total_relationship_count', 'income_category']]

# Rename columns to match df1
df2_cleaned.columns = ['customerid', 'churn', 'age', 'gender', 'tenure', 'numofproducts', 'estimatedsalary']

# Convert categorical 'income_category' to a numerical form for 'estimatedsalary'
# You may customize this mapping based on your preference
income_mapping = {
    'Less than $40K': 40000,
    '$40K - $60K': 50000,
    '$60K - $80K': 70000,
    '$80K - $120K': 100000,
    '$120K +': 120000,
    'Unknown': 0  # or handle it in another way
}
gender_mapping = {
    'M': 'Male',
    'F': 'Female',
    'Unknown': 0
}
df2_cleaned['estimatedsalary'] = df2_cleaned['estimatedsalary'].map(income_mapping)
df2_cleaned['gender'] = df2_cleaned['gender'].map(gender_mapping)
# Show the cleaned dataframe
print(df2_cleaned.head())

# Save the cleaned dataset to a new CSV file if needed
df2_cleaned.to_csv('Bank_churn_Dataset2_cleaned.csv', index=False)

--------------------------------------------------------------------------------------------------------------------------
# Load the first dataset (if not already loaded)
df1 = pd.read_csv('/content/data2/Bank_churn_Dataset1.csv')

# Select only the required columns
df1_cleaned = df1[['customerid', 'churn', 'age', 'gender', 'tenure', 'numofproducts', 'estimatedsalary']]

# Show the cleaned dataframe
print(df1_cleaned.head())

# Save the cleaned first dataset to a new CSV file if needed
df1_cleaned.to_csv('Bank_churn_Dataset1_cleaned.csv', index=False)

-------------------------------------------------------------------------------------------------------------------------------
>>>Analyzing data and performing PDA

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned datasets
df1_cleaned = pd.read_csv('Bank_churn_Dataset1_cleaned.csv')
df2_cleaned = pd.read_csv('Bank_churn_Dataset2_cleaned.csv')

# Function for quick EDA
def quick_eda(df, dataset_name):
    print(f"\n{dataset_name} - First 5 Rows:")
    print(df.head())
    
    print(f"\n{dataset_name} - Info:")
    print(df.info())
    
    print(f"\n{dataset_name} - Descriptive Statistics:")
    print(df.describe())
    
    # Plotting churn distribution
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x='churn', palette='Set2')
    plt.title(f'{dataset_name} - Churn Distribution')
    plt.xlabel('Churn')
    plt.ylabel('Count')
    plt.xticks(ticks=[0, 1], labels=['Not Churned', 'Churned'])
    plt.show()

# Perform EDA
quick_eda(df1_cleaned, 'Bank Churn Dataset 1')
quick_eda(df2_cleaned, 'Bank Churn Dataset 2')
---------------------------------------------------------------------------------------------------------------------------------------
>>>Preprocessing data

from sklearn.preprocessing import LabelEncoder, StandardScaler

# Combine datasets for preprocessing
combined_df = pd.concat([df1_cleaned, df2_cleaned], ignore_index=True)

# Encode categorical variables
label_encoder = LabelEncoder()
combined_df['gender'] = label_encoder.fit_transform(combined_df['gender'])

# Scaling numerical features
scaler = StandardScaler()
combined_df[['age', 'tenure', 'numofproducts', 'estimatedsalary']] = scaler.fit_transform(
    combined_df[['age', 'tenure', 'numofproducts', 'estimatedsalary']]
)

# Split back into original datasets
df1_processed = combined_df.iloc[:len(df1_cleaned)]
df2_processed = combined_df.iloc[len(df1_cleaned):]

# Verify the preprocessing
print(df1_processed.head())
print(df2_processed.head())
--------------------------------------------------------------------------------------------------------------------------------------------
>>>>Combining datasets


# Already processed in the previous step
combined_df = pd.concat([df1_cleaned, df2_cleaned], ignore_index=True)

# Encode categorical variables and scale features (similar as before)
combined_df['gender'] = label_encoder.fit_transform(combined_df['gender'])
combined_df[['age', 'tenure', 'numofproducts', 'estimatedsalary']] = scaler.fit_transform(
    combined_df[['age', 'tenure', 'numofproducts', 'estimatedsalary']]
)

----------------------------------------------------------------------------------------------------------------------------------------------
>>>>Using Logistic Regression for  training

# Prepare features and target using combined dataset
X = combined_df[['age', 'gender', 'tenure', 'numofproducts', 'estimatedsalary']]
y = combined_df['churn']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

--------------------------------------------------------------------------------------------------------------------------------------------------
>>>Creating User Interface for the ML model

!pip install gradio
--------------------------------------------------------------------------------------------------------------------------------------------------
import gradio as gr
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load and preprocess data
df1 = pd.read_csv('/content/Bank_churn_Dataset1_cleaned.csv')
df2 = pd.read_csv('/content/Bank_churn_Dataset2_cleaned.csv')

# Combine both datasets
combined_df = pd.concat([df1, df2], ignore_index=True)

# Encode categorical variables
label_encoder = LabelEncoder()
combined_df['gender'] = label_encoder.fit_transform(combined_df['gender'])

# Scale numerical features
scaler = StandardScaler()
combined_df[['age', 'tenure', 'numofproducts', 'estimatedsalary']] = scaler.fit_transform(
    combined_df[['age', 'tenure', 'numofproducts', 'estimatedsalary']]
)

# Prepare features and target
X = combined_df[['age', 'gender', 'tenure', 'numofproducts', 'estimatedsalary']]
y = combined_df['churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Define prediction function
def predict_churn(age, gender, tenure, numofproducts, estimatedsalary):
    # Preprocess input
    gender_encoded = label_encoder.transform([gender])[0]
    input_data = pd.DataFrame([[age, gender_encoded, tenure, numofproducts, estimatedsalary]],
                               columns=['age', 'gender', 'tenure', 'numofproducts', 'estimatedsalary'])
    input_data[['age', 'tenure', 'numofproducts', 'estimatedsalary']] = scaler.transform(input_data[['age', 'tenure', 'numofproducts', 'estimatedsalary']])
    
    # Make prediction
    prediction = model.predict(input_data)
    return "Churn" if prediction[0] == 1 else "No Churn"

# Create Gradio interface
inputs = [
    gr.Slider(minimum=18, maximum=100, label="Age"),
    gr.Radio(choices=["Male", "Female"], label="Gender"),
    gr.Slider(minimum=0, maximum=12, label="Tenure (in months)"),
    gr.Slider(minimum=1, maximum=5, label="Number of Products"),
    gr.Number(label="Estimated Salary")
]

outputs = gr.Textbox(label="Prediction")

gr.Interface(fn=predict_churn, inputs=inputs, outputs=outputs, title="Churn Prediction Model",
             description="Enter customer details to predict if they will churn.").launch()

----------------------------------------------------------------------------------------------------------------------------------------------------------------------

