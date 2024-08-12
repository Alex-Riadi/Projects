#packages-----------------------------------------------------------------------------------------------------------------
#region
# for data manipulation
import pandas as pd
import numpy as np

# for data visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# for preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge 
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor

# for model training
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# import model for regression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ipmort LGBM
from lightgbm import LGBMClassifier

# import catboost 
from catboost import CatBoostClassifier

# import tensorflow for creating neural networks
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

# for model evaluation
from sklearn.metrics import accuracy_score, classification_report

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

# function for Sub-Heading
def heading(title):
    print('-'*80)
    print(title.upper())
    print('-'*80)
#endregion

#load dataset
df = pd.read_csv('C:\\Users\\alexr\\Desktop\\   \\Work\\Data Science\\In Use\\Student Study Performances\\study_performance.csv')

#Statistical Summary-----------------------------------------------------------------------------------------------------

#data overview
df.head()

# Displaying the information about the DataFrame
df.info()

# Displaying the statistical summary of the DataFrame
df.describe()

#Shape of the dataset
# No of rows and columns
sh = df.shape
print(f'There are {sh[0]} rows and {sh[1]} columns in the dataset.')

#Missing values

# Calculate the percentage of missing values in each column
a = df.isnull().sum().sort_values(ascending=False) * 100 / len(df)

# Iterate over each column and its corresponding missing value percentage
for col, percentage in a.items():

    # Print the column name and its missing value percentage with 2 decimal places
    # '<30' in the f-string format specifies left alignment with a field width of 30 characters for the column names.
    print(f'{col:<30} {percentage:.2f}%')


#Heatmap for missing values

# Visualize the missing values
plt.figure(figsize=(18, 9))
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='RdPu')
plt.show()

#No of Catagorical and Numerical Columns:

# Initialize counters for categorical and numerical columns
objectt = 0
integers = 0

# Iterate over each column in the DataFrame
for i in df.columns:
    # Check if the column dtype is 'object', indicating a categorical column
    if df[i].dtype == 'object':
        # Increment the categorical column counter
        objectt += 1
    # Check if the column dtype is 'int64', indicating a numerical column
    elif df[i].dtype == 'int64':
        # Increment the numerical column counter
        integers += 1

# Print the total count of categorical and numerical columns in the dataset
print(f'There are {objectt} categorical and {integers} numerical columns in the dataset')


#Heatmap for correlation

# Initialize LabelEncoder object
label_encode = LabelEncoder()

# Create a copy of the DataFrame
df2 = df.copy()

# Iterate over columns in DataFrame
for i in df2.columns:
    # If column type is 'object', encode it using LabelEncoder
    if df2[i].dtype == 'object':
        df2[i] = LabelEncoder().fit_transform(df2[i])

# Plot correlation heatmap of modified DataFrame
plt.figure(figsize=(18, 12), facecolor='none')
sns.heatmap(
    df2.corr(),
    cmap=sns.diverging_palette(230, 60, as_cmap=True),
    annot=True,
    linewidths=0.2,
    annot_kws={'size': 12, 'weight': 'bold', 'color': 'black'},
    fmt='.2f',
)
plt.xticks(fontsize=14, weight='bold', rotation=45)
plt.yticks(fontsize=14, weight='bold')
plt.show()

#Data Deep Exploration--------------------------------------------------------------------------------------------------

#Function for bar charts
#def bar_charts(x, y, title):
"""
    Generate a bar chart using Plotly Express.

    Parameters:
    - x: Data for the x-axis (e.g., categories).
    - y: Data for the y-axis (e.g., corresponding values).
    - title: Title of the chart.

    Returns:
    - None
"""
    # Create a bar chart using Plotly Express
fig = px.bar(df,
        x= 'gender',  # Data for the x-axis
        y= 'math_score',  # Data for the y-axis
        title='title',  # Title of the chart
        color='math_score',  # Color the bars based on y-values
        labels={'x': 'Profession', 'y': 'Average Income'},  # Custom axis labels
        text='math_score',  # Add text labels to the bars
        orientation='h',
    )

#fig.show()
    # Customize the layout
fig.update_layout(
        paper_bgcolor='#111',  # Set the background color of the entire plot
        plot_bgcolor='#111',   # Set the background color of the plot area
        font_color='white',     # Set the font color
        font=dict(size=20)     # Set the font size
    )

fig.show()

    # Display the figure

#Function for Pie charts & Bar charts
#region
#def single_plot_distribution(column_name, dataframe, title):
"""
    Generate a pie chart and a bar chart to visualize the distribution of values in a single column.

    Parameters:
    - column_name: Name of the column to visualize.
    - dataframe: DataFrame containing the data.
    - title: Title of the plots.

    Returns:
    - None
"""
# Get the value counts of the specified column
value_counts = df['math_score'].value_counts()

# Set up the figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9), facecolor='black') 

# Set main title for the figure
fig.suptitle('title')

# Pie chart
pie_colors = ['#0077b6', '#00b4d8', '#90e0ef', '#caf0f8']
ax1.pie(value_counts, autopct='%0.001f%%', startangle=100, textprops={'fontsize': 20}, pctdistance=0.75, colors=pie_colors, labels=None)
centre_circle = plt.Circle((0,0),0.40,fc='black')
ax1.add_artist(centre_circle)
ax1.set_title(f"Distribution of {'math_score'}", fontsize=16, color='white')

# Bar chart
sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax2, palette=pie_colors) 
ax2.set_title(f"Count of {'math_score'}", fontsize=16, color='white')
ax2.set_xlabel('math_score', fontsize=14, color='white')
ax2.set_ylabel('Count', fontsize=14, color='white')

# Rotate x-axis labels for better readability
ax2.tick_params(axis='x', rotation=45, colors='white')
ax2.tick_params(axis='y', colors='white')

# Set background color for the subplots
ax1.set_facecolor('black')
ax2.set_facecolor('black')

# Show the plots
plt.tight_layout()
plt.show()

#Function for Sunburst
def sun_brust(data, path, color_map, title):
    """
    Generate a sunburst chart using Plotly Express.

    Parameters:
    - data: DataFrame or array-like object containing the data.
    - path: List-like object specifying the hierarchical structure of the sunburst chart.
    - color_map: Dictionary mapping values to colors for coloring the segments.
    - title: Title of the chart.

    Returns:
    - None
    """
    # Create a sunburst chart using Plotly Express
    fig = px.sunburst(
        data,  # Data for the chart
        path=path,  # Specifies the hierarchical structure
        color=path[0],  # Color the segments based on the first element of the path
        color_discrete_map=color_map  # Map values to colors
    )

    # Customize the layout
    fig.update_layout(
        title=title,  # Set the title of the chart
        paper_bgcolor='#111',  # Set the background color of the entire plot
        plot_bgcolor='#111',   # Set the background color of the plot area
        font_color='white',     # Set the font color
        font=dict(
            family='Comic Sans MS',  # Set the font family
            size=20,  # Set the font size
            color='white'  # Set the font color
        ),
        width=1000,  # Set the width of the plot
        height=600   # Set the height of the plot
    )

    # Update traces for markers and text
    fig.update_traces(
        marker=dict(line=dict(color='white', width=2)),  # Set the marker color and width
        textfont=dict(size=15)  # Set the text font size
    )

    # Display the chart
    fig.show()



#Gender of Students by Test Preparation Course

# Group the DataFrame by 'gender' and 'test_preparation_course', and count the occurrences
gn = df.groupby('gender')['test_preparation_course'].value_counts()

# Set up the figure for plotting
plt.figure(figsize=(18, 9))

# Create a count plot with seaborn, grouping by 'gender' and coloring by 'test_preparation_course'
sns.countplot(x='gender', hue='test_preparation_course', data=df, palette='Blues')

# Set the title of the plot
plt.title('Gender of Students by Test Preparation Course', fontdict={'size': 18, 'color': 'black', 'weight': 'bold'})

# Set labels for x-axis and y-axis
plt.xlabel('Gender', fontdict={'size': 14, 'color': 'black', 'weight': 'bold'})
plt.ylabel('Count', fontdict={'size': 14, 'color': 'black', 'weight': 'bold'})

# Display the plot
plt.show()

#Parental Level of Education by Gender and Test Preparation Course
#region
# Define a dictionary to map values to colors for the sunburst chart
color_map = {
    'male': 'darkblue',     # Color for 'male' category
    'female': '#90e0ef',    # Color for 'female' category
}

# Define the title for the sunburst chart
title = 'Parental Level of Education by Gender and Test Preparation Course'

# Call the sun_brust function with appropriate parameters
sun_brust(df,                                     # DataFrame containing the data
          ['gender', 'parental_level_of_education'],  # Hierarchical structure for the sunburst chart
          color_map,                               # Color map for segments
          title)                                   # Title of the chart
#endregion


# Plot the distribution of lunch in the dataset
single_plot_distribution('lunch', df, 'Lunch Distribution')


# Set up the figure for plotting
plt.figure(figsize=(15, 9))

# Create a count plot with seaborn, specifying the DataFrame (df), x-axis (parental_level_of_education), and hue (parental_level_of_education)
sns.countplot(data=df, x='parental_level_of_education', hue='parental_level_of_education')

# Set the title of the plot
plt.title('Parental Level of Education', fontdict={'size': 18, 'color': 'black', 'weight': 'bold'})

# Set labels for x-axis and y-axis
plt.xlabel('Parental Level of Education', fontdict={'size': 14, 'color': 'black', 'weight': 'bold'})
plt.ylabel('Count', fontdict={'size': 14, 'color': 'black', 'weight': 'bold'})

# Display the plot
plt.show()



# Set up the figure for plotting
plt.figure(figsize=(15, 9))

# Create a count plot with seaborn, specifying the DataFrame (df), x-axis (race_ethnicity), and hue (race_ethnicity)
sns.countplot(data=df, x='race_ethnicity', hue='race_ethnicity')

# Set the title of the plot
plt.title('Race and Ethnicity', fontdict={'size': 18, 'color': 'black', 'weight': 'bold'})

# Set labels for x-axis and y-axis
plt.xlabel('Race', fontdict={'size': 14, 'color': 'black', 'weight': 'bold'})
plt.ylabel('Count', fontdict={'size': 14, 'color': 'black', 'weight': 'bold'})

# Display the plot
plt.show()



# Group the DataFrame by 'gender' and calculate the mean of 'writing_score'
nm = df.groupby('gender')['writing_score'].mean().round(2)

# Call the bar_charts function to generate a bar chart
bar_charts(
    nm.index,           # Data for the x-axis (gender)
    nm.values,          # Data for the y-axis (average writing score)
    'Average Writing Score by Gender'  # Title of the chart
)



# Group the DataFrame by 'gender' and calculate the mean of 'math_score'
np = round(df.groupby('gender')['math_score'].mean().round(1))

# Call the bar_charts function to generate a bar chart
bar_charts(
    np.index,           # Data for the x-axis (gender)
    np.values,          # Data for the y-axis (average math score)
    'Average Maths Score by Gender'  # Title of the chart
)



# Group the DataFrame by 'gender' and calculate the mean of 'reading_score'
np = df.groupby('gender')['reading_score'].mean().round(2)

# Call the bar_charts function to generate a bar chart
bar_charts(
    np.index,           # Data for the x-axis (gender)
    np.values,          # Data for the y-axis (average reading score)
    'Average Reading Score by Gender'  # Title of the chart (corrected)
)

# Plot the distribution of parental level of education
single_plot_distribution('parental_level_of_education', df, 'Lunch Distribution')

X= df.drop('test_preparation_course', axis=1)
y = df['test_preparation_course']
# Define column transformer for feature encoding
column_trans = ColumnTransformer([
    ('one', OneHotEncoder(), ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch'])
], remainder='passthrough')

# Split data into training and testing sets

# Define models
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Support Vector Machine': SVC(),
    'XGBoost': XGBClassifier(),
}

# Encode target variable y
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)



# List to store accuracy scores of different models
accuracies = []

# Iterate over models and evaluate
for model_name, model in models.items():
    print(f"Evaluating {model_name}:")
    
    # Define pipeline for each model
    pipe = Pipeline([
        ('column_trans', column_trans),
        ('model', model)
    ])
    
    # Train model
    pipe.fit(X_train, y_train)
    
    # Predict
    y_pred = pipe.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    print(f"Accuracy: {accuracy}\n")


# Create a DataFrame to store model names and their corresponding accuracies
acc_df = pd.DataFrame({'Model': models.keys(), 'Accuracy': accuracies})

# Call the bar_charts function to generate a bar chart
bar_charts(
    acc_df['Model'],     # Data for the x-axis (model names)
    acc_df['Accuracy'],  # Data for the y-axis (model accuracies)
    'Model Accuracy'     # Title of the chart
)

# Define column transformer for feature encoding
column_trans = ColumnTransformer([
    ('one', OneHotEncoder(), ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch'])
], remainder='passthrough')

X_train_Scaled = column_trans.fit_transform(X_train)
X_test_Scaled = column_trans.transform(X_test)

# Define the architecture of the neural network
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_Scaled.shape[1],)),  # Input layer with 64 neurons and ReLU activation
    tf.keras.layers.Dense(128, activation='relu'),  # Hidden layer with 128 neurons and ReLU activation
    tf.keras.layers.Dense(64, activation='relu'),   # Hidden layer with 64 neurons and ReLU activation
    tf.keras.layers.Dense(32, activation='relu'),   # Hidden layer with 32 neurons and ReLU activation
    tf.keras.layers.Dense(16, activation='relu'),   # Hidden layer with 16 neurons and ReLU activation
    tf.keras.layers.Dense(8, activation='relu'),    # Hidden layer with 8 neurons and ReLU activation
    tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer with 1 neuron (for binary classification) and Sigmoid activation
])

# Compile the model
model.compile(optimizer='Adam',  # Use Adam optimizer
              loss='binary_crossentropy',  # Use binary crossentropy loss function (suitable for binary classification)
              metrics=['accuracy'])  # Use accuracy as the evaluation metric

# Define early stopping to prevent overfitting
# early_stopping = EarlyStopping(patience=50, monitor='val_loss')  # Stop training if validation loss doesn't improve for 5 epochs

# Train the model
history = model.fit(X_train_Scaled,           # Training data
                    y_train,                  # Target data
                    epochs=100,               # Number of epochs for training
                    validation_data=(X_test_Scaled, y_test),  # Validation data
                #     callbacks=[early_stopping],             # Use early stopping
                    batch_size=32,            # Batch size for training
                    verbose=1)                # Show training progress
#endregion