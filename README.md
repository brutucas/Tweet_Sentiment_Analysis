# NLP Challenge: Twitter Sentiment Analysis

In this project, I engaged in analyzing sentiment in tweets using the Sentiment140 dataset, which consists of 1.6 million tweets. I undertook the tasks of cleaning the data, extracting features from the text, training a sentiment analysis model, and visualizing sentiments regarding an AI company of my choice, specifically Microsoft.

## Project Overview

### 1. Dataset
I downloaded the Sentiment140 dataset from Kaggle [here](https://www.kaggle.com/datasets/kazanova/sentiment140). This dataset includes labeled tweets with sentiments categorized as positive or negative.

### 2. Data Preprocessing

- I loaded the dataset into a Pandas DataFrame.
- I cleaned the text data by removing special characters, handling missing values, and applying stemming and lemmatization techniques. I made a point not to use every single tweet in the provided dataset, focusing instead on a manageable subset for effective processing.

### 3. Feature Extraction

- For feature extraction from the text data, I opted to use:
  - TF-IDF (Term Frequency-Inverse Document Frequency) to transform the text into a meaningful representation of numbers which the model can understand.
  - I also considered other methods like word embeddings, but TF-IDF was sufficient for the initial stages of my analysis.

### 4. Model Selection and Training

- I chose Logistic Regression as the primary model for this analysis due to its effectiveness in binary classification tasks.
- I split the data into training and testing sets.
- The model was then trained on the training data.

### 5. Sentiment Analysis

- After training the model, I performed sentiment analysis specifically focusing on tweets mentioning Microsoft, filtering out approximately 500 tweets related to this AI-involved company.
- I then repeated the sentiment analysis using the same machine learning model.

### 6. Visualizations of Sentiment for Microsoft

- I used the trained model to predict sentiment on tweets related to Microsoft.
- I created various visualizations including heatmaps, bar charts, and word clouds to showcase how Twitter users feel about Microsoft.

## Submission Components

Included in my submission are:
- The code for data preprocessing, feature extraction, model training, and sentiment analysis.
- Visualizations concerning the sentiment related to Microsoft.
- A brief summary of my approach and results which highlights the accuracy score, confusion matrix, classification report, and the visualizations I generated.

## Evaluation and Adjustments

- I evaluated the model's performance using accuracy metrics and adjusted parameters using tools like pipeline configurations and GridSearchCV for optimization.

## Conclusion

This project allowed me to thoroughly explore the use of NLP in sentiment analysis on Twitter data, focusing on a specific AI company. The insights gained through this project were substantiated by the quantitative outputs from the model and the qualitative assessments from the visual data representations.

- Use cross-validation to fine-tune your model's hyperparameters.

- Pay attention to class imbalance if you choose binary sentiment analysis (positive/negative).

Good luck with the NLP Challenge, and enjoy exploring sentiment analysis on Twitter data!
