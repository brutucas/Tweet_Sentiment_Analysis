# NLP Challenge: Twitter Sentiment Analysis

In this challenge, you will work on analyzing sentiment in tweets from the Sentiment140 dataset, which contains 1.6 million tweets. You will clean the data, extract features from the text, train a sentiment analysis model, and visualize sentiment regarding an AI company or product of your choice.

## Challenge Overview

1\. **Dataset**: Download the Sentiment140 dataset from Kaggle [here](https://www.kaggle.com/datasets/kazanova/sentiment140). This dataset contains labeled tweets with sentiments (positive or negative).

2\. **Data Preprocessing**:

   - Read the dataset into a Pandas DataFrame.

   - Clean the text data as you see fit (e.g., removing special characters, handling missing values, stemming, or lemmatization). **NOTE**: you do NOT need to use every single tweet in the provided dataset.

3\. **Feature Extraction**:

   - Extract features from the text data. You can choose from various techniques such as:

     - TF-IDF (Term Frequency-Inverse Document Frequency).

     - Word embeddings using Gensim or other word embedding models.

     - Any other feature extraction method you find suitable.

4\. **Model Selection and Training**:

   - Choose a machine learning model of your preference (e.g., Logistic Regression, Random Forest, Support Vector Machine, XGBoost etc.).

   - Split your data into training and testing sets.

   - Train your model on the training data.

   - Evaluate the model's performance using accuracy metrics.

5\. **Sentiment Analysis**:

   - After training your model, perform sentiment analysis on the dataset. You can choose to focus on binary sentiment (positive/negative) or include a neutral category.

6\. **Visualizations on the Sentiment of an AI Company**:

   - Select an AI company or product of your choice. Search for tweets mentioning your choice in the 1.6 million tweet dataset (Make sure your selection has at least ~100 tweets)

   - Use your trained model to predict sentiment on tweets related to the chosen company or product.

   - Create visualizations (e.g., bar charts, word clouds, sentiment distribution plots) to showcase how people on Twitter feel about the selected company or product.

## Submission Guidelines

- Create a Jupyter Notebook or Python script to document your work.

- Include comments and explanations for each step of your code.

- Provide clear visualizations with appropriate labels and titles.

- You can use libraries like Matplotlib, Seaborn, or Plotly for your visualizations.

- Your final submission should include:

  - Code for data preprocessing, feature extraction, model training, and sentiment analysis.

  - Visualizations related to the chosen AI company or product sentiment.

  - A brief summary of your approach and results.

## Evaluation

You will be evaluated based on the following criteria:

- Data preprocessing and cleaning.

- Feature extraction method and rationale.

- Model selection, training, and evaluation.

- Accuracy and performance of sentiment analysis.

- Quality and clarity of visualizations.

- Code readability and documentation.

## Tips

- Experiment with different preprocessing techniques and feature extraction methods to improve your model's performance.

- Explore various machine learning algorithms to find the best one for sentiment analysis.

- Use cross-validation to fine-tune your model's hyperparameters.

- Pay attention to class imbalance if you choose binary sentiment analysis (positive/negative).

Good luck with the NLP Challenge, and enjoy exploring sentiment analysis on Twitter data!