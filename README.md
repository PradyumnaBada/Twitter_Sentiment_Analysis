# NLP with Disaster Tweets

This project uses Natural Language Processing (NLP) techniques to classify tweets as disaster-related or not. The notebook walks through the process of data loading, preprocessing, exploratory data analysis (EDA), feature engineering, applying baseline machine learning models, and evaluating their performance.

---

## 🚀 Project Overview

The primary goal is to build a model that can accurately distinguish tweets describing real disasters from those that are not. This involves several key NLP and machine learning steps.

---

## 📝 Notebook Structure and Key Steps:

1.  **Setup and Imports:**
    * Imports necessary libraries such as `numpy`, `pandas`, `matplotlib`, `seaborn`, `nltk`, and `sklearn`.
    * Downloads `stopwords` and `punkt` from `nltk`.

2.  **Data Loading:**
    * The data for this project is sourced from the [Kaggle "Natural Language Processing with Disaster Tweets" competition](https://www.kaggle.com/competitions/nlp-getting-started/data). You will need to download `train.csv` and `test.csv` from Kaggle.
    * The notebook reads these files into pandas DataFrames (originally by mounting Google Drive).

3.  **Initial Data Exploration (EDA) and Preprocessing:**
    * Displays the first few rows and info of the training and testing datasets.
    * Handles missing values in `keyword` and `location` columns by filling them.
    * **Text Cleaning:**
        * Removes URLs and punctuation from the tweet text.
        * Creates a new column `text_clean` with the cleaned text.
    * **Feature Creation for EDA:**
        * `Character_Count`: Calculates the length of the cleaned text.
        * `word_count`: Calculates the number of words in the original tweet text.
    * Analyzes `word_count` and `Character_Count` distributions for disaster vs. non-disaster tweets, noting that disaster tweets tend to have more words and characters on average.

4.  **Advanced Text Preprocessing for Modeling:**
    * **Tokenization:** Tokenizes the `text_clean` column.
    * **Lowercasing:** Converts tokens to lowercase.
    * **Stopword Removal:** Removes common English stopwords.
    * Creates a new column `stopwords_removed_sentence` by joining the processed tokens.

5.  **Exploratory Data Analysis (Post-preprocessing):**
    * Analyzes the frequency distribution of the top 50 most common words in disaster and non-disaster tweets (after stopword removal) to identify differing word usage patterns.

6.  **Data Preparation for Modeling:**
    * Splits the training data into training (90%) and validation (10%) sets.
    * **Feature Engineering (Vectorization):**
        * **Bag of Words (BoW):** Uses `CountVectorizer` (ngram_range=(1, 3), binary=True).
        * **TF-IDF:** Uses `TfidfVectorizer` (ngram_range=(1, 3), binary=True, smooth_idf=False).
        * **GloVe Embeddings:**
            * Downloads pre-trained GloVe Twitter embeddings (`glove-twitter-100`).
            * Converts sentences into embeddings by averaging word vectors.

7.  **Model Training and Evaluation (Baseline Models):**
    * Trains and evaluates Multinomial Naive Bayes, Logistic Regression, Random Forest, and a simple Neural Network using the different feature sets (BoW, TF-IDF, GloVe).

---

## 📊 Results and Performance

The notebook evaluates several baseline models with different text vectorization techniques. Key accuracy results are:

* **Multinomial Naive Bayes:**
    * With Bag of Words (BoW):
        * Validation accuracy: 0.768
        * Training accuracy: 0.984
    * With TF-IDF:
        * Validation accuracy: 0.780
        * Training accuracy: 0.983
* **Logistic Regression:**
    * With Bag of Words (BoW):
        * Validation accuracy: 0.776
        * Training accuracy: 0.985
    * With TF-IDF:
        * Validation accuracy: 0.790
        * Training accuracy: 0.919
    * With GloVe Embeddings (initial):
        * Validation accuracy: 0.811
        * Training accuracy: 0.800
    * With GloVe Embeddings (after GridSearchCV):
        * Best hyperparameters: {'C': 1, 'penalty': 'l1', 'solver': 'liblinear'}
        * Validation accuracy: 0.810
        * Training accuracy (with best_model): 0.801
* **Random Forest Classifier (n_estimators=200):**
    * With Bag of Words (BoW):
        * Validation accuracy: 0.769
        * Training accuracy: 0.988
    * With TF-IDF:
        * Validation accuracy: 0.764
        * Training accuracy: 0.988
    * The notebook also shows an attempt to run Random Forest with GloVe embeddings, but the accuracy results for this specific combination are not explicitly printed in the final part of the cell.
* **Simple Neural Network (PyTorch with GloVe):**
    * The training and validation losses are plotted across 100 epochs, showing the learning process. The training accuracy was 0.98, and the validation accuracy was 0.78.

**Summary of Best Performance (among baselines tested):**
Logistic Regression with GloVe embeddings (both initial and after GridSearchCV) showed the highest validation accuracy around 0.810 - 0.811. Traditional methods like BoW and TF-IDF with Logistic Regression and Naive Bayes also performed reasonably well, with validation accuracies generally in the range of 0.76 to 0.79.

---

## ⚙️ Dependencies:

* `numpy`
* `pandas`
* `matplotlib`
* `seaborn`
* `nltk`
* `scikit-learn` (`sklearn`)
* `gensim`
* `torch`

---

## 📋 How to Use:

1.  **Download the Data:** Obtain `train.csv` and `test.csv` from the [Kaggle "Natural Language Processing with Disaster Tweets" competition page](https://www.kaggle.com/competitions/nlp-getting-started/data).
2.  **Set up Environment:** Ensure all dependencies listed above are installed.
3.  **Update File Paths (If Necessary):** If you are not using Google Colab with the files in a specific Drive path mentioned in the notebook, you'll need to adjust the file loading paths in the notebook to where you've saved `train.csv` and `test.csv`.
    * The notebook originally uses: `/content/drive/MyDrive/CS_441_final_project/`
4.  **Run Notebook:** Execute the Jupyter notebook cells sequentially.

