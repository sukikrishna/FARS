import streamlit as st
from joblib import dump, load

import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import re

from datetime import datetime
from time import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, pairwise_distances
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import spacy

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_colwidth", None)


from collections import Counter
from nltk import ngrams
from itertools import chain


def app():

    # CHANGE category - done
    category = "Electronics"

    # CHANGE ML_model_file_path - done
    ML_model_file_path = "../Ester Tsai (Bigram and ML)/ML Models/"

    ML_model_file_name = f"2022_05_11 rf_model {(category)}.joblib"
    rf_model = load(ML_model_file_path + ML_model_file_name)

    # CHANGE bigram_file_path - done
    bigram_file_path = "../Ester Tsai (Bigram and ML)/Bigrams/"

    gold_bigram_file_name = f"{(category)} gold_bigrams.csv"
    gold_bigram_df = pd.read_csv(bigram_file_path + gold_bigram_file_name, index_col=0)
    # display(gold_bigram_df.head(3))

    fake_bigram_file_name = f"{(category)} fake_bigrams.csv"
    fake_bigram_df = pd.read_csv(bigram_file_path + fake_bigram_file_name, index_col=0)
    # display(fake_bigram_df.head(3))

    def data_cleaning(df):

        start_1 = time()

        # Removing emtpy cells
        df.dropna(inplace=True)
        df["review_cleaned"] = df["review_body"].copy()

        # Removing Unicode Chars (URL)
        df["review_cleaned"] = df["review_cleaned"].apply(lambda rev: re.sub(r"(\w+:\/\/\S+)|^rt|http.+?", "", rev))

        # Replace HTML keywords with blank space ("&quot;", "br", "&#34")
        remove_dict = {"<br /><br />": " ", "<br />": " ", "br ": "", "&quot;": " ", "&#34": " ", "<BR>": " ", "_": ""}
        for key, val in remove_dict.items():
            df["review_cleaned"] = df["review_cleaned"].apply(lambda x: x.replace(key, val))

        end_1 = time()

        print(f"\n######## [{end_1 - start_1:0.2f} secs] Remove URL and HTML Keywords Complete ########")

        start_2 = time()

        # Remove Punctuations and numbers
        tokenizer = RegexpTokenizer(r"\w+")
        df["review_cleaned"] = df["review_cleaned"].apply(lambda x: " ".join([word for word in tokenizer.tokenize(x)]))

        remove_dict = {"0": "", "1": "", "2": "", "3": "", "4": "", "5": "", "6": "", "7": "", "8": "", "9": "", "(": "", ")": ""}
        for key, val in remove_dict.items():
            df["review_cleaned"] = df["review_cleaned"].apply(lambda x: x.replace(key, val))

        end_2 = time()

        print(f"\n######## [{end_2 - start_2:0.2f} secs] Remove Punctuation and Numbers Complete ########")

        start_3 = time()

        # Lowercase Words
        df["review_cleaned"] = df["review_cleaned"].str.lower()

        end_3 = time()

        print(f"\n######## [{end_3 - start_3:0.2f} secs] Lowercase Complete ########")

        start_4 = time()

        # Remove Stop Words.
        stop = stopwords.words("english")

        df["review_cleaned"] = df["review_cleaned"].apply(
            lambda x: " ".join([word for word in x.split() if word.strip() not in stop])
        )

        end_4 = time()

        print(f"\n######## [{end_4 - start_4:0.2f} secs] Remove Stop Words Complete ########")

        start_5 = time()

        # Lemmatization using .lemma_
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        df["review_cleaned"] = df["review_cleaned"].apply(lambda x: " ".join([token.lemma_ for token in nlp(x)]))

        end_5 = time()

        print(f"\n######## [{end_5 - start_5:0.2f} secs] Lemmatization Complete ########")

        return df

    def find_ngrams(input_list, n):
        return list(zip(*[input_list[i:] for i in range(n)]))

    def add_bigram_column(df):
        copy = df.copy()
        copy["bigrams"] = copy["review_cleaned"].map(lambda x: find_ngrams(x.split(), 2))
        return copy

    def get_bigram_count(bigrams, bigram_dict):
        if len(bigrams) == 0:
            return 0

        count = 0
        for bigram in bigrams:
            if bigram in bigram_dict.keys():
                count += 1
        return count

    def get_count_percent(bigrams, bigram_dict):
        if len(bigrams) == 0:
            return 0

        count = get_bigram_count(bigrams, bigram_dict)
        return count / len(bigrams)

    def get_averge_top_score(bigrams, bigram_dict, topN=5):
        if len(bigrams) == 0:
            return 0

        scores = np.array([])
        for bigram in bigrams:
            if bigram in bigram_dict.keys():
                scores = np.append(scores, bigram_dict[bigram])

        if len(bigrams) <= topN:
            return scores.mean()

        sort_descending = -np.sort(-scores)[:topN]
        avg_top_score = sort_descending.mean()
        return avg_top_score

    def get_normalized_score(bigrams, bigram_dict):
        if len(bigrams) == 0:
            return 0

        score = 0
        for bigram in bigrams:
            if bigram in bigram_dict.keys():
                score += bigram_dict[bigram]
        return score / len(bigrams)

    def get_unique_percent(bigrams, bigram_dict, the_other_bigram_dict, unique_threshold=0):
        # Count the number of bigrams in a review that appear in bigram_dict but not the_other_bigram_dict.
        # Can adjust unique_threshold so you can count also the bigrams that appear in ...
        # ...the_other_bigram_dict fewer than unique_threshold times

        if len(bigrams) == 0:
            return 0

        count = get_bigram_count(bigrams, bigram_dict)
        for bigram in bigrams:
            if bigram in the_other_bigram_dict.keys():
                if the_other_bigram_dict[bigram] > unique_threshold:
                    count -= 1
        return count / len(bigrams)

    def has_fake_keywords(bigrams):
        fake_keywords = [
            "honest",
            "unbiased",
            "unbias",
            "biased",
            "bias",
            "neutral",
            "impartial",
            "truthful",
            "discount",
            "free",
            "promotion",
            "promote",
            "complimentary",
            "test",
            "influence",
            "influencer",
            "independent",
        ]
        fake_tuples = [
            ("receive", "product"),
            ("product", "receive"),
            ("provide", "review"),
            ("product", "exchange"),
            ("exchange", "review"),
            ("review", "opinion"),
            ("sample", "provide"),
            ("provide", "sample"),
            ("sample", "review"),
            ("review", "sample"),
            ("sample", "product"),
            ("supply", "sample"),
            ("receive", "sample"),
            ("sample", "receive"),
        ]

        for kw in fake_tuples:
            if kw in bigrams:
                return True

        for bigram in bigrams:
            for kw in fake_keywords:
                if kw in bigram:
                    return True

        return False

    def add_gold_fake_features(df):
        for gold_or_fake in ["gold", "fake"]:

            exec(
                f"df['{gold_or_fake}%'] = df['bigrams'].apply(\
                lambda x: get_count_percent(x, {gold_or_fake}_bigram_dict_filtered))"
            )

            exec(
                f"df['{gold_or_fake}_unique%'] = df['bigrams'].apply(\
                lambda x: get_unique_percent(x, {gold_or_fake}_bigram_dict, {gold_or_fake}_bigram_dict, 3))"
            )

            exec(
                f"df['{gold_or_fake}_score'] = df['bigrams'].apply(\
                lambda x: get_normalized_score(x, {gold_or_fake}_bigram_dict))"
            )

            exec(
                f"df['{gold_or_fake}_top_score'] = df['bigrams'].apply(\
                lambda x: get_averge_top_score(x, {gold_or_fake}_bigram_dict, 1))"
            )

            exec(
                f"df['{gold_or_fake}_top_avg_score'] = df['bigrams'].apply(\
                lambda x: get_averge_top_score(x, {gold_or_fake}_bigram_dict, 5))"
            )

        df["has_fake_keywords"] = df["bigrams"].apply(lambda x: has_fake_keywords(x))

        return df

    def gold_bigram_df_to_vars(gold_bigram_df):
        global gold_bigrams, gold_bigram_dict, gold_bigram_dict_filtered
        gold_bigrams = gold_bigram_df["bigram"].to_list()
        str_to_tuple = lambda x: (x.split("'")[1], x.split("'")[3])
        gold_bigrams = list(map(str_to_tuple, gold_bigrams))
        gold_bigram_dict = {gold_bigrams[i]: gold_bigram_df["count"].iloc[i] for i in range(len(gold_bigrams))}
        gold_bigram_dict_filtered = dict((k, v) for k, v in gold_bigram_dict.items() if v >= 2)

    def fake_bigram_df_to_vars(fake_bigram_df):
        global fake_bigrams, fake_bigram_dict, fake_bigram_dict_filtered
        fake_bigrams = fake_bigram_df["bigram"].to_list()
        str_to_tuple = lambda x: (x.split("'")[1], x.split("'")[3])
        fake_bigrams = list(map(str_to_tuple, fake_bigrams))
        fake_bigram_dict = {fake_bigrams[i]: fake_bigram_df["count"].iloc[i] for i in range(len(fake_bigrams))}
        fake_bigram_dict_filtered = dict((k, v) for k, v in fake_bigram_dict.items() if v >= 2)

    def clean_review_and_add_features(review, gold_bigram_df, fake_bigram_df):

        # Create dataframe for the input review
        df = pd.DataFrame(data={"review_body": review}, index=[0])

        # Create helper variables
        #     gold_bigram_df_to_vars(gold_bigram_df)
        #     fake_bigram_df_to_vars(fake_bigram_df)

        # Clean the user input
        df = data_cleaning(df)

        # Add the column 'bigrams'
        df["bigrams"] = df["review_cleaned"].map(lambda x: find_ngrams(x.split(), 2))
        df["bigram_count"] = df["bigrams"].apply(lambda x: len(x))

        # Add features to score the user input
        df = add_gold_fake_features(df)

        return df

    # Create helper variables (TAKES ~20 SECONDS TO LOAD)
    gold_bigram_df_to_vars(gold_bigram_df)
    fake_bigram_df_to_vars(fake_bigram_df)

    # streamlit app
    st.title("Amazon Reviews ML Model!!!")

    review_body_raw = st.text_input("Enter the review body")
    user_input_processed_df = clean_review_and_add_features(review_body_raw, gold_bigram_df, fake_bigram_df)

    # DISPLAY THIS TABLE!!!!!!!!!!!!!!!!!!!!!
    st.dataframe(user_input_processed_df)

    def prepare_df_for_prediction(processed_df):
        features = [
            "bigram_count",
            "fake%",
            "fake_unique%",
            "fake_score",
            "fake_top_avg_score",
            "fake_top_score",
            "gold%",
            "gold_unique%",
            "gold_score",
            "gold_top_avg_score",
            "gold_top_score",
            "has_fake_keywords",
        ]
        return processed_df[features]

    df_for_prediction = prepare_df_for_prediction(user_input_processed_df)

    prediction, probabilities = rf_model.predict(df_for_prediction)[0], rf_model.predict_proba(df_for_prediction)[0]
    # prediction: 0 = unverified, 1 = verified
    # probabilities[0] = prob of unverified  |  probabilities[1] = prob of verified

    def interpret_prediction(review, pred, proba):
        prob_unverified, prob_verified = round(proba[0] * 100, 1), round(proba[1] * 100, 1)
        st.subheader(f'\nREVIEW: "{review}" \n')
        if pred == 1:
            st.subheader(f"PREDICTION: VERIFIED ({prob_verified}%) | UNVERIFIED ({prob_unverified}%)")
        if pred == 0:
            st.subheader(f"PREDICTION: UNVERIFIED ({prob_unverified}%) | VERIFIED ({prob_verified}%)")

    interpret_prediction(review_body_raw, prediction, probabilities)

    """
    # What the Features Mean

    To reduce word space, let "gold" represent "verified," and let "fake" represent "unverified."
    NOTE: An "unverified" review isn't necessarily fake. A review is defined as "unverified" if the user did not buy the product from Amazon.

    ### Setup
    - `gold_bigram_dict` contains all the bigrams that appeared in verified reviews and their # of occurrences 
    - `fake_bigram_dict` contains all the bigrams that appeared in unverified reviews and their # of occurrences 

    ### Features
    `bigram_count` = # of bigrams in the review

    `gold%` = # bigrams in the review that appear in `gold_bigram_dict` at least 2 times / bigram_count

    `gold_unique%` = (# of bigrams that exist in `gold_bigram_dict` - # of bigrams that appear in `fake_bigram_dict` at least 3 times) / bigram_count

    `gold_score` = sum of all the bigrams' # of occurrences in verified reviews / bigram_count

    `gold_top_avg_score` = sum of the top 5 bigrams' # of occurrences in verified reviews / 5

    `gold_top_score` = highest # of occurrences in verified reviews

    Similar logic for the features `fake%`, `fake_unique%`, `fake_score`, `fake_top_avg_score`, and `fake_top_score`."""
