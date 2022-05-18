import streamlit as st
import pandas as pd
import numpy as np

###
import re

import nltk.corpus

nltk.download("stopwords")
from nltk.corpus import stopwords

# from nltk.stem.porter import PorterStemmer

nltk.download("wordnet")
from nltk.stem import WordNetLemmatizer

##
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

### model
from joblib import load

# from sklearn.neighbors import KNeighborsClassifier

import sklearn  # noqa: F401

# sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base


###
# adi imports?
nltk.download("omw-1.4")


def app():
    def df_cleaning(df, col):
        # Drop rows with na values
        df.dropna(inplace=True)

        new_col_name = "new_" + col

        df[new_col_name] = df[col].copy()

        # Remove unwanted formatting characters
        format_strs = dict.fromkeys(["<br /><br />", "&#34", "br", "&quot", "<br />"], " ")

        for key in format_strs:
            df[new_col_name] = df[new_col_name].apply(lambda review: review.replace(key, format_strs[key]))
        # removing quotes produces smthg like this --> 'The product has great ;sound; --> we must remove punctuation

        # Case normalization (lower case)
        df[new_col_name] = df[new_col_name].str.lower()

        remove_dict = {"0": "", "1": "", "2": "", "3": "", "4": "", "5": "", "6": "", "7": "", "8": "", "9": "", "(": "", ")": ""}
        for key, val in remove_dict.items():
            df[new_col_name] = df[new_col_name].apply(lambda x: x.replace(key, val))

        # Remove stopwords
        stop_lst = stopwords.words("english")
        # stop_lst += (["can't", "i'm" "i'd", "i've", "i'll", "that's", "there's", "they're"])
        # ****Do we not have to take stopwords out BEFORE removing punctuation? Otherwise words with punct like “cant” remains there
        df[new_col_name] = df[new_col_name].apply(
            lambda text_body: " ".join([word for word in text_body.split() if word not in (stop_lst)])
        )

        # Removing Unicode Chars (punctuation, URL, @)
        df[new_col_name] = df[new_col_name].apply(
            lambda rev: re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", rev)
        )

        # Lemmatization
        word_lemmatizer = WordNetLemmatizer()
        df[new_col_name] = df[new_col_name].apply(
            lambda txt: " ".join([(word_lemmatizer.lemmatize(word)) for word in txt.split()])
        )

        return df

    analyser = SentimentIntensityAnalyzer()

    def get_sentiment_scores(review):
        """
        create new dataframe with just the proportions for each review
        four columns
        neg_prop, pos_prop, neu_prop, compound_prop and will contain these values
        obtained from the vator sentiment algorithm
        """
        snt = analyser.polarity_scores(review)
        # print(f"{sentence} {str(snt)}")
        return snt

    def id_for_dictionary(dic):
        if len(dic) == 4:
            ind = list(dic.values()).index(max(list(dic.values())[0:-1]))  # remove the compound
        else:
            ind = list(dic.values()).index(max(list(dic.values())))

        if ind == 1:
            return 0  # neutral
        elif ind == 0:
            return -1  # neg
        else:
            return 1  # positive

    def id_for_prop(prop):
        if prop < 0.45:
            return -1
        elif prop > 0.55:
            return 1
        else:
            return 0

    st.title("Amazon Reviews ML Model!!!")

    categories = ["Electronics", "Beauty", "Toys", "Office Products", "Apparel"]

    selected_category = st.selectbox(
        "Choose A Product Category!",
        categories,
    )

    path = "../KNNModelFiles/"
    name = "knn_working_model_updated.joblib"

    models = {
        "Electronics": "knn_electronics_million_model.joblib",
        "Beauty": "knn_beauty_model.joblib",
        "Toys": "knn_toys_model.joblib",
        "Office Products": "knn_office_products_model.joblib",
        "Apparel": "knn_apparel_model.joblib",
    }

    name = models[selected_category]

    product_title_raw = st.text_input("Enter the product title")
    star_rating_raw = int(st.number_input("Enter the star rating", 1, 5, 3))
    review_title_raw = st.text_input("Enter the review title")
    review_body_raw = st.text_input("Enter the review body")
    total_votes_raw = int(st.number_input("Enter the number of total votes", 0, None))
    helpful_votes_raw = int(st.number_input("Enter the number of helpful votes", 0, int(total_votes_raw)))
    # defaults to 1 if total_votes_raw changes

    df = pd.DataFrame(
        {
            "product_title_raw": [product_title_raw],
            "review_title_raw": [review_title_raw],
            "review_body_raw": [review_body_raw],
        }
    )

    df = df_cleaning(df, "product_title_raw")
    df = df_cleaning(df, "review_title_raw")
    df = df_cleaning(df, "review_body_raw")
    # st.write(df)

    product_title_sentiment = df["new_product_title_raw"].apply(get_sentiment_scores).iloc[0]
    review_title_sentiment = df["new_review_title_raw"].apply(get_sentiment_scores).iloc[0]
    review_body_sentiment = df["new_review_body_raw"].apply(get_sentiment_scores).iloc[0]

    # calculations displayed for coolness points
    "product_title_sentiment"
    product_title_sentiment

    "review_title_sentiment"
    review_title_sentiment

    "review_body_sentiment"
    review_body_sentiment

    chart_df = pd.DataFrame(
        [product_title_sentiment, review_title_sentiment, review_body_sentiment],
        index=["product_title_sentiment", "review_title_sentiment", "review_body_sentiment"],
    )
    chart_df = chart_df.transpose()
    st.dataframe(chart_df)
    st.bar_chart(chart_df)

    # inputs
    product_title = product_title_sentiment["compound"]
    star_rating = star_rating_raw
    review_title = review_title_sentiment["compound"]
    review_body = id_for_dictionary(review_body_sentiment)
    helpful_proportion_id = id_for_prop(0 if total_votes_raw == 0 else (helpful_votes_raw / total_votes_raw))

    args_for_KNN_model = np.array([[product_title, star_rating, review_title, review_body, helpful_proportion_id]])
    # st.write(args_for_KNN_model)

    # name = "knn_working_model_updated.joblib"
    # path = "../KNNModelFiles/"
    knn_classifier = load(path + name)
    prediction, probabilities = knn_classifier.predict(args_for_KNN_model), knn_classifier.predict_proba(args_for_KNN_model)[0]

    def interpret_prediction(review, pred, proba):
        proba = [round(proba[0], 3), round(proba[1], 3)]
        if prediction[0] == "Y":
            st.subheader(
                f"{review} is predicted to be a VERIFIED review, with {round(proba[1]*100, 2)}% probability of being VERIFIED and {proba[0]*100}% probability of being UNVERIFIED"
            )
        if prediction[0] == "N":
            st.subheader(
                f"{review} is predicted to be an UNVERIFIED review, with {round(proba[0]*100, 2)}% probability of being UNVERIFIED and {proba[1]*100}% probability of being VERIFIED"
            )

    interpret_prediction("This review", prediction, probabilities)

    # st.subheader(
    #    "This review is predicted to be a VERIFIED review, with 95.0% probability of being VERIFIED and 5.0% probability of being UNVERIFIED"
    # )
