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

    # def category_change(selected_category):

    selected_category = st.selectbox(
        "Choose A Product Category!",
        categories,
        # on_change=category_change(),
        # args=('Electronics',)
    )
    verified_checkbox = st.checkbox(
        "Verified Review",
        value=True,
    )

    path = "../KNNModelFiles/"
    name = "knn_working_model_updated.joblib"
    placeholder = ["placeholder"]
    current_field_value = {
        "Electronics": [
            {
                "review_body": "These are great rechargable batteries. Much easier to use than the type I used to have. They work right away - the charger is very easy to use - and they last a long long time!! Great way to do my little part to help the earth and save myself some $$$.",
                "review_headline": "I love an easy way to help the earth!! (and save $)",
                "product_title": "Sanyo Eneloop NiMH Battery Charger with 4AA NiMH Rechargable Batteries (Discontinued by Manufacturer)",
                "star_rating": 5,
                "helpful_votes": 2,
                "total_votes": 4,
                "verified_purchase": "N",
            },
            {
                "review_body": "cord fell APART WITH ALMOST NO USE! Not oem, very poor cheap design. do not buy! laptop will not recog properly.",
                "review_headline": "very bad product",
                "product_title": "Techno Earth® NEW AC Adapter/Power Supply Cord for HP/Compaq nx7400",
                "star_rating": 1,
                "helpful_votes": 0,
                "total_votes": 0,
                "verified_purchase": "Y",
            },
        ],
        "Beauty": [
            {
                "review_body": "Was recommended this product through a friend. She gave me sample from the bottle that she purchased. Started using the product and I saw results within a few days! Definetely will continue to use bFortuna Gold Allure as my skin care product. AMAZING product!!",
                "review_headline": "Amazing Product!",
                "product_title": "bFortuna Gold Allure - The GOLD STANDARD Vitamin C serum VEGAN CERTIFIED Vitamin C + Vitamin E + Hyaluronic Acid + Amino Complex serum. Anti-Aging serum to help improve the appearance of fine lines and wrinkles. Gold Allure's secret formula is The ONLY serum known to be used by DOCTORS due to the quantity and QUALITY of each ingredient that Gold Allure contains",
                "star_rating": 5,
                "helpful_votes": 2,
                "total_votes": 3,
                "verified_purchase": "N",
            },
            {
                "review_body": "They are exactly what I expected, they are excellently made. Not cheap, delivery was fast and punctual. Def. Will buy & support this seller again.",
                "review_headline": "Perfect!",
                "product_title": "Lot of 20 Dozen Assorted Small and Large Perm Rods",
                "star_rating": 5,
                "helpful_votes": 0,
                "total_votes": 0,
                "verified_purchase": "Y",
            },
        ],
        "Toys": [
            {
                "review_body": "The plastic cars come in 8 colors provided in the product description. The are well sculpted for playing pieces. But, BUT the cars received are not the cars from the photo in the description. The cars I got are much more cartoonish in design. For a more accurate idea of what cars you will be getting, look up Bolide or Detroit-Cleveland Grand Prix games.",
                "review_headline": "Nice cars. Wrong item pictured!",
                "product_title": "Plastic Race Car: Set of 16 Black, White, Red, Orange, Yellow, Green, Blue, and Purple Color Board Game Playing Pieces (Racing Car Tokens & Markers, Colored School Classroom Supplies, Arts & Crafts Projects, Teaching & Education Toy Vehicle Resource Components, Extra Instructional Play Materials)",
                "star_rating": 3,
                "helpful_votes": 1,
                "total_votes": 1,
                "verified_purchase": "N",
            },
            {
                "review_body": "Can't wait to introduce people that have not played thus game lol love it good fun times had by all as long as u r not too serious lol came very quick and we'll packed and very happy with it",
                "review_headline": "Love It fun game",
                "product_title": "Cards Against Humanity",
                "star_rating": 5,
                "helpful_votes": 0,
                "total_votes": 0,
                "verified_purchase": "Y",
            },
        ],
        "Office Products": [
            {
                "review_body": "This products are very bright and the chalk color stands out really well.",
                "review_headline": "Five Stars",
                "product_title": "CHALK MARKERS, Best 8 Pack Set of Bold Vibrant Colors by SillySticks, Erasable from Glass, Plastic, Mirrors, Metal, Whiteboards, Chalkboards, Perfect for Teachers, Kids and Moms, Nontoxic and Odorless, Enhance Your Arts and Crafts Projects Now!",
                "star_rating": 5,
                "helpful_votes": 0,
                "total_votes": 0,
                "verified_purchase": "N",
            },
            {
                "review_body": "Arrived very quickly, well packaged. Followed the provided instructions for installation and the emailed instructions for reset. Ran the internal print test and font list and the image looks good. Happy so far.",
                "review_headline": "Great Initial Results",
                "product_title": "EPS Replacement Brother TN450 Toner Cartridge, High Yield (2,600 Yield) - Black",
                "star_rating": 5,
                "helpful_votes": 0,
                "total_votes": 0,
                "verified_purchase": "Y",
            },
        ],
        "Apparel": [
            {
                "review_body": "a bit uncomfortable and when I put them on they get stretched out and looked faded",
                "review_headline": "One Star",
                "product_title": "The Walking Dead Daryl Transfer Print Socks",
                "star_rating": 1,
                "helpful_votes": 0,
                "total_votes": 0,
                "verified_purchase": "N",
            },
            {
                "review_body": "This was intended as a Christmas gift and I was not at all impressed with the condition of the item. The design was somewhat faded and it had such an ordor, I had to wash it before wrapping. Definitely wont be ordering again from this seller.",
                "review_headline": "Not happy",
                "product_title": "Juniors: Batman - Action Duo Juniors (Slim) T-Shirt Size M",
                "star_rating": 1,
                "helpful_votes": 0,
                "total_votes": 0,
                "verified_purchase": "Y",
            },
        ],
    }

    current_field_value = current_field_value[selected_category][int(verified_checkbox)]

    models = {
        "Electronics": "knn_electronics_million_model.joblib",
        "Beauty": "knn_beauty_model.joblib",
        "Toys": "knn_toys_model.joblib",
        "Office Products": "knn_office_products_model.joblib",
        "Apparel": "knn_apparel_model.joblib",
    }

    name = models[selected_category]

    product_title_raw = st.text_input(
        "Enter the product title", value=current_field_value["product_title"], placeholder=placeholder[0]
    )
    star_rating_raw = int(st.number_input("Enter the star rating", 1, 5, value=current_field_value["star_rating"]))
    review_title_raw = st.text_input(
        "Enter the review title", value=current_field_value["review_headline"], placeholder=placeholder[0]
    )
    review_body_raw = st.text_input("Enter the review body", value=current_field_value["review_body"], placeholder=placeholder[0])
    total_votes_raw = int(st.number_input("Enter the number of total votes", 0, None, value=current_field_value["total_votes"]))
    helpful_votes_raw = int(
        st.number_input("Enter the number of helpful votes", 0, int(total_votes_raw), value=current_field_value["helpful_votes"])
    )
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
    # st.write(product_title_sentiment, review_title_sentiment, review_body_sentiment)

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
