import streamlit as st  # noqa: F401
from multiapp import MultiApp
import KNN_blank, KNN_w_examples, Bigram_blank  # also can import your app modules here w separate folder w from __folder__ # noqa: F401

app = MultiApp()

# Add all your application here
app.add_app("KNN Model", KNN_blank.app)
app.add_app("KNN Model with Examples", KNN_w_examples.app)
# app.add_app("Bigram Model", Bigram_blank.app)

# The main app
app.run()
