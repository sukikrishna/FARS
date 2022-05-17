import streamlit as st  # noqa: F401
from multiapp import MultiApp
import KNN_blank, KNN_w_examples, Bigram_blank  # also can import your app modules here w separate folder w from __folder__

app = MultiApp()

# Add all your application here
app.add_app("KNN Blank", KNN_blank.app)
app.add_app("KNN with Examples", KNN_w_examples.app)
app.add_app("Bigram Blank - easy!", Bigram_blank.app)

# The main app
app.run()
