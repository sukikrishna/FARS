import streamlit as st
from multiapp import MultiApp
import KNN_blank, KNN_w_examples, Bigram_blank # also can import your app modules here w seperate folder w from __folder__ 

app = MultiApp()

# Add all your application here
app.add_app("Home", KNN_blank.app)
app.add_app("Data Stats", KNN_w_examples.app)
app.add_app("Data Stats", Bigram_blank.app)

# The main app
app.run()
