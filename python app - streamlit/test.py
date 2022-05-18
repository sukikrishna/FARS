import streamlit as st
import numpy as np
import pandas as pd
import time

my_bar = st.progress(0)

# for percent_complete in range(100):
#     time.sleep(0.1)
#     my_bar.progress(percent_complete + 1)


chart_data = pd.DataFrame(np.random.randn(5, 3), columns=["a", "b", "c"])
chart_data["index"] = ["Thing1", "thing2", "thing3", "thing4", "thing5"]
chart_data = chart_data.set_index("index")
st.dataframe(chart_data)
st.area_chart(chart_data)
"ello"


{"neg": 0, "neu": 0, "pos": 0, "compound": 0}
