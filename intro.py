import streamlit as st

st.set_page_config(
    page_title="Auc Roc Explained",
    page_icon="👋",
)

st.sidebar.success("Select a section below")

st.write(
    """
    Initially, my intention for naming this app was "AUC ROC clearly explained." 
    However, I believe that name should be reserved for [statquest](https://www.youtube.com/@statquest), 
    as they specialize in explaining such concepts. 
    By the way, if you're not familiar with this channel, I recommend checking it out.
    
    Moving on, let's dive into the main topic. 
    This app aims to address various questions about AUC ROC. 
    It's important to note that this metric is not simple or straightforward, 
    as it encompasses numerous properties of a classifier.
    """
)

st.write("> To get most of this app, go through each section.")

st.write(
    """
# What we will learn in this module?

1. What is AUC-ROC? and How to plot it?
2. Back to reality
3. A random Classifier
4. Distinguishing Between a Good Classifier and a Bad Classifier
5. Influence of Class Imbalance
6. Selecting the Optimal Threshold from the AUC ROC Curve

Most important of all, we will try to build intuition for this metric.
"""
)
