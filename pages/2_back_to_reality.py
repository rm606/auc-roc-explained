import streamlit as st
import numpy as np
from utils import *


def main():
    st.write(
        """
    # Back to reality

    The cases above are perfect cases where classifier is perfect. 
    But in reality you won't get such cases, 
    it is rare that you get a threshold where tpr = 1 and fpr = 0.

    Below is a more realistic classifier.
    """
    )

    m, n = 1000, 100
    some_rand_input = np.random.rand(m, n)
    labels = np.hstack(
        [
            np.ones(int(np.ceil(m / 2))).astype(np.uint8),
            np.zeros(m // 2).astype(np.uint8),
        ]
    )
    clf = DummyClassifier(pos_label_args=(0.65, 0.15), neg_label_args=(0.35, 0.15))
    probs = clf.predict(some_rand_input, y=labels)

    tpr, fpr = make_tpr_fpr(probs, labels)

    def f(threshold=0):
        plot2(probs, labels, tpr, fpr, threshold, area=True)

    slider_value3 = st.slider(
        "Select a threshold",
        min_value=0.0,
        max_value=1.0,
        step=0.1,
        format="%.1f",
        key="slider3",
    )
    f(slider_value3)

    st.write(
        """
        Take note of the overlap in model output between positive and negative labels. 
        To fully comprehend this metric, it is crucial to consider two elements:
        1. The presence of false positives, which occur to the right of 
        the threshold for negative labels (shown in red).
        2. The existence of false negatives, which occur to the left of 
        the threshold for positive labels (shown in blue).

        The objective of this metric is to minimize both false positives 
        and false negatives by reducing the overlapping areas.
    """
    )
    st.write("In brief, a higher AUC-ROC indicates a better classifier.")


st.set_page_config(page_title="Back To Reality")
st.sidebar.header("Back to Reality")
main()
