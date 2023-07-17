import streamlit as st
import numpy as np
from utils import *


def main():
    st.write(
        """
            # Selecting the Optimal Threshold from the AUC ROC Curve

            Now that we understand the ROC curve, with TPR on the y-axis
            and FPR on the x-axis, our objective is to maximize the TPR
            and minimize the FPR. The threshold that achieves this balance
            is the desired solution.

            Using the provided widget, can you identify such a threshold? 
            It should be the one that strikes the optimal trade-off between TPR and FPR.

            For me, it would be 0.4 or 0.5. What do you think?
        """
    )
    m, n = 10000, 100
    some_rand_input = np.random.rand(m, n)
    labels = np.hstack(
        [np.ones(m // 2).astype(np.uint8), np.zeros(m // 2).astype(np.uint8)]
    )
    clf = DummyClassifier(pos_label_args=(0.60, 0.15), neg_label_args=(0.4, 0.15))
    probs = clf.predict(some_rand_input, y=labels)

    tpr, fpr = make_tpr_fpr(probs, labels)

    def f(threshold):
        plot2(probs, labels, tpr, fpr, threshold, area=True)

    slider_value6 = st.slider(
        "Select a threshold",
        min_value=0.0,
        max_value=1.0,
        step=0.1,
        format="%.1f",
        key="selecting_optimal_threshold",
    )
    f(slider_value6)


st.set_page_config(page_title="Optimal Threshold")
st.sidebar.header("Optimal Threshold")
main()
