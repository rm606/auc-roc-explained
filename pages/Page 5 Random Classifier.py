import streamlit as st
import numpy as np
from pages.definition import DummyClassifier, make_tpr_fpr, plot2


def main():
    st.write(
        """
        # A random Classifier

        Now if you are observant enough, you may have seen a straigh,
        linear line in ROC curve graph. In legend, it is shown as random
        classifier, what does it mean? Now consider the following example.
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
    clf = DummyClassifier(pos_label_args=(0.5, 0.1), neg_label_args=(0.5, 0.1))
    probs = clf.predict(some_rand_input, y=labels)

    tpr, fpr = make_tpr_fpr(probs, labels)

    plot2(probs, labels, tpr, fpr, threshold=0.5, area=True)

    st.write(
        """
        Take note of the overlap in the distribution of model
        outputs between positive and negative labels.

        Try to identify a threshold that can effectively separate
        the positive and negative labels. I challenge you to find one.
        Notice how the True Positive Rate (TPR) and False Positive Rate (FPR)
        remain equal for all thresholds.
    """
    )


st.set_page_config(page_title="Random Classifier")
main()
