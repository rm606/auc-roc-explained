import streamlit as st
import numpy as np
from pages.definition import DummyClassifier, make_tpr_fpr, plot2


def main():
    st.write(
        """
        # Distinguishing Between a Good Classifier and a Bad Classifier

        Now let's differentiate between a good classifier and a bad classifier.
        In the following cell, you will find four adjustable parameters:
        means and standard deviations for positive labels and negative labels
        in the model's output distribution.

        By tuning these parameters, aim to achieve the best AUC.
        This exercise will provide you with insights into the type of
        probability distribution required from a classifier.

        Well, this is just a demonstration. In reality you don't have direct
        access to these distribution parameters,
        but you can make different classifier that does the best job for roc curve.
        """
    )

    positive_mean = st.slider(
        "Positive Mean",
        min_value=0.0,
        max_value=1.0,
        step=0.1,
        format="%.1f",
        key="positive_mean",
        value=0.7,
    )
    positive_std = st.slider(
        "Positive Std.",
        min_value=0.0,
        max_value=0.5,
        step=0.01,
        format="%.2f",
        key="positive_std",
        value=0.1,
    )
    negative_mean = st.slider(
        "Negative Mean",
        min_value=0.0,
        max_value=1.0,
        step=0.1,
        format="%.1f",
        key="negative_mean",
        value=0.3,
    )
    negative_std = st.slider(
        "Negative Std.",
        min_value=0.0,
        max_value=0.5,
        step=0.01,
        format="%.2f",
        key="negative_std",
        value=0.1,
    )

    m, n = 1000, 100
    some_rand_input = np.random.rand(m, n)
    labels = np.hstack(
        [
            np.ones(int(np.ceil(m / 2))).astype(np.uint8),
            np.zeros(m // 2).astype(np.uint8),
        ]
    )

    def f(
        pos_label_mean=0.5,
        pos_label_std=0.1,
        neg_label_mean=0.5,
        neg_label_std=0.1,
    ):
        threshold = 0.5
        clf = DummyClassifier(
            pos_label_args=(pos_label_mean, pos_label_std),
            neg_label_args=(neg_label_mean, neg_label_std),
        )
        probs = clf.predict(some_rand_input, y=labels)
        tpr, fpr = make_tpr_fpr(probs, labels)
        plot2(probs, labels, tpr, fpr, threshold, area=True)

    f(positive_mean, positive_std, negative_mean, negative_std)

    st.write(
        """
        Are you able to find a good and bad classifier? If not well here is one.
    """
    )
    st.image("assets/p3.png")

    st.write(
        """
        ## One more question 💡"

        Can we have roc curve worse than random classifier?
        Can you guess the distribution?

        I hope you guessed it right"""
    )
    st.image("assets/p4.png")

    st.write(
        """
        When your model output distribution mean for negative and
        positive labels are flipped. When your classifier cannot even
        learn to separate the two classes. 

        You may try these in above visualization.
        Make the mean for positive labels < negative labels
    """
    )


st.set_page_config(page_title="Good Bad Classifier")
main()
