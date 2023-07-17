import streamlit as st
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from utils import *


def main():
    st.write(
        """
    # Influence of Class Imbalance

    In numerous scenarios, we encounter imbalanced datasets
    where the number of positive samples is significantly
    smaller than the number of negative samples.

    Our objective is to assess the reliability of the metric
    used in conjunction with a classifier and determine if it
    accurately represents the properties of the model output.
    In essence, we aim to determine the metric's dependability.

    Let's illustrate this with an example.
    Suppose we have a dataset consisting of 1000 negative samples
    and only 100 positive samples.
    Now, we have two classifiers at our disposal—one is good,
    while the other is bad. We will evaluate these classifiers based on two metrics:

    1. Accuracy
    2. AUC ROC

    By comparing the performance of these classifiers using
    these metrics, we can determine which one is more meaningful
    and appropriate for our specific needs.
    """
    )
    m, n = 1100, 100
    some_rand_input = np.random.rand(m, n)
    labels = np.hstack([np.ones(100).astype(np.uint8), np.zeros(1000).astype(np.uint8)])
    clf = DummyClassifier(pos_label_args=(0.70, 0.1), neg_label_args=(0.3, 0.1))
    probs = clf.predict(some_rand_input, y=labels)

    tpr, fpr = make_tpr_fpr(probs, labels)

    st.write(
        "Let's see how does model output distribution looks for a good classifier."
    )
    plot2(probs, labels, tpr, fpr, threshold=0.5, area=True)

    st.write(
        "Look how we have some threshold, that can do a good job \
        of separating these to labels"
    )

    st.write("Compute Accuracy and AUC-ROC")
    auc_roc1 = roc_auc_score(labels, probs)

    max_acc1 = (0, 0)
    for t in range(0, 11, 1):
        t = t / 10
        acc = accuracy_score(labels, (probs >= t).astype(np.uint8))
        if max_acc1[0] < acc:
            max_acc1 = (acc, t)

    st.write(f"Area under the ROC curve is: {auc_roc1}")
    st.write(f"Maximum Accuracy: {max_acc1[0]} at theshold: {max_acc1[1]}")
    st.markdown("---")

    st.write("Let's See how it is for a bad classifier")
    clf = DummyClassifier(pos_label_args=(0.60, 0.15), neg_label_args=(0.4, 0.15))
    probs = clf.predict(some_rand_input, y=labels)

    tpr, fpr = make_tpr_fpr(probs, labels)
    plot2(probs, labels, tpr, fpr, threshold=0.5, area=True)

    auc_roc2 = roc_auc_score(labels, probs)

    max_acc2 = (0, 0)
    for t in range(0, 11, 1):
        t = t / 10
        acc = accuracy_score(labels, (probs >= t).astype(np.uint8))
        if max_acc2[0] < acc:
            max_acc2 = (acc, t)

    st.write(f"Area under the ROC curve is: {auc_roc2}")
    st.write(f"Maximum Accuracy: {max_acc2[0]} at theshold: {max_acc2[1]}")

    diff_acc = (max_acc1[0] - max_acc2[0]) / max_acc1[0] * 100
    diff_roc = (auc_roc1 - auc_roc2) / auc_roc1 * 100
    st.write(
        f"<p color='red'>The AUC decreased by {diff_roc}% </p>", unsafe_allow_html=True
    )
    st.write(
        f"<p color='red'> The ACC decreased by {diff_acc}% </p>", unsafe_allow_html=True
    )
    st.write(
        "We see that we got a 17% decrease in auc roc. \
        Whereas we got 7% decrease in acc. \
        Hence we can say auc captures the model output reliably.\
        Whenever there is a class imbalance, the auc roc metric will \
        penalize more on wrong classification of minor class than a simple\
        acc metric."
    )

    st.write("## What if we have more positive samples than negative")
    m, n = 1100, 100
    some_rand_input = np.random.rand(m, n)
    labels = np.hstack([np.ones(1000).astype(np.uint8), np.zeros(100).astype(np.uint8)])
    clf = DummyClassifier(pos_label_args=(0.60, 0.15), neg_label_args=(0.4, 0.15))
    probs = clf.predict(some_rand_input, y=labels)

    tpr, fpr = make_tpr_fpr(probs, labels)
    plot2(probs, labels, tpr, fpr, threshold=0.5, area=True)
    auc_roc = roc_auc_score(labels, probs)

    max_acc = (0, 0)
    for t in range(0, 11, 1):
        t = t / 10
        acc = accuracy_score(labels, (probs >= t).astype(np.uint8))
        if max_acc[0] < acc:
            max_acc = (acc, t)

    st.write(f"Area under the ROC curve is: {auc_roc}")
    st.write(f"Maximum Accuracy: {max_acc[0]} at theshold: {max_acc[1]}")


st.set_page_config(page_title="Class Imbalance")
st.sidebar.header("Class Imbalance")
main()
