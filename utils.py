import streamlit as st
import numpy as np
import plotly.graph_objects as go
from typing import Tuple
from plotly.subplots import make_subplots


class DummyClassifier:
    def __init__(
        self, pos_label_args: Tuple[float, float], neg_label_args: Tuple[float, float]
    ):
        self.pos_label_mean, self.pos_label_std = pos_label_args
        self.neg_label_mean, self.neg_label_std = neg_label_args

    def predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        assert X.ndim == 2
        m, n = X.shape
        np.random.seed(40)
        postives = (
            np.random.randn(y[y == 1].shape[0]) * self.pos_label_std
            + self.pos_label_mean
        )
        np.random.seed(40)
        negatives = (
            np.random.randn(y[y == 0].shape[0]) * self.neg_label_std
            + self.neg_label_mean
        )
        return np.concatenate([postives, negatives], axis=-1)


def plot1(
    probabilities: np.ndarray, labels: np.ndarray, threshold: float
) -> np.ndarray:
    confusion_matrix = [[0, 0], [0, 0]]
    for actual, predicted in zip(labels, (probabilities >= threshold).astype(np.uint8)):
        confusion_matrix[actual][predicted] += 1

    fig = make_subplots(rows=2, cols=1)
    fig.add_trace(
        go.Histogram(x=probabilities[labels == 1], name="Positive Labels", nbinsx=100),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Histogram(x=probabilities[labels == 0], name="Negatives Labels", nbinsx=100),
        row=1,
        col=1,
    )
    fig["layout"]["xaxis"]["title"] = "Model Output Probabilities"
    fig["layout"]["yaxis"]["title"] = "Num Counts"
    # Add vertical line at the threshold
    fig.add_shape(
        type="line",
        x0=threshold,
        x1=threshold,
        y0=0,
        y1=len(labels) / 10,
        line=dict(color="lightgray", dash="dot"),
    )
    text = [["TN: ", "FP: "], ["FN: ", "TP: "]]
    for i in range(2):
        for j in range(2):
            text[i][j] += str(confusion_matrix[i][j])
    # Create the confusion matrix plot
    fig.add_trace(
        go.Heatmap(
            z=confusion_matrix,
            x=["Predicted negative", "Predicted positive"],
            y=["Actual negative", "Actual positive"],
            colorscale="gray",
            showscale=False,
            reversescale=True,
            text=text,
            hoverinfo="text",
            texttemplate="%{text}",
            textfont={"size": 20},
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        title_text=f"Histogram Plot and Confusion matrix for thershold = {threshold}",
        width=800,
        height=800,
    )
    st.plotly_chart(fig)
    return confusion_matrix


def make_tpr_fpr(
    probabilities: np.ndarray, labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Data for the AUC-ROC curve
    fpr = []
    tpr = []
    for t in range(11):
        t = t / 10
        confusion_matrix = [[0, 0], [0, 0]]
        for actual, predicted in zip(labels, (probabilities >= t).astype(np.uint8)):
            confusion_matrix[actual][predicted] += 1
        fn, tp = confusion_matrix[1]
        tn, fp = confusion_matrix[0]
        tpr.append(tp / (tp + fn + 1e-20))
        fpr.append(fp / (fp + tn + 1e-20))
    return np.array(tpr), np.array(fpr)


def plot2(
    probabilities: np.ndarray,
    labels: np.ndarray,
    tpr: np.ndarray,
    fpr: np.ndarray,
    threshold: float,
    area: bool = False,
) -> np.ndarray:
    fig = make_subplots(rows=2, cols=1)
    fig.add_trace(
        go.Histogram(x=probabilities[labels == 1], name="Positive labels", nbinsx=100),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Histogram(x=probabilities[labels == 0], name="Negative labels", nbinsx=100),
        row=1,
        col=1,
    )
    fig["layout"]["xaxis"]["title"] = "Model Output Probabilities"
    fig["layout"]["yaxis"]["title"] = "Num Counts"
    # Add vertical line at the threshold
    fig.add_shape(
        type="line",
        x0=threshold,
        x1=threshold,
        y0=0,
        y1=len(labels) / 10,
        line=dict(color="lightgray", dash="dot"),
        name="threshold",
    )
    # Roc curve
    if area:
        fig.add_trace(
            go.Scatter(
                x=fpr, y=tpr, mode="lines", fill="tozeroy", name="AUC ROC Curve"
            ),
            row=2,
            col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC Curve"), row=2, col=1
        )

    n = len(tpr)
    rand_series = np.arange(n) / 10
    fig.add_trace(
        go.Scatter(
            x=rand_series,
            y=rand_series,
            mode="lines",
            name="Random Classifier",
            line=dict(color="gray", dash="dot"),
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title="Prob distribution and ROC curve",
        xaxis2=dict(title="FPR"),
        yaxis2=dict(title="TPR"),
        xaxis2_range=[0, 1],
        yaxis2_range=[0, 1],
        showlegend=True,
        height=800,
    )

    fig.add_annotation(
        x=fpr[int(threshold * 10)],
        y=tpr[int(threshold * 10)],
        ax=0,
        ay=-40,
        text=f"threshold={threshold}",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        row=2,
        col=1,
    )
    st.plotly_chart(fig)
