import streamlit as st


def main():
    st.write(
        """
        # One last bit!
        If you have prepared for an interview,
        you might have come across the statement explaining AUC ROC as follows:
        
        > AUC, which stands for Area Under the ROC Curve,
        represents the probability that a classifier will rank
        a randomly selected positive instance higher than a randomly
        selected negative instance.

        However, this statement can be quite complex to understand.
        So, let's break it down into simpler terms.

        To begin, carefully observe the two images provided 
        and answer the following questions:"""
    )

    st.image("assets/p5.png")

    st.write(
        """
        1. Which image (left or right) has a higher AUC
        or is considered a better classifier?

        Answer: The right one, as it better separates
        the positive and negative labels.

        2. In which image is there a higher chance of a positive sample
        being positioned to the right of the optimal threshold? Remember, optimal
        threshold is one which separates positive and negative sample with high TPR
        and low FPR.

        Answer: The right one.

        Okay, Re read the answers above and once you have done so. Go ahead.

        Let's say I have a very good classifier. This means, that ground truth 
        positives will be correctly classified. There is only one way to do this,
        if model gives high probability (which is the model output) to positive examples
        and low probility to negative examples.
        If this is the case then, given a randomly chosen positive, and 
        negative example, the positive example will have high model output 
        than a negative example.

        So let's say we rank examples with model output, 
        then positives will be ranked higher than the negatives.
        """
    )


st.set_page_config(page_title="One Last Bit")
st.sidebar.header("One Last Bit")
main()
