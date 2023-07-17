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

        2. In which image is there a higher probability of a positive sample
        being positioned to the right of the optimal threshold?

        Answer: The right one. The threshold aims to maximize
        the number of positive samples on the right side.

        Combining these answers, we can conclude that positive samples
        tend to have higher model outputs. 
        Consequently, when ranking based on the model output,
        randomly chosen positive samples are more likely to be ranked higher
        than randomly chosen negative samples.
        """
    )


st.set_page_config(page_title="One Last Bit")
main()
