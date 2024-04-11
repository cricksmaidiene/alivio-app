"""The main page components of the Streamlit app."""

import streamlit as st

def home_page():
    with open("app/docs/home.md") as f:
        home_markdown: str = f.read()
    st.markdown(home_markdown)
    st.markdown("---")
    st.image("https://www.ischool.berkeley.edu/sites/all/themes/custom/i_school/images/logos/berkeleyischool-logo-blue.svg", width=200)