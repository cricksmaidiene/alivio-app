"""The main page components of the Streamlit app."""

import streamlit as st

def home_page():
    with open("app/docs/home.md") as f:
        home_markdown: str = f.read()
    st.markdown(home_markdown)
    st.markdown("---")