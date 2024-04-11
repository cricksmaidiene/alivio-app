"""Steamlit app for Alivio project."""

import streamlit as st
import requests
import h3
import pandas as pd
import pydeck as pdk
import re

from project_description import home_page
from viz_hurricane_data import select_and_display_hurricane

def main():
    # Create tabs
    tabs: list[str] = ["Home 🏠", "Live Building Damage Prediction 🔮", "Vulnerability Map 🗺"]
    selected_tab = st.sidebar.selectbox("Select a tab", tabs)

    if selected_tab == "Home 🏠":
        home_page()
    
    elif selected_tab == "Vulnerability Map 🗺":
        select_and_display_hurricane()


if __name__ == "__main__":
    main()
