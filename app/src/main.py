"""Steamlit app for Alivio project."""

import streamlit as st
import requests
import h3
import pandas as pd
import pydeck as pdk
import re

from project_description import home_page
from viz_hurricane_data import select_and_display_hurricane
from live_model_prediction import live_prediction
from xview2_gallery import display_xview2_gallery


def main():
    # Create tabs
    tabs: list[str] = [
        "Home",
        "Model Prediction",
        "Vulnerability Map",
        "Xview2 Gallery",
        "Disaster Simulation",
    ]
    tab_captions: list[str] = [
        "Project Page & Description 🏠",
        "Live Building Damage Classifier  🔮",
        "Vulnerable Populations through Demographics & Building Damage 🗺",
        "Satellite Image Browser for Inputs 🎞",
        "[Coming Soon] Simulate Impacts of Hypothetical Disasters 🌪"
    ]
    selected_tab = st.sidebar.radio("Select a tab", tabs, captions=tab_captions)
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    [**Alivio**](https://www.ischool.berkeley.edu/projects/2024/alivio) is an open-source project developed at the [UC Berkeley School of Information](https://ischool.berkeley.edu) available under the 
    [MIT License](https://opensource.org/license/mit). Data used in this project is subject to individual data-source licenses.
    """)
    st.sidebar.image("https://www.ischool.berkeley.edu/sites/all/themes/custom/i_school/images/logos/berkeleyischool-logo-blue.svg", width=200)

    if selected_tab == "Home":
        home_page()

    elif selected_tab == "Model Prediction":
        live_prediction()

    elif selected_tab == "Vulnerability Map":
        select_and_display_hurricane()

    elif selected_tab == "Xview2 Gallery":
        display_xview2_gallery()

    elif selected_tab == "Disaster Simulation":
        st.warning("This feature is coming soon! 🚧")


if __name__ == "__main__":
    main()
