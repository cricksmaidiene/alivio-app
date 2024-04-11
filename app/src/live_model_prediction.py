import streamlit as st
import requests

def upload_files():
    st.write("### Upload Disaster Image ")
    polygon = st.file_uploader("Map polygons", type=["json", "csv"])
    post_image = st.file_uploader("Upload post-disaster image", type=["jpg", "png"])

    if st.button("Assess Damage"):
        if polygon is not None and post_image is not None:
            files = {"polygon": polygon, "post_image": post_image}
            response = requests.post("http://localhost:8000/assess_damage", files=files)
            result = response.json()
            st.write("Assessment Result:", result["result"])
        else:
            st.write("Please upload both map polygon and post image.")