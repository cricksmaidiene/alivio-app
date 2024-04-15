import streamlit as st
import pandas as pd
import boto3
import json


@st.cache_data
def _download_gallery_data() -> pd.DataFrame:
    with st.spinner("Loading Gallery Metadata 📽"):
        df = pd.read_parquet(
            "s3://alivio/datasets/xview2/processed/app_gallery_dataset.parquet"
        )
        df = df[
            [
                "disaster",
                "dataset",
                "image_id",
                "building_count",
                "no_damage_buildings",
                "image_filepath",
                "json_filepath",
            ]
        ]
        return df


def _read_s3_data_as_bytes(s3_path: str) -> bytes | None:
    # Create an S3 client
    bucket_name = s3_path.split("/")[2]
    key = "/".join(s3_path.split("/")[3:])
    s3 = boto3.client("s3")

    # Get the object from the bucket
    try:
        response = s3.get_object(Bucket=bucket_name, Key=key)
        image_bytes = response["Body"].read()
        return image_bytes
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def display_xview2_gallery():
    st.markdown("## Xview2 Building Damage Satellite Image Gallery 🎞")
    gal_info = st.empty()
    gal_info.info(
        "Gallery data is loaded at start of session. It will persist until session ends."
    )
    df = _download_gallery_data()
    gal_info.empty()
    st.markdown(
        """
    This section contains a full list of hurricane images and labels that our Vision Transformer was trained on.
    The data contains train, hold and test set images. You can filter for a specific image or use `Magic` to provide a random one.
    
    **Use `image_id` to preview the image and json file.**         
    """
    )
    st.write(
        "If you have the Image ID already, enter it below and ignore other filters. Clear Image ID to use filters."
    )
    image_id = st.text_input("Enter Image ID", "hurricane-matthew_00000302")
    st.markdown("---")
    if not image_id:
        st.write("Or use the filters below to find an image")
        hurricane_selection = st.selectbox("Select Hurricane", df["disaster"].unique())
        dataset_selection = st.selectbox("Select Dataset", df["dataset"].unique())
        filter_building_count = st.slider(
            "Filter by Building Count Greater Than:", 0, 100, 0
        )

    if image_id:
        df = df[df["image_id"] == image_id]
        if df.empty:
            st.warning(f"No image found for image_id: {image_id}")
            return

    if not image_id and any(
        [hurricane_selection, dataset_selection, filter_building_count]
    ):
        if hurricane_selection:
            df = df[(df["disaster"] == hurricane_selection)]
        if dataset_selection:
            df = df[(df["dataset"] == dataset_selection)]
        if filter_building_count:
            df = df[(df["building_count"] > filter_building_count)]

    st.dataframe(df.drop(columns=["image_filepath", "json_filepath"]))

    if len(df) == 1:
        view_data = st.selectbox(
            "Select File to View & Download", ["None", "Image", "JSON"]
        )
        if view_data != "None":
            for _, row in df.iterrows():
                if view_data == "Image":
                    st.markdown(f"`Image ID`: {row['image_id']}")

                    with st.spinner("Loading Image 🖼"):
                        img_data = _read_s3_data_as_bytes(row["image_filepath"])

                    st.markdown("##### Image Preview 🖼")
                    st.download_button(
                        label="Download Image",
                        data=img_data,
                        file_name=f"{image_id}_post_disaster.png",
                        mime="image/png",
                    )
                    st.image(img_data, use_column_width=True)

                if view_data == "JSON":

                    with st.spinner("Loading JSON File 📄"):
                        json_bytes = _read_s3_data_as_bytes(row["json_filepath"])
                        json_data = json.loads(json_bytes.decode("utf-8"))

                    st.markdown("##### JSON File Preview 📄")
                    st.download_button(
                        label="Download JSON",
                        data=json_bytes,
                        file_name=f"{image_id}_post_disaster.json",
                        mime="application/json",
                    )

                    st.json(json_data)

    elif len(df) != 1:
        st.warning(
            "To View / Download: Please select a single image by narrowing data to 1 row using filters or Image ID."
        )
        return
