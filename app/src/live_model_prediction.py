"""A module for live model prediction using Pytorch"""

import warnings

warnings.filterwarnings("ignore")

import json
import math
import os
import shutil
from datetime import date
from sys import stdout
from typing import Literal

import boto3
import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import shapely.wkt
import streamlit as st
import torch
import torch.nn as nn
from loguru import logger
from PIL import Image
from shapely import wkt
from torch import tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torcheval.metrics.functional import (
    multiclass_accuracy,
    multiclass_f1_score,
    multiclass_precision,
    multiclass_recall,
)
from torchvision import datasets, transforms

log_handler_id = None


def set_logger_level(level: str):
    """Sets the minimum level for the logger to send a message to the sink"""
    global log_handler_id

    # Remove existing logger handler
    logger.remove(log_handler_id)

    # Change logger level
    log_handler_id = logger.add(sink=stdout, level=level)


set_logger_level("INFO")

ROOT_DIR: str = os.getcwd()
INFERENCE_ROOT: str = os.path.join(
    ROOT_DIR, "data", "xview_building_damage", "inference"
)
UPLOADS_IMG: str = os.path.join(INFERENCE_ROOT, "upload", "img")
UPLOADS_JSON: str = os.path.join(INFERENCE_ROOT, "upload", "json")
POST_PROCESSED: str = os.path.join(INFERENCE_ROOT, "postprocesssed")
PRE_PROCESSED: str = os.path.join(INFERENCE_ROOT, "preprocesssed")

logger.debug(f"{ROOT_DIR=}")
logger.debug(f"{INFERENCE_ROOT=}")
logger.debug(f"{UPLOADS_IMG=}")
logger.debug(f"{UPLOADS_JSON=}")
logger.debug(f"{POST_PROCESSED=}")
logger.debug(f"{PRE_PROCESSED=}")

os.makedirs(UPLOADS_IMG, exist_ok=True)
os.makedirs(UPLOADS_JSON, exist_ok=True)
os.makedirs(POST_PROCESSED, exist_ok=True)
os.makedirs(PRE_PROCESSED, exist_ok=True)


def _get_uploaded_files() -> tuple:
    """Get the streamlit uploaded files in the upload directory"""
    uploaded_imgs = [f"{UPLOADS_IMG}/{f}" for f in os.listdir(UPLOADS_IMG)]
    uploaded_jsons = [f"{UPLOADS_JSON}/{f}" for f in os.listdir(UPLOADS_JSON)]
    return uploaded_imgs, uploaded_jsons


def create_inference_csv_for_upload():
    """Create a csv file for the buildings contained in the images"""
    label_json_data: list[dict] = []

    def read_and_store_label_json(label_json_path: str):
        """A thread-safe function that reads a json as a dictionary and writes to a global list"""
        with open(label_json_path) as f:
            label_json_data.append(json.load(f))

    uploaded_imgs, uploaded_jsons = _get_uploaded_files()
    read_and_store_label_json(uploaded_jsons[0])

    label_df_original: pd.DataFrame = pd.json_normalize(pd.Series(label_json_data))

    lbl_df: pd.DataFrame = label_df_original.copy()
    CHALLENGE_TYPE: Literal["train", "test", "hold"] = "test"

    def json_df_to_csv(label_df: pd.DataFrame):
        label_df_lng_lat: pd.DataFrame = (
            label_df.drop(columns=["features.xy", "features.lng_lat"])
            .join(label_df["features.lng_lat"].explode())
            .reset_index(drop=True)
        )

        label_df_features: pd.DataFrame = (
            label_df.drop(columns=["features.xy", "features.lng_lat"])
            .join(label_df["features.xy"].explode())
            .reset_index(drop=True)
        )

        lng_lat_normalized: pd.DataFrame = pd.json_normalize(
            label_df_lng_lat["features.lng_lat"]
        ).rename(
            columns={
                "wkt": "map_polygon",
                "properties.feature_type": "map_feature_type",
                "properties.subtype": "map_damage",
                "properties.uid": "building_id",
            }
        )

        features_normalized: pd.DataFrame = pd.json_normalize(
            label_df_features["features.xy"]
        ).rename(
            columns={
                "wkt": "image_polygon",
                "properties.feature_type": "image_feature_type",
                "properties.subtype": "image_damage",
                "properties.uid": "building_id",
            }
        )

        label_df_lng_lat_normalized = label_df_lng_lat.drop(
            columns=["features.lng_lat"]
        ).join(lng_lat_normalized)

        label_df_features_normalized = label_df_features.drop(
            columns=["features.xy"]
        ).join(features_normalized)

        label_df_final: pd.DataFrame = label_df_lng_lat_normalized.merge(
            label_df_features_normalized[
                [
                    "metadata.id",
                    "image_polygon",
                    "image_feature_type",
                    "image_damage",
                    "building_id",
                ]
            ],
            "left",
            ["metadata.id", "building_id"],
        )

        label_df_final = (
            label_df_final.rename(
                columns={
                    c: c.replace("metadata.", "")
                    for c in label_df_final.columns
                    if c.startswith("metadata.")
                }
            )
            .drop(
                columns=[
                    "map_feature_type",
                    "map_damage",
                ]
            )
            .rename(
                columns={
                    "image_feature_type": "feature_type",
                    "image_damage": "damage",
                }
            )
        )

        label_df_final["dataset"] = CHALLENGE_TYPE
        label_df_final["capture_date"] = pd.to_datetime(label_df_final["capture_date"])

        label_df_final["image_id"] = (
            label_df_final["img_name"]
            .dropna()
            .apply(lambda cell: "_".join(cell.split("_")[0:2]))
        )
        label_df_final["is_pre_image"] = (
            label_df_final["img_name"]
            .dropna()
            .apply(lambda cell: "_pre_disaster" in cell)
        )
        label_df_final["is_post_image"] = (
            label_df_final["img_name"]
            .dropna()
            .apply(lambda cell: "_post_disaster" in cell)
        )

        label_df_final.to_parquet(f"{CHALLENGE_TYPE}.parquet")

        concat_list: list[pd.DataFrame] = [
            pd.read_parquet(pq_file)
            for pq_file in os.listdir()
            if pq_file.endswith(".parquet")
        ]

        df = pd.concat(concat_list).reset_index(drop=True)
        df.to_parquet(os.path.join(POST_PROCESSED, "inference_data.parquet"))

        df.to_csv(os.path.join(POST_PROCESSED, "inference_data.csv"), index=False)

    json_df_to_csv(lbl_df)


def preprocess_image(image_path: str) -> np.ndarray:
    def _get_df_with_class_numeric_labels(df_name: pd.DataFrame) -> pd.DataFrame:
        df_name["damage_class"] = df_name["damage"]
        keys = list(df_name["damage_class"].value_counts().keys())
        df_name["damage_class"] = df_name["damage_class"].apply(keys.index)
        df_name["damage_class"].value_counts()
        return df_name

    def _get_metadata() -> pd.DataFrame:
        infer_csv = pd.read_csv(os.path.join(POST_PROCESSED, "inference_data.csv"))
        data = infer_csv[infer_csv["image_polygon"].notna()]
        df_disaster = data[data["damage"] != "un-classified"]
        df_disaster["mask_file_names"] = (
            df_disaster["img_name"].str.replace(".png", "_")
            + df_disaster["building_id"]
            + ".png"
        )
        df_disaster_class_labels = _get_df_with_class_numeric_labels(df_disaster)
        return df_disaster_class_labels

    def _polygons_mask(polygons, im_size: tuple = (1024, 1024)) -> np.ndarray:
        """Create a mask from polygons."""
        img_mask = np.zeros(im_size, np.uint8)

        if not polygons:
            return img_mask

        int_coords = lambda x: np.array(x).round().astype(np.int32)  # noqa

        exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
        interiors = [
            int_coords(pi.coords) for poly in polygons for pi in poly.interiors
        ]

        cv2.fillPoly(img_mask, exteriors, 1)
        cv2.fillPoly(img_mask, interiors, 0)

        return img_mask

    def _create_image_mask_overall(
        root_dir: str, meta_df: pd.DataFrame, im_size: tuple = (1024, 1024)
    ):
        input_dir: str = os.path.join(root_dir, "upload")
        dest_dir: str = os.path.join(root_dir, str(date.today()))
        img_input: str = os.path.join(input_dir, "img")

        if os.path.exists(dest_dir):
            logger.warning(f"Removing the dir with name: {dest_dir}")
            os.system("rm -rf " + dest_dir)

        logger.debug(f"creating empty dir with name {dest_dir}")
        os.makedirs(dest_dir)

        img_overlay = os.path.join(dest_dir, "img_mask_overlay")
        if os.path.exists(img_overlay):
            logger.warning(f"Removing the dir with name: {img_overlay}")
            os.system("rm -rf " + img_overlay)

        logger.debug(f"creating empty dir with name {img_overlay}")
        os.makedirs(img_overlay)

        df = meta_df[meta_df["is_post_image"] == True]

        logger.info("Starting : Mask overlay")
        for idx, file_name in enumerate(df["mask_file_names"]):
            image = cv2.imread(os.path.join(img_input, df.iloc[idx]["img_name"]))
            mask = np.zeros(image.shape[:2], dtype="uint8")
            _mask = _polygons_mask([shapely.wkt.loads(df.iloc[idx]["image_polygon"])])
            masked = cv2.bitwise_and(image, image, mask=_mask)
            plt.imsave(os.path.join(img_overlay, file_name), masked)

        logger.info("Ending : Mask overlay")

    def _get_bounds_tp(image_wkt: str) -> tuple[float]:
        bounds = wkt.loads(image_wkt).bounds
        return (bounds[0], bounds[1], bounds[2], bounds[3])  # type: ignore

    def _crop_save_masked_images(
        root_dir: str,
        meta_df: pd.DataFrame,
        crop_output_dir_name: str = "img_mask_overlay_crops",
    ):
        input_dir = os.path.join(root_dir, str(date.today()))
        img_crop_overlay = os.path.join(input_dir, crop_output_dir_name)

        if os.path.exists(img_crop_overlay):
            logger.warning(f"Removing the dir with name: {img_crop_overlay}")

        os.system("rm -rf " + img_crop_overlay)

        logger.debug(f"creating empty dir with name {img_crop_overlay}")
        os.makedirs(img_crop_overlay)

        logger.info("Starting Cropping the images")

        for idx, file_name in enumerate(meta_df["mask_file_names"]):
            img = Image.open(os.path.join(input_dir, "img_mask_overlay", file_name))
            minx, miny, maxx, maxy = _get_bounds_tp(meta_df.iloc[idx]["image_polygon"])
            cropped_img = img.crop((minx - 5, miny - 5, maxx + 5, maxy + 5))
            cropped_img.save(os.path.join(img_crop_overlay, file_name))

        logger.info("Finished Cropping the images")

    def _sort_masks_by_class(
        top_dir: str, meta_df: pd.DataFrame, cls_path="img_mask_ov_crop_class"
    ) -> None:
        input_dir = os.path.join(top_dir, str(date.today()))
        disas_post_mask = os.path.join(input_dir, "img_mask_overlay_crops")  # source

        logger.debug(
            f"Source root : {disas_post_mask}",
        )
        disas_class_path = os.path.join(input_dir, "img_mask_ov_crop_class")

        if os.path.exists(disas_class_path):
            logger.warning(f"Removing the dir with name: {disas_class_path}")

        os.system("rm -rf " + disas_class_path)

        logger.debug(f"creating empty dir with name {disas_class_path}")
        os.makedirs(disas_class_path)

        logger.debug(f"Destination root : {disas_class_path}")

        df = meta_df[meta_df["is_post_image"] == True]

        logger.info("Started moving the mask files to class folder ")

        for idx, file_name in enumerate(df["mask_file_names"]):
            source = os.path.join(disas_post_mask, df.iloc[idx]["mask_file_names"])
            destination = os.path.join(disas_class_path, df.iloc[idx]["damage"])
            if os.path.exists(destination):
                pass
            else:
                logger.debug(f"Creating dir for {df.iloc[idx]['damage']}")
                os.makedirs(destination)

            if os.path.exists(source):
                shutil.copy(source, destination)
        logger.info("Finished moving the mask files to class folder ")

    df = _get_metadata()
    _create_image_mask_overall(INFERENCE_ROOT, df)
    _crop_save_masked_images(
        root_dir=INFERENCE_ROOT,
        meta_df=df,
        crop_output_dir_name="img_mask_overlay_crops",
    )
    _sort_masks_by_class(INFERENCE_ROOT, df)


def load_local_vit_model(model_path: str = "vit2024-04-11.pkl"):
    vit_model = joblib.load(os.path.join(model_path))

    for layer_name, p in vit_model.named_parameters():
        logger.debug(
            "Layer Name: {}, Frozen: {}".format(layer_name, not p.requires_grad)
        )

    return vit_model


def make_prediction():
    def _calculate_weight_decay(batch, train_data_len, nepoches, lambda_norm):
        return lambda_norm * math.sqrt((batch / (train_data_len * nepoches)))

    test_transform = transforms.Compose(
        [
            # Resize the images to 64x64
            transforms.Resize(size=(224, 224)),
            # Flip the images randomly on the horizontal
            transforms.RandomHorizontalFlip(p=0.5),
            # Turn the image into a torch.Tensor
            transforms.ToTensor(),  # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0
        ]
    )

    batch_size = 128
    inference_all_dataset = datasets.ImageFolder(
        os.path.join(INFERENCE_ROOT, str(date.today()), "img_mask_ov_crop_class"),
        transform=test_transform,
    )
    inference_loader = DataLoader(inference_all_dataset, batch_size=batch_size)

    class_names = inference_all_dataset.classes
    logger.debug(f"class_names {class_names}")

    class_dict = inference_all_dataset.class_to_idx
    logger.debug(f"class_dict {class_dict}")

    def get_class_weights(labels):
        class_counts = np.bincount(labels)
        num_classes = len(class_counts)
        total_samples = len(labels)

        class_weights = []
        for count in class_counts:
            weight = 1 / (count / total_samples)
            class_weights.append(weight)

        return class_weights

    def get_metrics(preds_list, target_list, num_classes=4) -> tuple:
        pred_ts = tensor(preds_list)
        target_ts = tensor(target_list)

        accuracy = multiclass_accuracy(pred_ts, target_ts, num_classes=4)

        f1_score = multiclass_f1_score(
            pred_ts, target_ts, num_classes=4, average="weighted"
        )

        precision = multiclass_precision(
            pred_ts, target_ts, num_classes=4, average="weighted"
        )
        recall = multiclass_recall(
            pred_ts, target_ts, num_classes=4, average="weighted"
        )
        f1_score_class_wise = multiclass_f1_score(
            pred_ts, target_ts, num_classes=4, average=None
        )

        logger.info(f"Accuracy : {accuracy}")
        logger.info(f"F1-score : {f1_score}")
        logger.info(f"F1-score Classwise : {f1_score_class_wise}")
        logger.info(f"Precision : {precision}")
        logger.info(f"Recall : {recall}")
        return accuracy, f1_score, precision, recall

    def accuracy_per_class(class_correct, class_total, class_names, accuracy):
        n_class = len(class_names)

        class_accuracy = class_correct / class_total

        logger.info("Test Accuracy of Classes")

        for c in range(n_class):
            logger.info(
                "{}\t: {}% \t ({}/{})".format(
                    class_names[c],
                    int(class_accuracy[c] * 100),
                    int(class_correct[c]),
                    int(class_total[c]),
                )
            )

        logger.info(
            "Test Accuracy of Dataset: \t {}% \t ({}/{})".format(
                int(accuracy), int(np.sum(class_correct)), int(np.sum(class_total))
            )
        )

    def model_eval(model, test_loader, class_names) -> tuple:
        criterion = nn.CrossEntropyLoss()
        model.eval()

        preds_list, target_list, output_list = [], [], []
        test_loss, accuracy = 0.0, 0
        n_class = len(class_names)
        class_correct, class_total = np.zeros(n_class), np.zeros(n_class)
        # move model back to cpu
        model = model.to("cpu")

        for images, targets in test_loader:
            outputs = model(images)
            # calculate loss
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            # test_focal_loss += fl.item()
            preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)

            target_list.extend(targets)
            output_list.extend(torch.argmax(F.softmax(outputs, dim=1), dim=1))
            preds_list.extend(preds)

            correct_preds = (preds == targets).type(torch.FloatTensor)

            # calculate and accumulate accuracy
            accuracy += torch.mean(correct_preds).item() * 100

            # calculate test accuracy for each class
            for c in range(n_class):
                targets = targets.to("cpu")
                class_total[c] += (targets == c).sum()
                class_correct[c] += ((correct_preds) * (targets == c)).sum()

        # get average accuracy
        accuracy = accuracy / len(test_loader)
        # get average loss
        test_loss = test_loss / len(test_loader)

        logger.debug("Test Loss: {:.6f}".format(test_loss))
        # print('Test Focal Loss: {:.6f}'.format(test_focal_loss))
        accuracy_per_class(class_correct, class_total, class_names, accuracy)
        metrics = get_metrics(preds_list, target_list)

        class_data = {
            class_names[c]: {"correct": class_correct[c], "total": class_total[c]}
            for c in range(n_class)
        }

        return metrics, class_data

    vit_model = load_local_vit_model()
    logger.info("Making Model Prediction")
    metrics, class_data = model_eval(
        vit_model,
        inference_loader,
        class_names,
    )
    return metrics, class_data


def download_vit_model(model_name: str = "vit2024-04-11.pkl"):
    """Download the Vision Transformer (ViT) model from the cloud"""
    logger.info(f"Downloading the Vision Transformer (ViT) model: {model_name}")
    s3_client = boto3.client("s3")
    s3_client.download_file("alivio", f"models/{model_name}", model_name)
    logger.info(f"Model downloaded as {model_name}")


def live_prediction():
    st.markdown("# Live Building Damage Prediction 🔮")
    st.markdown(
        """
        This is a live building damage prediction tool for the xView2 dataset.
        Upload an image and its corresponding JSON file to get started.
        Our Vision transformer (ViT) model will use the uploaded files to make a prediction based on its pretraining
        
        ---
        """
    )
    st.info("Looking for uploads? Find satellite images with labeled JSON data in the **Gallery** section of the Menu")
    uploaded_img = st.file_uploader("Upload xView2 Image 📸")
    uploaded_json = st.file_uploader("Upload xView2 JSON 🗂")

    tr_wait = st.empty()
    tr_wait.info(
        "Once loaded, our pretrained model will be available until the session ends 🚀"
    )

    if not os.path.exists(os.path.join(os.getcwd(), "vit2024-04-11.pkl")):
        with st.spinner("Initializing Transformer Model 🤖"):
            download_vit_model()

    tr_wait.empty()

    predict_bool = st.button("Predict 🧙")

    if uploaded_img is not None and uploaded_json is not None and predict_bool:
        img_bytes = uploaded_img.getvalue()
        json_data = json.loads(uploaded_json.getvalue().decode("utf-8"))

        img_name: str = uploaded_img.name
        json_name: str = uploaded_json.name

        st.subheader("Uploaded Data Preview")

        st.markdown("#### Uploaded Image 🖼")
        st.image(img_bytes)
        st.markdown("#### Uploaded Label JSON 🗃")
        st.json(json_data)

        if os.path.exists(UPLOADS_IMG):
            logger.warning(f"Removing the dir with name: {UPLOADS_IMG}")

        os.system("rm -rf " + UPLOADS_IMG)

        if os.path.exists(UPLOADS_JSON):
            logger.warning(f"Removing the dir with name: {UPLOADS_JSON}")

        os.system("rm -rf " + UPLOADS_JSON)

        os.makedirs(UPLOADS_IMG, exist_ok=True)
        os.makedirs(UPLOADS_JSON, exist_ok=True)

        with open(
            f"data/xview_building_damage/inference/upload/img/{img_name}", "wb"
        ) as f:
            f.write(img_bytes)
            logger.info(f"Image saved as {UPLOADS_IMG}/{img_name}")

        with open(
            f"data/xview_building_damage/inference/upload/json/{json_name}", "w"
        ) as f:
            f.write(json.dumps(json_data))
            logger.info(f"JSON saved as {UPLOADS_JSON}/{json_name}")

        st.markdown("---")
        st.markdown("## Prediction Panel 🧙")
        pl = st.empty()
        pl.warning("Prediction usually takes about 1 minute ⏳")
        create_inference_csv_for_upload()
        with st.spinner(text="Preprocessing Images & Buildings Masks 🤿"):
            preprocess_image(INFERENCE_ROOT)
        with st.spinner(text="Making Predictions 🧙"):
            metrics, class_data = make_prediction()
            accuracy, f1_score, precision, recall = metrics

        pl.empty()
        pl.success("Prediction Complete 🪄")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{accuracy:.0%}")
        col2.metric("F1 Score", f"{f1_score:.0%}")
        col3.metric("Precision", f"{precision:.0%}")
        col4.metric("Recall", f"{recall:.0%}")

        building_count: int = pd.read_csv(
            os.path.join(POST_PROCESSED, "inference_data.csv")
        ).shape[0]

        st.write("##### What do these metrics mean? 🤔")
        st.markdown(
            f"""
            The uploaded satellite image consisted of a number of damaged buildings,
            and the resulting JSON file had corresponding labels and polygons for each building.
            This is used as ground truth, and the model predicts the damage level of each building.
            The metrics are calculated against the total number of buildings in the image,
            compared with the model's damage classifications per building.
            
            ##### Building Count in Image: `{building_count}`
            """
        )
        st.markdown("### Predictions by Class 📊")

        plot_df = pd.DataFrame(
            {
                "Class": [k for k in class_data.keys()],
                "Correct": [v["correct"] for v in class_data.values()],
                "Incorrect": [v["total"] - v["correct"] for v in class_data.values()],
            }
        )

        # Plotting the stacked bar chart
        fig = px.bar(
            plot_df,
            x="Class",
            y=["Correct", "Incorrect"],
            labels={"value": "Number of Predictions", "variable": "Prediction Type"},
            color_discrete_map={"Correct": "green", "Incorrect": "red"},
            title="Correct vs Incorrect Predictions by Class",
            template="plotly_white",
        )

        st.write(fig)

    elif predict_bool and not uploaded_img and not uploaded_json:
        st.warning(
            "Please upload an image and its corresponding JSON file to make a prediction"
        )


if __name__ == "__main__":
    if not os.path.exists(os.path.join(os.getcwd(), "vit2024-04-11.pkl")):
        download_vit_model()

    create_inference_csv_for_upload()
    preprocess_image(INFERENCE_ROOT)
    make_prediction()
