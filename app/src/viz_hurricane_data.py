"""Visualize building damage data and demographics + GDP on a map."""

import streamlit as st
import pandas as pd
import pydeck as pdk
from shapely import wkt
from shapely.geometry import Polygon
import time


GLOBAL_FILES: dict = {}


def _get_color(damage_category: str) -> list[int]:
    """Returns the color corresponding to the damage category."""
    color_map: dict[str, list[int]] = {
        "destroyed": [255, 0, 0],
        "major-damage": [255, 255, 0],
        "minor-damage": [0, 0, 255],
        "no-damage": [0, 255, 0],
    }
    if not color_map.get(damage_category):
        return [128, 128, 128]  # Gray color for unknown category
    else:
        return color_map[damage_category]


def load_viz_files():
    """Load the building and hurricane data for visualization."""
    global GLOBAL_FILES

    building_df: pd.DataFrame = pd.read_parquet("../files/buildings.parquet.gz")
    building_df["map_polygon_shape"] = (
        building_df["map_polygon"].dropna().apply(wkt.loads)
    )
    building_df["geometry"] = (
        building_df["map_polygon_shape"]
        .dropna()
        .apply(lambda cell: list(cell.exterior.coords))
    )
    building_df = building_df.dropna(subset=["geometry"]).reset_index(drop=True)
    building_df["color"] = building_df["damage"].apply(_get_color)
    building_df = building_df[
        ["geometry", "map_polygon", "color", "damage", "disaster"]
    ]
    GLOBAL_FILES["building_df"] = building_df

    h3_8_df: pd.DataFrame = pd.read_parquet("../files/hurricanes_h3_8.parquet.gz")
    GLOBAL_FILES["h3_8_df"] = h3_8_df


def _render_h3_layer(h3_resolution: int = 8) -> tuple:
    """Takes a layer number and returns the corresponding hexagon layer and tooltip for the map."""

    pdk_layer = pdk.Layer(
        "H3HexagonLayer",
        GLOBAL_FILES["h3_8_df"],
        pickable=True,
        stroked=True,
        filled=True,
        extruded=False,
        get_hexagon=f"h3_cell_{h3_resolution}",
        get_fill_color="[255-gdp_per_capita, 255, gdp_per_capita]",
        opacity=0.1,
        get_line_color=[255, 255, 255],
        line_width_min_pixels=2,
    )
    tool_tip = {
        "text": "Population: {sum_population}\nGDP Per Capita: {gdp_per_capita}"
    }

    return pdk_layer, tool_tip


def select_layers() -> list:
    show_h3_layer: bool = st.toggle("Show Demographics (H3)", value=False)
    show_buildings_layer: bool = st.toggle("Show Buildings (Geometries)", value=False)
    h3_layer = None

    if show_h3_layer:
        h3_layer = st.select_slider("Select H3 level", [7, 8, 9, 10], value=8)

    map_layers = []
    tool_tip = None

    if show_buildings_layer:
        polygon_layer = pdk.Layer(
            "PolygonLayer",
            GLOBAL_FILES["hurricane_df"],
            stroked=True,
            get_polygon="geometry",
            filled=True,
            extruded=False,
            get_fill_color="color",
            get_line_color=[255, 255, 255],
            auto_highlight=True,
            pickable=True,
        )
        buildings_tooltip = {"text": "Damage: {damage_level}"}
        tool_tip = buildings_tooltip
        map_layers.append(polygon_layer)

    if h3_layer:
        lyr, h3_tooltip = _render_h3_layer(h3_layer)
        map_layers.append(lyr)
        tool_tip = h3_tooltip

    return map_layers, tool_tip


def select_and_display_hurricane():
    load_viz_files()
    hurricane: str = st.selectbox(
        "Select a hurricane", ["Michael", "Harvey", "Florence", "Matthew"]
    )

    bud_df = GLOBAL_FILES["building_df"]
    GLOBAL_FILES["hurricane_df"] = bud_df[
        bud_df["disaster"].str.contains(hurricane, case=False)
    ]
    select_hurricane_df = GLOBAL_FILES["hurricane_df"].copy()

    map_layers, tool_tip = select_layers()

    random_hurricane_building_polygon: Polygon = (
        select_hurricane_df["map_polygon"]
        .dropna()
        .sample(1)
        .apply(wkt.loads)
        .iloc[0]
    )

    view_lng, view_lat = random_hurricane_building_polygon.centroid.coords[0]

    view_state = pdk.ViewState(latitude=view_lat, longitude=view_lng, zoom=12)

    if tool_tip:
        if tool_tip["text"].startswith("Damage:"):
            with open(f"../docs/{hurricane.lower()}.md") as f:
                markdown_stream = f.read()

            if markdown_stream:

                def stream_data():
                    for word in markdown_stream.split(" "):
                        yield word + " "
                        time.sleep(0.04)

                st.write_stream(stream_data)

            st.subheader(f"{hurricane} Building Damage Level - Distribution")
            st.bar_chart(
                select_hurricane_df["damage"]
                .value_counts()
                .to_frame()
                .reset_index()
                .rename(columns={"damage": "Damage Level", "count": "Building Count"}),
                x="Damage Level",
                y="Building Count",
            )

    st.write("### Map of Disaster Area")
    with st.spinner(text="Building Map - Please wait..."):
        r = pdk.Deck(layers=map_layers, initial_view_state=view_state, tooltip=tool_tip)
        st.pydeck_chart(r)
