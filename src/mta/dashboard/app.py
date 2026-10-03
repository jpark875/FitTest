"""Streamlit dashboard. Run with `streamlit run src/mta/dashboard/app.py` or `mta dashboard`."""

from __future__ import annotations

from decimal import Decimal

import plotly.express as px
import streamlit as st

from mta.config import get_settings
from mta.dashboard.data import load_frame
from mta.models.listing import TrendTag
from mta.storage.repository import Repository


@st.cache_resource
def _repository(url: str) -> Repository:
    get_settings().ensure_directories()
    repository = Repository(url)
    repository.create_schema()
    return repository


def main() -> None:
    """Render the page: thresholds in the sidebar, ranked table and charts in the body."""
    settings = get_settings()
    st.set_page_config(page_title="Micro-Trend Arbitrage", layout="wide")
    st.title("Micro-trend arbitrage")

    with st.sidebar:
        st.header("Buy bar")
        margin = st.slider("Minimum return", 0, 300, int(settings.min_margin_pct * 100), 5, "%d%%")
        absolute = st.number_input(
            "Minimum profit", min_value=0.0, value=float(settings.min_absolute_margin), step=1.0
        )
        confidence = st.slider("Minimum confidence", 0.0, 1.0, settings.min_confidence, 0.05)
        trends = st.multiselect(
            "Trends", [t for t in TrendTag if t is not TrendTag.NONE], format_func=lambda t: t.value
        )

    repository = _repository(settings.database_url)
    frame = load_frame(
        repository,
        min_margin_pct=Decimal(margin) / 100,
        min_absolute_margin=Decimal(str(absolute)),
        min_confidence=confidence,
        trends=trends or None,
    )

    if repository.count_listings() == 0:
        st.info("The warehouse is empty. Run `mta run` to ingest listings.")
        return
    if frame.empty:
        st.warning("No listings clear the current buy bar. Lower a threshold in the sidebar.")
        return

    left, middle, right = st.columns(3)
    left.metric("Opportunities", len(frame))
    middle.metric("Total profit", f"{frame['profit'].sum():,.2f}")
    right.metric("Median return", f"{frame['return_pct'].median():.0f}%")

    st.dataframe(
        frame.drop(columns=["reasoning", "model"]),
        hide_index=True,
        column_config={
            "url": st.column_config.LinkColumn("link"),
            "return_pct": st.column_config.NumberColumn("return %", format="%.0f"),
            "confidence": st.column_config.ProgressColumn("confidence", min_value=0, max_value=1),
        },
    )

    scatter, bars = st.columns(2)
    scatter.plotly_chart(
        px.scatter(
            frame,
            x="confidence",
            y="return_pct",
            color="trend",
            size="price",
            hover_name="title",
            title="Return vs confidence",
        ),
        use_container_width=True,
    )
    by_trend = frame.groupby("trend", as_index=False)["profit"].sum()
    bars.plotly_chart(
        px.bar(by_trend, x="trend", y="profit", title="Expected profit by trend"),
        use_container_width=True,
    )

    st.subheader("Why")
    for _, row in frame.head(10).iterrows():
        with st.expander(f"{row['title']} ({row['trend']})"):
            st.write(row["reasoning"])
            st.caption(f"{row['model']} - {row['comps']} comparable(s)")


main()
