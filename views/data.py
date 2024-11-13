import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def show_data():
    st.header("\:bookmark_tabs: Exploratory Data Analysis")
    data = pd.read_csv("data/waze_dataset.csv", index_col=0)
    # 데이터셋 기본 정보
    st.subheader("\:one: Data Information", divider=True)
    st.markdown(
        """This dataset is supplied as part of the Google Advanced Data Analytics Professional Certificate program courses on Coursera.    
        According to Google, this dataset contains synthetic data created in partnership with Waze."""
    )
    st.divider()
    # describe()
    st.subheader("\:two: Data Statistics", divider=True)
    st.dataframe(data.describe())
    st.divider()
    # head()
    st.subheader("\:three: Data Head", divider=True)
    st.dataframe(data.head())
    st.divider()
    # 시각화
    st.subheader("\:four: Data Visualization", divider=True)
    col1, col2 = st.columns([0.3, 0.7])
    with col1:
        columns = data.columns
        columns = columns.drop("label")
        selected_column = st.selectbox("Select Column", columns)
    with col2:
        fig, ax = plt.subplots()
        if selected_column != "device":
            sns.kdeplot(
                x=data[selected_column],
                data=data,
                fill=True,
                ec="gray",
                fc="white",
                legend=True,
                ax=ax,
            )
            sns.kdeplot(
                x=data[selected_column],
                data=data,
                hue=data["label"],
                fill=True,
                legend=True,
                ax=ax,
            )
        else:
            sns.countplot(
                x=data[selected_column], data=data, fill=False, legend=False, ax=ax
            )
            sns.countplot(
                x=data[selected_column],
                data=data,
                hue=data["label"],
                fill=True,
                legend=True,
                ax=ax,
            )
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        ax.set_title(f"{selected_column} 이탈 분포", fontsize=16)
        st.pyplot(fig)
    # 추가 시각화? 정보?
