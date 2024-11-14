import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def show_data():
    st.header(":bookmark_tabs: Exploratory Data Analysis")
    data = pd.read_csv("data/waze_dataset.csv", index_col=0)
    # 데이터셋 기본 정보
    st.subheader(":one: Data Information", divider=True)
    st.markdown(
        """This dataset is supplied as part of the Google Advanced Data Analytics Professional Certificate program courses on Coursera.    
        According to Google, this dataset contains synthetic data created in partnership with Waze."""
    )
    st.divider()
    # describe()
    #
    st.subheader(":two: Data Statistics", divider=True)
    st.dataframe(data.describe())
    st.divider()
    # head()
    st.subheader(":three: Data Head", divider=True)
    st.dataframe(data.head())
    st.divider()
    # 시각화
    st.subheader(":four: Data Visualization", divider=True)
    col3, col4 = st.columns([0.3, 0.7])
    with col3:
        columns = data.columns
        columns = columns.drop("label")
        selected_column = st.selectbox("Select Column", columns)
    with col4:
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
    # 상관계수
    st.subheader(":five: Correlation", divider=True)
    co15, col6, col7 = st.columns([0.1, 0.8, 0.1])
    with col6:
        fig, ax = plt.subplots()
        fig.set_figwidth(8)
        fig.set_figheight(8)
        sns.heatmap(
            data=data.select_dtypes(include="number").corr(),
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            cmap="Blues",
            ax=ax,
        )
        ax.set_title("상관계수", fontsize=16)
        st.pyplot(fig)
    st.divider()
    # 추가 시각화? 정보?
