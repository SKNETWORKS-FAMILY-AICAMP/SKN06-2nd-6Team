import seaborn as sns
import matplotlib.pyplot as plt


class Visualizer:
    def __init__(self, data, columns):
        self.data = data
        self.columns = columns

    def target_plot(self):
        plt.figure(figsize=(12, 15))
        for i, col in enumerate(self.columns):
            if col != "device":
                plt.subplot(6, 2, i + 1)
                sns.kdeplot(
                    x=self.data[col],
                    data=self.data,
                    fill=True,
                    ec="gray",
                    fc="white",
                    legend=True,
                )
                sns.kdeplot(
                    x=self.data[col],
                    data=self.data,
                    hue=self.data["label"],
                    fill=True,
                    legend=True,
                )
            else:
                plt.subplot(6, 2, 11)
                sns.countplot(
                    x=self.data[col], data=self.data, fill=False, legend=False
                )
                sns.countplot(
                    x=self.data[col],
                    data=self.data,
                    hue=self.data["label"],
                    fill=True,
                    legend=False,
                )
        plt.suptitle("각 column 별 이탈 분포", fontsize=16)
        plt.tight_layout()
        plt.show()

    def feature_plot(self):
        plt.figure(figsize=(12, 15))
        for i, col in enumerate(self.columns):
            if col != "label":
                plt.subplot(6, 2, i + 1)
                self.data[col].hist()
                plt.grid(linestyle=":")
            else:
                plt.subplot(6, 2, 11)
                plt.pie(x=self.data[col].value_counts())
            plt.xlabel(col)
        plt.suptitle("각 column 별 분포", fontsize=16)
        plt.tight_layout()
        plt.show()

    def box_plot(self):
        plt.figure(figsize=(12, 15))
        for i, col in enumerate(self.columns):
            plt.subplot(5, 2, i + 1)
            plt.boxplot(self.data[col])
            plt.xlabel(col)
        plt.suptitle("각 column 별 box plot", fontsize=16)
        plt.tight_layout()
        plt.show()
