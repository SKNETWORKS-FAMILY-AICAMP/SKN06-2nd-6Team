from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd

class Preprocessor:
    def __init__(self, data):
        self.data = data

    def preprocess(self):
        # 결측치 삭제
        self.data.dropna(inplace=True)
        # label encoding
        l_encoder = LabelEncoder()
        self.data["label"] = l_encoder.fit_transform(self.data["label"])
        # one hot encoding
        self.data = pd.get_dummies(self.data, columns=["device"])
        return self.data
