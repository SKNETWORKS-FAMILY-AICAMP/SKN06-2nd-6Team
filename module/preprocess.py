from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd

class Preprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
    def preprocess(self):
        data = pd.read_csv(self.data_path, index_col=0)
        # 결측치 삭제
        data.dropna(inplace=True)
        # label encoding
        l_encoder = LabelEncoder()
        if "label" in data.columns:
            data["label"] = l_encoder.fit_transform(data["label"])
        # one hot encoding
        data = pd.get_dummies(data, columns=["device"])
        return data
