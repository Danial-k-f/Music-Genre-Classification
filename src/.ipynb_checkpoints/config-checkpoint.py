import os
from sklearn.preprocessing import LabelEncoder

# مسیر دیتاست
DATA_PATH = "data/gtzan_dataset/genres"

# لیست ژانرها (مرتب برای سازگاری پایدار)
GENRES = sorted(os.listdir(DATA_PATH))

# LabelEncoder ثابت
encoder = LabelEncoder()
encoder.fit(GENRES)
