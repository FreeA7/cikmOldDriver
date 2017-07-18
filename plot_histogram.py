import pandas as pd
import numpy as np

df = pd.read_csv(
    'D:\\workplace\\cikmOldDriver\\data\\training\\data_train.csv', header=None)
# df[4].apply(pd.value_counts).plot(kind='bar', subplots=True)
list_level_1 = df[3].unique().tolist()
list_level_2 = df[4].unique().tolist()
list_level_3 = df[5].unique().tolist()


df[5][df[5] == "Women"] = df[4][df[5] == "Women"]
