import numpy as np
import pandas as pd

import os 

from sklearn.model_selection import train_test_split

df = pd.read_csv('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')

df.drop(columns=['tweet_id'],inplace=True)
final_df = df[df['sentiment'].isin(['happiness','sadness'])]
final_df['sentiment'].replace({'happiness':1, 'sadness':0},inplace=True)


train_data, test_data = train_test_split(final_df, test_size=0.2, random_state=42)


data_path = os.path.join("data", "raw")
os.makedirs(data_path,exist_ok=True)

train_data.to_csv(os.path.join(data_path,"train.csv"))
test_data.to_csv(os.path.join(data_path,"test.csv")) 

# import numpy as np
# import pandas as pd
# import os
# from sklearn.model_selection import train_test_split

# df = pd.read_csv('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')

# df.drop(columns=['tweet_id'], inplace=True)
# final_df = df[df['sentiment'].isin(['happiness', 'sadness'])].copy()
# final_df['sentiment'].replace({'happiness': 1, 'sadness': 0}, inplace=True)

# train_data, test_data = train_test_split(final_df, test_size=0.2, random_state=42)

# data_path = os.path.join("data", "raw")
# os.makedirs(data_path, exist_ok=True)

# train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
# test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)
