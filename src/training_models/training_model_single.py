import pandas as pd
import sys
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from classification_models import ClassificationModel

doc_config = open(sys.argv[1], 'r')

df_to_train = pd.read_csv(doc_config.readline().replace("\n", ""))
output_path = doc_config.readline().replace("\n", "")
random_seed = int(doc_config.readline().replace("\n", ""))

doc_config.close()

print("Processing random seed: ", random_seed)

responses = df_to_train["Label"].values
df_data = df_to_train.drop(columns=["Label"])

train_data, validation_data, train_response, validation_response = train_test_split(df_data, responses, random_state=random_seed, test_size=.30)

class_models = ClassificationModel(
    train_response=train_response,
    train_values=train_data,
    test_response=validation_response,
    test_values=validation_data
)

name_export = f"{output_path}exploring_{random_seed}.csv"
df_exploration = class_models.apply_exploring()
df_exploration.to_csv(name_export, index=False)
