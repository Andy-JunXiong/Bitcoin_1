from catboost import CatBoostClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
data  = pd.read_csv("E:/DR ROBERT/Capstone-Project-Bitcoin-Prediction--master/ml5/ml5/train.csv", delimiter="\t")
del data["user_tags"]
data = data.fillna(0)
X_train, X_validation, y_train, y_validation = train_test_split(data.iloc[:,:-1],data.iloc[:,-1],test_size=0.3 , random_state=1234)

categorical_features_indices = np.where(X_train.dtypes != np.float)[0]
model = CatBoostClassifier(iterations=100, depth=5,cat_features=categorical_features_indices,learning_rate=0.5, loss_function='Logloss',
                            logging_level='Verbose')

model.fit(X_train,y_train,eval_set=(X_validation, y_validation),plot=True)

import matplotlib.pyplot as plt
fea_ = model.feature_importances_
fea_name = model.feature_names_
plt.figure(figsize=(10, 10))
plt.barh(fea_name,fea_,height =0.5)

