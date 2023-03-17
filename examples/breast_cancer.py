import pandas as pd
import pdcast; pdcast.attach()
import pdlearn; pdlearn.attach()
import sklearn.datasets


# load data
data = sklearn.datasets.load_breast_cancer(as_frame=True)

# convert to DataFrame
df = data["data"]
df["malignant"] = pdcast.cast(data["target"], categorical=True)

# split train, test
train, test = df.split_train_test(train_size=0.67, seed=123)

# train
model = df.fit("malignant", time_limit=60, ensemble_size=1)

# predict
predictions = pd.DataFrame({"observed": test["malignant"]})
predictions["predicted"] = model.predict(test)["malignant"]

# score
score = model.score(test)
