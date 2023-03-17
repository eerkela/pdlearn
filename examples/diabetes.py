import pandas as pd
import pdcast; pdcast.attach()
import pdlearn; pdlearn.attach()
import sklearn.datasets


# load data
data = sklearn.datasets.load_diabetes(as_frame=True)

# convert to DataFrame
df = data["data"]
df["progress"] = data["target"]

# split train, test
train, test = df.split_train_test(train_size=0.8, seed=123)

# train
model = df.fit("progress", time_limit=60, ensemble_size=1)

# predict
predictions = pd.DataFrame({"observed": test["progress"]})
predictions["predicted"] = model.predict(test)["progress"]

# score
score = model.score(test)
