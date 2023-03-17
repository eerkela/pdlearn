import pandas as pd
import pdcast; pdcast.attach()
import pdlearn; pdlearn.attach()
import sklearn.datasets


# load data
data = sklearn.datasets.load_wine(as_frame=True)

# convert to DataFrame
df = data["data"]
df["class"] = data["target_names"][data["target"]]

# split train, test
train, test = df.split_train_test(train_size=0.67, seed=123)

# train
model = df.fit("class", time_limit=60, ensemble_size=1)

# predict
predictions = pd.DataFrame({"observed": test["class"]})
predictions["predicted"] = model.predict(test)["class"]

# score
score = model.score(test)
