import pandas as pd
import pdcast; pdcast.attach()
import pdlearn; pdlearn.attach()
import sklearn.datasets


# load data
data = sklearn.datasets.load_digits(as_frame=True)

# convert to DataFrame
df = data["data"]
df["digit"] = pdcast.cast(data["target"], categorical=True)
# df["digit"] = pd.Series(data["target"])

# split train, test
train, test = df.split_train_test(train_size=0.8, seed=123)

# train
model = df.fit("digit", time_limit=60, ensemble_size=1)

# predict
predictions = pd.DataFrame({"observed": test["digit"]})
predictions["predicted"] = model.predict(test)["digit"]

# score
score = model.score(test)
