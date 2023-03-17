import numpy as np
import pandas as pd
import pdcast; pdcast.attach()
import pdlearn; pdlearn.attach()
import sklearn.datasets


# load data
data = sklearn.datasets.fetch_20newsgroups_vectorized(as_frame=True)

# convert to DataFrame
df = data["data"]
# df["category"] = np.array(data["target_names"])[data["target"]]

# split train, test
train, test = df.split_train_test(train_size=0.67, seed=123)

# train
model = df.fit("category", time_limit=600, ensemble_size=1)

# predict
predictions = pd.DataFrame({"observed": test["category"]})
predictions["predicted"] = model.predict(test)["category"]

# score
score = model.score(test)
