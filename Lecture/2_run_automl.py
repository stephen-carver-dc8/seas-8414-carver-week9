# Filename: 2_run_automl.py
import h2o
from h2o.automl import H2OAutoML

h2o.init()
train = h2o.import_file("dga_dataset_train.csv")
x = ['length', 'entropy']  # Features
y = "class"  # Target
train[y] = train[y].asfactor()
aml = H2OAutoML(max_models=20, max_runtime_secs=120, seed=1)
aml.train(x=x, y=y, training_frame=train)
print("H2O AutoML process complete.")
print("Leaderboard:")
print(aml.leaderboard.head())