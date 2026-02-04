import joblib
from .f1_score_threshold import Model
from pathlib import Path


model_path=Path(__file__).absolute().parents[1]/'models'/"best_threshold.pkl"
# print(model_path)
clf=joblib.load(model_path)

print(clf.get_params()['clf__threshold'])