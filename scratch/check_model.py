import joblib
import os
ROOT_DIR = r"e:\AI-proj\race-rc-project"
MODELS_DIR = os.path.join(ROOT_DIR, "models")
model_path = os.path.join(MODELS_DIR, "question_ranker.pkl")
model = joblib.load(model_path)
print(f"Model features: {model.n_features_in_}")
