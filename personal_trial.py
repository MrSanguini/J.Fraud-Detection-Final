
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for p in (ROOT / "pipeline", ROOT / "model"):
    sys.path.insert(0, str(p))

import joblib
import xgboost as xgb

import inference
from common import read_table
from model_prep import ModelPrep  # noqa: F401  (needed to unpickle the joblib)

prep = joblib.load(ROOT / "artifacts" / "model_prep.joblib")
model = xgb.XGBClassifier(enable_categorical=True)
model.load_model(str(ROOT / "artifacts" / "final_xgb.json"))
cal = joblib.load(ROOT / "artifacts" / "calibrator.joblib")

transaction_df = read_table(ROOT / "data" / "training_set").head(5)

X = prep.transform(transaction_df)        # SAME preprocessor as training
booster = model.get_booster()
result = inference.score(model, cal, X.iloc[[0]],
                         txn=transaction_df.iloc[0].to_dict(),
                         booster=booster,
                         feature_order=booster.feature_names)
print(result.to_dict())
print()
print(inference.explain(result))
