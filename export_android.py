import json
from pathlib import Path
import joblib
import numpy as np

OUT_DIR = Path("out")
MODEL_PATH = OUT_DIR / "recommender.joblib"
META_PATH = OUT_DIR / "meta.json"

def main():
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    skills = meta["skill_vocab"]

    # For OneVsRest(LogReg), model.estimators_ list aligns with labels (skills)
    # We want a global "importance" per skill. We'll use L1 norm of coefficients.
    importances = []
    for est in model.estimators_:
        coef = getattr(est, "coef_", None)
        if coef is None:
            importances.append(0.0)
        else:
            importances.append(float(np.abs(coef).sum()))

    pairs = list(zip(skills, importances))
    pairs.sort(key=lambda x: x[1], reverse=True)

    export = {
        "version": 1,
        "note": "Trained locally from roles.json synthetic data. Global skill importance derived from model coefficients.",
        "skills": [{"skill": s, "importance": round(w, 6)} for s, w in pairs]
    }

    (OUT_DIR / "skill_priority_map.json").write_text(
        json.dumps(export, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("saved:", OUT_DIR / "skill_priority_map.json")

if __name__ == "__main__":
    main()
