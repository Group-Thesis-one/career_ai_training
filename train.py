import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
import joblib


ROLES_PATH = Path("roles.json")
OUT_DIR = Path("out")
OUT_DIR.mkdir(exist_ok=True)


def load_roles(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_skill_vocab(roles):
    skills = set()
    for r in roles:
        for s in r.get("requiredSkills", []):
            skills.add(s.strip().lower())
        for s in r.get("optionalSkills", []):
            skills.add(s.strip().lower())
    return sorted(skills)


def get_weights(role):
    rw = {k.strip().lower(): int(v) for k, v in (role.get("requiredSkillWeights") or {}).items()}
    ow = {k.strip().lower(): int(v) for k, v in (role.get("optionalSkillWeights") or {}).items()}
    return rw, ow


def exp_bucket(years: int) -> int:
    # 0..5 bucket (cap at 5)
    if years < 0:
        years = 0
    if years > 5:
        years = 5
    return years


def focus_set(role, years: int):
    bucket_key = "0-1" if years <= 1 else ("2-3" if years <= 3 else "4+")
    focus = role.get("skillFocusByExperience", {}).get(bucket_key, []) or []
    return set([x.strip().lower() for x in focus])


def sample_applicant_for_role(role, role_index, skill_to_i, n_roles: int):
    required = [s.strip().lower() for s in role.get("requiredSkills", [])]
    optional = [s.strip().lower() for s in role.get("optionalSkills", [])]

    all_role_skills = list(dict.fromkeys(required + optional))
    if not all_role_skills:
        return None

    years = random.choice([0, 0, 1, 1, 2, 3, 4, 5])
    focus = focus_set(role, years)

    # simulate partial learning: has 30%–75% of role skills
    frac = random.uniform(0.30, 0.75)
    have_count = max(1, int(len(all_role_skills) * frac))
    have = set(random.sample(all_role_skills, have_count))

    # occasionally add a few random unrelated skills (noise)
    if random.random() < 0.25:
        vocab = list(skill_to_i.keys())
        for _ in range(random.randint(1, 3)):
            have.add(random.choice(vocab))

    missing_required = [s for s in required if s not in have]
    missing_optional = [s for s in optional if s not in have]

    rw, ow = get_weights(role)

    # label: recommend top K missing skills by (weight + required bonus + focus bonus)
    scored = []
    for s in missing_required:
        base = rw.get(s, 4)
        score = base + 2 + (2 if s in focus else 0)
        scored.append((score, s))
    for s in missing_optional:
        base = ow.get(s, 2)
        score = base + (2 if s in focus else 0)
        scored.append((score, s))

    scored.sort(reverse=True)
    if not scored:
        return None

    k = 5
    top = [s for _, s in scored[:k]]

    # X vector:
    # [skills multi-hot | exp bucket one-hot (6) | role one-hot (n_roles)]
    x = np.zeros(len(skill_to_i) + 6 + n_roles, dtype=np.float32)

    for s in have:
        i = skill_to_i.get(s)
        if i is not None:
            x[i] = 1.0

    x[len(skill_to_i) + exp_bucket(years)] = 1.0
    x[len(skill_to_i) + 6 + role_index] = 1.0

    # Y multilabel over skills (only predict missing skills to work on)
    y = np.zeros(len(skill_to_i), dtype=np.int32)
    for s in top:
        i = skill_to_i.get(s)
        if i is not None:
            y[i] = 1

    return x, y


def main():
    roles = load_roles(ROLES_PATH)
    role_titles = [r.get("title", f"role_{i}") for i, r in enumerate(roles)]

    skill_vocab = collect_skill_vocab(roles)
    skill_to_i = {s: i for i, s in enumerate(skill_vocab)}

    X, Y = [], []
    samples_per_role = 1200

    for role_index, role in enumerate(roles):
        for _ in range(samples_per_role):
            sample = sample_applicant_for_role(role, role_index, skill_to_i, n_roles=len(roles))
            if sample is None:
                continue
            x, y = sample
            X.append(x)
            Y.append(y)

    X = np.stack(X)
    Y = np.stack(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    model = OneVsRestClassifier(
        LogisticRegression(max_iter=2000, solver="liblinear")
    )
    model.fit(X_train, Y_train)

    joblib.dump(model, OUT_DIR / "recommender.joblib")

    with (OUT_DIR / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "skill_vocab": skill_vocab,
                "roles": role_titles
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    print("trained samples:", X.shape[0])
    print("saved:", OUT_DIR / "recommender.joblib")
    print("saved:", OUT_DIR / "meta.json")


if __name__ == "__main__":
    main()
