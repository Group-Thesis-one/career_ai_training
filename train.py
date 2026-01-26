import json
import random
from dataclasses import dataclass  # optional (not used, but kept)
from pathlib import Path

import numpy as np
import joblib

# matplotlib (headless-safe)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split


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
    # reproducibility
    random.seed(42)
    np.random.seed(42)

    print("starting train.py...", flush=True)

    roles = load_roles(ROLES_PATH)
    role_titles = [r.get("title", f"role_{i}") for i, r in enumerate(roles)]
    print("roles loaded:", len(roles), flush=True)

    skill_vocab = collect_skill_vocab(roles)
    skill_to_i = {s: i for i, s in enumerate(skill_vocab)}
    print("skills in vocab:", len(skill_vocab), flush=True)

    X, Y = [], []
    samples_per_role = 1200
    print("generating samples... samples_per_role =", samples_per_role, flush=True)

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
    print("dataset ready. X shape:", X.shape, "Y shape:", Y.shape, flush=True)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    print("split done.", flush=True)

    # ---------------------------------------------------------
    # Graph: accuracy vs training iterations (real batch updates)
    # We train one SGDClassifier per label and log accuracy vs steps.
    # ---------------------------------------------------------
    rng = np.random.default_rng(42)

    train_eval_n = min(2000, X_train.shape[0])
    test_eval_n = min(2000, X_test.shape[0])

    train_eval_idx = rng.choice(X_train.shape[0], size=train_eval_n, replace=False)
    test_eval_idx = rng.choice(X_test.shape[0], size=test_eval_n, replace=False)

    X_train_eval = X_train[train_eval_idx]
    Y_train_eval = Y_train[train_eval_idx]
    X_test_eval = X_test[test_eval_idx]
    Y_test_eval = Y_test[test_eval_idx]

    n_labels = Y_train.shape[1]
    clfs = [SGDClassifier(loss="log_loss", random_state=42) for _ in range(n_labels)]

    batch_size = 512
    epochs = 5
    log_every = 5  # record a point every N batches

    steps = []
    train_acc = []
    test_acc = []

    indices = np.arange(X_train.shape[0])
    global_step = 0
    first = True

    print("training iteration model (SGD) for graph...", flush=True)

    for epoch in range(epochs):
        rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start:start + batch_size]
            Xb = X_train[batch_idx]
            Yb = Y_train[batch_idx]

            for j in range(n_labels):
                ybj = Yb[:, j]
                if first:
                    clfs[j].partial_fit(Xb, ybj, classes=np.array([0, 1]))
                else:
                    clfs[j].partial_fit(Xb, ybj)

            first = False
            global_step += 1

            if global_step % log_every == 0:
                pred_train = np.column_stack([clf.predict(X_train_eval) for clf in clfs])
                pred_test = np.column_stack([clf.predict(X_test_eval) for clf in clfs])

                tr = (pred_train == Y_train_eval).all(axis=1).mean()
                te = (pred_test == Y_test_eval).all(axis=1).mean()

                steps.append(global_step)
                train_acc.append(tr)
                test_acc.append(te)

                print(f"iter {global_step}  train={tr:.4f}  test={te:.4f}", flush=True)

    plt.figure(figsize=(9, 4))
    plt.plot(steps, train_acc, label="train (eval subset)")
    plt.plot(steps, test_acc, label="test (eval subset)")
    plt.xlabel("Training iterations (batch updates)")
    plt.ylabel("Accuracy (subset exact match)")
    plt.title("Accuracy vs training iterations")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plot_path = OUT_DIR / "training_accuracy_vs_iterations.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print("saved:", plot_path, flush=True)

    # ---------------------------------------------------------
    # Final model training (keeps your original output behavior)
    # ---------------------------------------------------------
    print("training final model...", flush=True)

    model = OneVsRestClassifier(
        LogisticRegression(max_iter=2000, solver="liblinear")
    )
    model.fit(X_train, Y_train)

    joblib.dump(model, OUT_DIR / "recommender.joblib")

    with (OUT_DIR / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(
            {"skill_vocab": skill_vocab, "roles": role_titles},
            f,
            ensure_ascii=False,
            indent=2
        )

    print("trained samples:", X.shape[0], flush=True)
    print("saved:", OUT_DIR / "recommender.joblib", flush=True)
    print("saved:", OUT_DIR / "meta.json", flush=True)


if __name__ == "__main__":
    main()
