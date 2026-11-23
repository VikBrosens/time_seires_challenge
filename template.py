%%writefile hms_sklearn_starter.py
"""HMS EEG classification: simple scikit-learn teaching baseline.

Run in a Kaggle notebook with the HMS competition dataset attached, or locally:
    python hms_sklearn_starter.py --data-dir /path/to/hms-harmful-brain-activity-classification

Requires: numpy, pandas, scipy, scikit-learn, pyarrow.
This is a teaching baseline, not a competitive Kaggle solution.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

TARGETS = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
FS = 200
# A small, interpretable subset of EEG electrodes (not the EKG channel).
CHANNELS = ["Fp1", "F7", "T3", "T5", "O1", "Fp2", "F8", "T4", "T6", "O2"]
BANDS = [(0.5, 4), (4, 8), (8, 13), (13, 30)]


def extract_features(eeg_path, offset_seconds=0.0):
    """Extract features from the central 10 s of a 50 s annotated segment.

    train.csv offsets refer to the START of each 50 s segment; the labels
    concern its central 10 s. Test EEG files are exactly 50 s long.
    """
    frame = pd.read_parquet(eeg_path)
    channels = [c for c in CHANNELS if c in frame.columns]
    if not channels:
        raise ValueError(f"No expected EEG channels in {eeg_path}")

    start = int(round((float(offset_seconds) + 20.0) * FS))
    end = start + 10 * FS
    if end > len(frame):
        raise ValueError(f"EEG window outside file: {eeg_path}, offset={offset_seconds}")
    segment = frame.iloc[start:end][channels].to_numpy(dtype=np.float64)

    features = []
    for j in range(len(channels)):
        x = segment[:, j]
        if not np.isfinite(x).any():
            x = np.zeros_like(x)
        else:
            fill = np.nanmedian(x[np.isfinite(x)])
            x = np.where(np.isfinite(x), x, fill)
        x = np.clip(x, -1000, 1000)
        features.extend([np.mean(x), np.std(x), np.ptp(x)])
        frequencies, psd = welch(x, fs=FS, nperseg=min(512, len(x)))
        for low, high in BANDS:
            mask = (frequencies >= low) & (frequencies < high)
            features.append(np.log1p(np.sum(psd[mask])))
    return np.asarray(features, dtype=np.float32)


def build_features(metadata, eeg_folder, is_train=True):
    X = []
    for i, row in enumerate(metadata.itertuples(index=False)):
        eeg_path = eeg_folder / f"{row.eeg_id}.parquet"
        offset = row.eeg_label_offset_seconds if is_train else 0.0
        X.append(extract_features(eeg_path, offset))
        if (i + 1) % 200 == 0:
            print(f"  Extracted {i + 1}/{len(metadata)} recordings")
    return np.vstack(X)


def align_probabilities(model, X, n_classes=6):
    """Keep all six classes even when a small training subset omits one."""
    p = np.zeros((len(X), n_classes), dtype=float)
    p[:, model.classes_.astype(int)] = model.predict_proba(X)
    return np.clip(p, 1e-8, None) / np.clip(p, 1e-8, None).sum(axis=1, keepdims=True)


def kl_per_eeg(soft, prob):
    """KL divergence between expert votes and prediction, one value per EEG."""
    y = np.clip(soft, 1e-12, 1.0)
    return np.sum(y * (np.log(y) - np.log(prob)), axis=1)


MODELS = ["rf", "pca_logreg"]


def make_model(name):
    if name == "rf":
        return RandomForestClassifier(n_estimators=150, min_samples_leaf=2, random_state=42, n_jobs=-1)
    if name == "pca_logreg":
        # Step 4: standardise (features have very different units), keep the principal
        # components that explain 95% of the variance, then multinomial logistic regression.
        return make_pipeline(StandardScaler(), PCA(n_components=0.95, svd_solver="full"),
                             LogisticRegression(C=1.0, max_iter=2000))
    raise ValueError(name)


def fit_soft(model, X, soft):
    """Fit on soft labels; a pipeline needs the weights routed to its last step."""
    Xr, yr, wr = expand_soft(X, soft)
    if hasattr(model, "steps"):
        model.fit(Xr, yr, **{f"{model.steps[-1][0]}__sample_weight": wr})
    else:
        model.fit(Xr, yr, sample_weight=wr)
    return model


def cross_validate(name, X, soft, groups, n_splits=5):
    """Step 3: out-of-fold predictions with GroupKFold, so every EEG is validated once
    and no patient is in both the training and the validation part of a fold."""
    oof = np.zeros_like(soft)
    for fold, (tr, va) in enumerate(GroupKFold(n_splits=n_splits).split(X, groups=groups)):
        model = fit_soft(make_model(name), X[tr], soft[tr])
        oof[va] = align_probabilities(model, X[va])
        print(f"  fold {fold + 1}/{n_splits}: KL {kl_per_eeg(soft[va], oof[va]).mean():.4f}")
    return oof


def temperature_scale(prob, T):
    """Step 6: soften (T > 1) or sharpen (T < 1) predicted probabilities."""
    logp = np.log(prob) / T
    p = np.exp(logp - logp.max(axis=1, keepdims=True))
    return p / p.sum(axis=1, keepdims=True)


def fit_temperature(soft, prob, grid=np.linspace(0.5, 5.0, 46)):
    """Step 6: choose the T that minimises the out-of-fold KL (a one-parameter
    maximum-likelihood fit on validation predictions)."""
    kls = [kl_per_eeg(soft, temperature_scale(prob, T)).mean() for T in grid]
    return float(grid[int(np.argmin(kls))])


def bootstrap_ci(values, q=1000, alpha=0.05, seed=42):
    """Step 5: bootstrap the mean KL: resample the EEGs with replacement q times."""
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.array([values[rng.integers(0, n, n)].mean() for _ in range(q)])
    return np.quantile(means, [alpha / 2, 1 - alpha / 2]), means.std(ddof=1)


def expand_soft(X, soft):
    """Step 2: repeat each EEG once per class, weighted by that class's vote fraction.

    Maximising the weighted log-likelihood of these rows is the same as minimising
    the cross-entropy to the vote distribution, i.e. the KL divergence up to a constant.
    """
    X_rep = np.repeat(X, soft.shape[1], axis=0)
    y_rep = np.tile(np.arange(soft.shape[1]), len(X))
    w_rep = soft.reshape(-1)
    keep = w_rep > 0
    return X_rep[keep], y_rep[keep], w_rep[keep]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("/kaggle/input/hms-harmful-brain-activity-classification"))
    parser.add_argument("--max-train", type=int, default=0, help="Limit examples for a quick classroom run; 0 uses all rows")
    parser.add_argument("--submission", action="store_true", help="Also predict the provided test EEGs")
    args = parser.parse_args()

    train = pd.read_csv(args.data_dir / "train.csv")
    # Keep one annotated window per EEG for a fast, simple starting point.
    train = train.drop_duplicates(subset="eeg_id").reset_index(drop=True)
    if args.max_train > 0 and len(train) > args.max_train:
        train = train.sample(n=args.max_train, random_state=42).reset_index(drop=True)

    # Soft targets are the normalized expert votes; the model is trained on them (step 2).
    votes = train[TARGETS].to_numpy(dtype=float)
    soft_targets = votes / votes.sum(axis=1, keepdims=True)

    # Step 1: feature extraction is slow on all EEGs, so save the features once and reuse them.
    cache = Path(f"train_features_{len(train)}.npy")
    if cache.exists():
        print(f"Loading cached training features from {cache}")
        X = np.load(cache)
    else:
        print("Extracting training EEG features...")
        X = build_features(train, args.data_dir / "train_eegs", is_train=True)
        np.save(cache, X)

    # Reference: always predict the average vote distribution (no EEG information).
    prior = np.tile(soft_targets.mean(axis=0), (len(soft_targets), 1))
    print(f"Reference KL (always predict the mean vote distribution): {kl_per_eeg(soft_targets, prior).mean():.4f}")

    groups = train["patient_id"].to_numpy()
    oof, cv_kl, temps = {}, {}, {}
    for name in MODELS:
        print(f"5-fold GroupKFold cross-validation of {name} (grouped by patient)...")
        oof[name] = cross_validate(name, X, soft_targets, groups)
        print(f"CV KL divergence {name}: {kl_per_eeg(soft_targets, oof[name]).mean():.4f}")
        temps[name] = fit_temperature(soft_targets, oof[name])
        oof[name] = temperature_scale(oof[name], temps[name])
        cv_kl[name] = kl_per_eeg(soft_targets, oof[name]).mean()
        print(f"  after temperature scaling (T={temps[name]:.2f}): {cv_kl[name]:.4f}")
    best = min(cv_kl, key=cv_kl.get)
    print(f"Best model on CV: {best}")

    # Step 5: how certain are these numbers, and is the difference between the models real?
    per_eeg = {name: kl_per_eeg(soft_targets, oof[name]) for name in MODELS}
    for name in MODELS:
        (low, high), se = bootstrap_ci(per_eeg[name])
        print(f"{name}: KL {per_eeg[name].mean():.4f}, bootstrap 95% CI [{low:.4f}, {high:.4f}], SE {se:.4f}")
    # Paired test: both models are evaluated on the same EEGs; KL values are not normal.
    stat, p = wilcoxon(per_eeg[MODELS[0]], per_eeg[MODELS[1]])
    print(f"Wilcoxon signed-rank test {MODELS[0]} vs {MODELS[1]}: T={stat:.1f}, p={p:.3g}")

    if args.submission:
        print(f"Refitting {best} on all selected training recordings...")
        model = fit_soft(make_model(best), X, soft_targets)
        test = pd.read_csv(args.data_dir / "test.csv")
        print("Extracting test EEG features...")
        X_test = build_features(test, args.data_dir / "test_eegs", is_train=False)
        pred = temperature_scale(align_probabilities(model, X_test), temps[best])
        submission = pd.DataFrame(pred, columns=TARGETS)
        submission.insert(0, "eeg_id", test["eeg_id"].to_numpy())
        submission.to_csv("submission.csv", index=False)
        print("Saved submission.csv")


if __name__ == "__main__":
    main()
