%%writefile hms_sklearn_starter.py
"""HMS EEG classification: simple scikit-learn teaching baseline.

Run in a Kaggle notebook with the HMS competition dataset attached, or locally:
    python hms_sklearn_starter.py --data-dir /path/to/hms-harmful-brain-activity-classification

Requires: numpy, pandas, scipy, scikit-learn, pyarrow.
This is a teaching baseline, not a competitive Kaggle solution.
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
import pandas as pd
from scipy.signal import welch
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

TARGETS = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
FS = 200
# Step 9: the bipolar "double banana" montage that neurologists read: differences between
# neighbouring electrodes along four chains (left/right lateral, left/right parasagittal).
CHAINS = {
    "LL": ["Fp1", "F7", "T3", "T5", "O1"],
    "RL": ["Fp2", "F8", "T4", "T6", "O2"],
    "LP": ["Fp1", "F3", "C3", "P3", "O1"],
    "RP": ["Fp2", "F4", "C4", "P4", "O2"],
}
PAIRS = [(chain[k], chain[k + 1]) for chain in CHAINS.values() for k in range(4)]
BANDS = [(0.5, 2), (2, 4), (4, 8), (8, 13), (13, 20), (20, 30)]
FEATURE_VERSION = "v9"


def clean(x):
    """Replace missing samples by the channel median and clip extreme artefacts."""
    if not np.isfinite(x).any():
        return np.zeros_like(x)
    x = np.where(np.isfinite(x), x, np.nanmedian(x[np.isfinite(x)]))
    return np.clip(x, -1000, 1000)


def extract_features(eeg_path, offset_seconds=0.0):
    """Extract features from the central 10 s of a 50 s annotated segment.

    train.csv offsets refer to the START of each 50 s segment; the labels
    concern its central 10 s. Test EEG files are exactly 50 s long.
    """
    frame = pd.read_parquet(eeg_path)
    start = int(round((float(offset_seconds) + 20.0) * FS))
    end = start + 10 * FS
    if end > len(frame):
        raise ValueError(f"EEG window outside file: {eeg_path}, offset={offset_seconds}")
    segment = frame.iloc[start:end]
    raw = {c: clean(segment[c].to_numpy(dtype=np.float64)) for c in {e for p in PAIRS for e in p}}

    features, log_power = [], []
    for a, b in PAIRS:
        x = raw[a] - raw[b]
        frequencies, psd = welch(x, fs=FS, nperseg=400)
        power = np.array([psd[(frequencies >= lo) & (frequencies < hi)].sum() for lo, hi in BANDS])
        features.extend([np.std(x), np.ptp(x), np.mean(np.abs(np.diff(x)))])  # amplitude + "line length"
        features.extend(np.log1p(power))                      # absolute band power
        features.extend(power / (power.sum() + 1e-12))        # relative band power
        log_power.append(np.log1p(power))
    log_power = np.array(log_power).reshape(len(CHAINS), 4, len(BANDS)).mean(axis=1)  # per chain
    features.extend(log_power[0] - log_power[1])  # left - right, lateral chains
    features.extend(log_power[2] - log_power[3])  # left - right, parasagittal chains
    return np.asarray(features, dtype=np.float32)


def build_features(metadata, eeg_folder, is_train=True):
    """Step 9: extract the features of all EEGs in parallel on all CPU cores."""
    paths = [eeg_folder / f"{eeg_id}.parquet" for eeg_id in metadata["eeg_id"]]
    offsets = metadata["eeg_label_offset_seconds"].to_numpy() if is_train else np.zeros(len(metadata))
    X = Parallel(n_jobs=-1, batch_size=32, verbose=2)(
        delayed(extract_features)(p, o) for p, o in zip(paths, offsets))
    return np.vstack(X)


SPEC_REGIONS = ["LL", "RL", "LP", "RP"]
SPEC_BANDS = [(0.5, 4), (4, 8), (8, 13), (13, 20)]


def extract_spec_features(spec_path, offset_seconds=0.0):
    """Step 10: features from the 10-minute Kaggle spectrogram around the labelled segment.

    Spectrogram columns are named like 'LL_1.17' (region_frequency in Hz), one row per 2 s.
    We take log band power per region over the whole 10 min and over the central 60 s.
    """
    spec = pd.read_parquet(spec_path)
    t = spec["time"].to_numpy()
    window = spec[(t >= offset_seconds) & (t <= offset_seconds + 600)]
    centre = spec[(t >= offset_seconds + 270) & (t <= offset_seconds + 330)]
    features = []
    for region in SPEC_REGIONS:
        cols = [c for c in spec.columns if c.startswith(region + "_")]
        freqs = np.array([float(c.split("_")[1]) for c in cols])
        for lo, hi in SPEC_BANDS:
            band_cols = [c for c, f in zip(cols, freqs) if lo <= f < hi]
            for part in (window, centre):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN time points are filtered below
                    values = np.nanmean(np.log1p(np.clip(part[band_cols].to_numpy(dtype=np.float64), 0, None)), axis=1)
                values = values[np.isfinite(values)]
                features.extend([values.mean(), values.std()] if len(values) else [0.0, 0.0])
    return np.asarray(features, dtype=np.float32)


def build_spec_features(metadata, spec_folder, is_train=True):
    paths = [spec_folder / f"{sid}.parquet" for sid in metadata["spectrogram_id"]]
    offsets = metadata["spectrogram_label_offset_seconds"].to_numpy() if is_train else np.zeros(len(metadata))
    X = Parallel(n_jobs=-1, batch_size=32, verbose=2)(
        delayed(extract_spec_features)(p, o) for p, o in zip(paths, offsets))
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


MODELS = ["rf", "pca_logreg", "hgb", "hgb_slow"]
ENSEMBLE = ("rf", "hgb")  # step 7: average of the two tree models


def make_model(name):
    if name == "rf":
        return RandomForestClassifier(n_estimators=150, min_samples_leaf=2, random_state=42, n_jobs=-1)
    if name == "hgb":
        # Step 7: gradient boosting builds trees one after another, each correcting the previous ones.
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                              l2_regularization=1.0, random_state=42)
    if name == "hgb_slow":
        # Step 11: smaller learning rate with more trees, bigger leaves and a random half of the
        # features per split: each tree learns less, which reduces overfitting (bias-variance).
        return HistGradientBoostingClassifier(max_iter=600, learning_rate=0.03, max_leaf_nodes=31,
                                              min_samples_leaf=40, max_features=0.5, l2_regularization=1.0,
                                              early_stopping=False, random_state=42)
    if name == "pca_logreg":
        # Step 4: standardise (features have very different units), keep the principal
        # components that explain 95% of the variance, then multinomial logistic regression.
        return make_pipeline(StandardScaler(), PCA(n_components=0.95, svd_solver="full"),
                             LogisticRegression(C=1.0, max_iter=2000))
    raise ValueError(name)


def fit_soft(model, X, soft, weight=None):
    """Fit on soft labels; a pipeline needs the weights routed to its last step."""
    Xr, yr, wr = expand_soft(X, soft, weight)
    if hasattr(model, "steps"):
        model.fit(Xr, yr, **{f"{model.steps[-1][0]}__sample_weight": wr})
    else:
        model.fit(Xr, yr, sample_weight=wr)
    return model


def cross_validate(name, X, soft, groups, weight=None, n_splits=5):
    """Step 3: out-of-fold predictions with GroupKFold, so every EEG is validated once
    and no patient is in both the training and the validation part of a fold."""
    oof = np.zeros_like(soft)
    for fold, (tr, va) in enumerate(GroupKFold(n_splits=n_splits).split(X, groups=groups)):
        model = fit_soft(make_model(name), X[tr], soft[tr], None if weight is None else weight[tr])
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


def holm(pvalues):
    """Step 7: Holm-Bonferroni adjusted p-values (controls the family-wise error rate)."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adjusted[i] = running
    return adjusted


def expand_soft(X, soft, weight=None):
    """Step 2: repeat each EEG once per class, weighted by that class's vote fraction.

    Maximising the weighted log-likelihood of these rows is the same as minimising
    the cross-entropy to the vote distribution, i.e. the KL divergence up to a constant.
    """
    X_rep = np.repeat(X, soft.shape[1], axis=0)
    y_rep = np.tile(np.arange(soft.shape[1]), len(X))
    w_rep = soft.reshape(-1)
    if weight is not None:  # step 8: per-EEG weight (number of votes)
        w_rep = w_rep * np.repeat(weight, soft.shape[1])
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
    # Step 8: an EEG judged by many experts has a more reliable label than one judged by 1-2.
    # Weight each EEG by its number of votes, capped at 10 (weight 0.1 ... 1.0).
    n_votes = votes.sum(axis=1)
    eeg_weight = np.minimum(n_votes, 10) / 10
    high_quality = n_votes >= 10
    print(f"EEGs with >= 10 votes: {high_quality.sum()} of {len(n_votes)}")

    # Step 1: feature extraction is slow on all EEGs, so save the features once and reuse them.
    cache = Path(f"train_features_{FEATURE_VERSION}_{len(train)}.npy")
    if cache.exists():
        print(f"Loading cached training features from {cache}")
        X = np.load(cache)
    else:
        print("Extracting training EEG features...")
        X = build_features(train, args.data_dir / "train_eegs", is_train=True)
        np.save(cache, X)
    # Step 10: add the spectrogram features (cached separately).
    spec_cache = Path(f"train_spec_features_{len(train)}.npy")
    if spec_cache.exists():
        print(f"Loading cached spectrogram features from {spec_cache}")
        X_spec = np.load(spec_cache)
    else:
        print("Extracting training spectrogram features...")
        X_spec = build_spec_features(train, args.data_dir / "train_spectrograms", is_train=True)
        np.save(spec_cache, X_spec)
    X = np.hstack([X, X_spec])
    print(f"Number of features per EEG: {X.shape[1]}")

    # Reference: always predict the average vote distribution (no EEG information).
    prior = np.tile(soft_targets.mean(axis=0), (len(soft_targets), 1))
    print(f"Reference KL (always predict the mean vote distribution): {kl_per_eeg(soft_targets, prior).mean():.4f}")

    groups = train["patient_id"].to_numpy()
    oof, cv_kl, temps = {}, {}, {}
    for name in MODELS:
        print(f"5-fold GroupKFold cross-validation of {name} (grouped by patient)...")
        oof[name] = cross_validate(name, X, soft_targets, groups, eeg_weight)
        print(f"CV KL divergence {name}: {kl_per_eeg(soft_targets, oof[name]).mean():.4f}")
        temps[name] = fit_temperature(soft_targets, oof[name])
        oof[name] = temperature_scale(oof[name], temps[name])
        cv_kl[name] = kl_per_eeg(soft_targets, oof[name]).mean()
        print(f"  after temperature scaling (T={temps[name]:.2f}): {cv_kl[name]:.4f}")
    # Step 7: ensemble = average of the (calibrated) predictions of the tree models.
    name = "ensemble"
    avg = np.mean([oof[m] for m in ENSEMBLE], axis=0)
    temps[name] = fit_temperature(soft_targets, avg)
    oof[name] = temperature_scale(avg, temps[name])
    cv_kl[name] = kl_per_eeg(soft_targets, oof[name]).mean()
    print(f"CV KL divergence ensemble {'+'.join(ENSEMBLE)} (T={temps[name]:.2f}): {cv_kl[name]:.4f}")
    best = min(cv_kl, key=cv_kl.get)
    print(f"Best model on CV: {best}")
    for name in oof:
        print(f"  {name}: KL on EEGs with >= 10 votes: {kl_per_eeg(soft_targets[high_quality], oof[name][high_quality]).mean():.4f}")

    # Step 5: how certain are these numbers, and is the difference between the models real?
    per_eeg = {name: kl_per_eeg(soft_targets, oof[name]) for name in oof}
    for name in oof:
        (low, high), se = bootstrap_ci(per_eeg[name])
        print(f"{name}: KL {per_eeg[name].mean():.4f}, bootstrap 95% CI [{low:.4f}, {high:.4f}], SE {se:.4f}")
    # Paired Wilcoxon test of the best model against every other model. Several tests at
    # once -> Holm-Bonferroni correction for multiple testing (step 7).
    others = [name for name in oof if name != best]
    raw_p = [wilcoxon(per_eeg[best], per_eeg[name]).pvalue for name in others]
    for name, p, p_adj in zip(others, raw_p, holm(raw_p)):
        print(f"Wilcoxon {best} vs {name}: p={p:.3g}, Holm-adjusted p={p_adj:.3g}")

    if args.submission:
        members = ENSEMBLE if best == "ensemble" else (best,)
        test = pd.read_csv(args.data_dir / "test.csv")
        print("Extracting test EEG features...")
        X_test = build_features(test, args.data_dir / "test_eegs", is_train=False)
        X_test = np.hstack([X_test, build_spec_features(test, args.data_dir / "test_spectrograms", is_train=False)])
        preds = []
        for m in members:
            print(f"Refitting {m} on all selected training recordings...")
            model = fit_soft(make_model(m), X, soft_targets, eeg_weight)
            p_m = align_probabilities(model, X_test)
            # ensemble members are calibrated one by one, exactly as in cross-validation
            preds.append(temperature_scale(p_m, temps[m]) if best == "ensemble" else p_m)
        pred = temperature_scale(np.mean(preds, axis=0), temps[best])
        submission = pd.DataFrame(pred, columns=TARGETS)
        submission.insert(0, "eeg_id", test["eeg_id"].to_numpy())
        submission.to_csv("submission.csv", index=False)
        print("Saved submission.csv")


if __name__ == "__main__":
    main()
