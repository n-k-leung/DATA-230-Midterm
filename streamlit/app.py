"""
Passenger Satisfaction Classification Dashboard
Interactive Streamlit app — updated with:
  - Log transformation pipeline option (delay columns)
  - Best hyperparameters toggle (no search)
  - LIME explainability page
  - Metric multiselect for drift chart
  - Predict New Passenger page removed
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve,
    precision_recall_curve, average_precision_score
)
from sklearn.inspection import permutation_importance
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.base import BaseEstimator, ClassifierMixin
from imblearn.over_sampling import SMOTE
import lime
import lime.lime_tabular

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Passenger Satisfaction ML Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #ffffff; }

    h1, h2, h3 {
        color: #111827 !important;
    }

    .stMetric {
        background: linear-gradient(135deg, #eef2ff 0%, #e0e7ff 100%);
        border: 1px solid #4F46E5;
        border-radius: 12px;
        padding: 16px;
    }

    .stMetric label {
        color: #1e293b !important;
        font-size: 0.8rem !important;
    }

    .stMetric [data-testid="metric-value"] {
        color: #111827 !important;
        font-size: 1.5rem !important;
    }

    .section-header {
        background: linear-gradient(90deg, #4F46E5, #c7d2fe);
        color: white;
        border-left: 4px solid #312e81;
        padding: 8px 16px;
        margin: 24px 0 16px 0;
        border-radius: 0 8px 8px 0;
    }

    .section-header b {
        color: white !important;
    }

    div[data-testid="stSidebar"] {
        background: #f1f5f9;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# WEIGHTED LDA
# ─────────────────────────────────────────────────────────────────
class WeightedLDA(BaseEstimator, ClassifierMixin):
    def __init__(self, solver="svd", shrinkage=None):
        self.solver = solver
        self.shrinkage = shrinkage

    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            w = sample_weight / sample_weight.min()
            counts = np.round(w).astype(int).clip(min=1)
            X = np.repeat(X, counts, axis=0)
            y = np.repeat(y, counts, axis=0)
        self.model_ = LinearDiscriminantAnalysis(solver=self.solver, shrinkage=self.shrinkage)
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):        return self.model_.predict(X)
    def predict_proba(self, X):  return self.model_.predict_proba(X)
    def score(self, X, y):       return self.model_.score(X, y)


# ─────────────────────────────────────────────────────────────────
# BEST HYPERPARAMETERS (from tuning run)
# ─────────────────────────────────────────────────────────────────
BEST_PARAMS = {
    "Hist Gradient Boosting": {"max_iter": 300, "max_depth": 7, "learning_rate": 0.1},
    "Neural Network (MLP)":   {"hidden_layer_sizes": (100, 50), "alpha": 0.001},
    "LDA":                    {"solver": "lsqr", "shrinkage": None},
}


# ─────────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────────
@st.cache_resource
def run_pipeline(n_samples, test_size, smote_on, weight_on, log_transform_on, log_passes, use_best_params, seed):
    FEATURES = [
        "flight_distance", "departure_delay_in_minutes", "arrival_delay_in_minutes",
        "seat_comfort", "inflight_wifi_service", "food_and_drink", "inflight_entertainment",
        "ease_of_online_booking", "online_boarding", "online_support",
        "age", "gender", "customer_type", "class"
    ]
    TARGET = "satisfaction"

    df = pd.read_csv("dataset/passenger_satisfaction_cleaned.csv")

    le_gender   = LabelEncoder()
    le_customer = LabelEncoder()
    le_class    = LabelEncoder()
    df["gender"]        = le_gender.fit_transform(df["gender"])
    df["customer_type"] = le_customer.fit_transform(df["customer_type"])
    df["class"]         = le_class.fit_transform(df["class"])

    # Log transformation of delay columns
    if log_transform_on:
        for _ in range(log_passes):
            df["departure_delay_in_minutes"] = np.log1p(df["departure_delay_in_minutes"])
            df["arrival_delay_in_minutes"]   = np.log1p(df["arrival_delay_in_minutes"])

    X = df[FEATURES].values
    y = df[TARGET].values

    original_counts = np.bincount(y)

    # SMOTE
    if smote_on:
        minority_count = int(np.sum(y == 0))
        k_nn = max(1, min(minority_count - 1, 5))
        smote = SMOTE(k_neighbors=k_nn, random_state=seed)
        X_res, y_res = smote.fit_resample(X, y)
    else:
        X_res, y_res = X, y

    resampled_counts = np.bincount(y_res)

    df_res = pd.DataFrame(X_res, columns=FEATURES)
    df_res[TARGET] = y_res

    disloyal_code = le_customer.transform(["disloyal Customer"])[0]
    business_code = le_class.transform(["Business"])[0]
    df_res["loyalty_group"] = df_res["customer_type"].apply(
        lambda x: "Disloyal" if x == disloyal_code else "Loyal")
    df_res["class_group"] = df_res["class"].apply(
        lambda x: "Business" if x == business_code else "Non-Business")

    # Sample weights
    weights = np.ones(len(df_res))
    if weight_on:
        is_dis = df_res["customer_type"] == disloyal_code
        is_non = df_res["class"] != business_code
        weights[is_dis & is_non]  *= 1.8
        weights[is_dis & ~is_non] *= 1.4
        weights[~is_dis & is_non] *= 1.3

    X_train, X_test, y_train, y_test, sw_train, _ = train_test_split(
        X_res, y_res, weights,
        test_size=test_size, random_state=seed, stratify=y_res
    )

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Models: default or best params
    if use_best_params:
        models = {
            "Hist Gradient Boosting": HistGradientBoostingClassifier(**BEST_PARAMS["Hist Gradient Boosting"]),
            "Neural Network (MLP)":   MLPClassifier(max_iter=500, **BEST_PARAMS["Neural Network (MLP)"]),
            "LDA":                    WeightedLDA(**BEST_PARAMS["LDA"]),
        }
    else:
        models = {
            "Hist Gradient Boosting": HistGradientBoostingClassifier(),
            "Neural Network (MLP)":   MLPClassifier(max_iter=500),
            "LDA":                    WeightedLDA(),
        }

    def evaluate(name, model):
        model.fit(X_train_sc, y_train, sample_weight=sw_train)
        y_pred = model.predict(X_test_sc)
        y_prob = model.predict_proba(X_test_sc)[:, 1]
        cm_    = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm_.ravel()
        return {
            "model":       name,
            "accuracy":    accuracy_score(y_test, y_pred),
            "precision":   precision_score(y_test, y_pred),
            "recall":      recall_score(y_test, y_pred),
            "f1":          f1_score(y_test, y_pred),
            "auc":         roc_auc_score(y_test, y_prob),
            "specificity": tn / (tn + fp),
            "model_obj":   model,
            "cm": cm_, "prob": y_prob
        }

    results = [evaluate(name, model) for name, model in models.items()]

    return {
        "results": results,
        "X_test_sc":  X_test_sc,
        "X_train_sc": X_train_sc,
        "y_test":  y_test,
        "y_train": y_train,
        "sw_train": sw_train,
        "FEATURES": FEATURES,
        "original_counts":  original_counts,
        "resampled_counts": resampled_counts,
        "df_res": df_res,
        "best_models": {r["model"]: r["model_obj"] for r in results},
    }

# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Dashboard Controls")
    st.markdown("---")

    st.markdown("### Dataset")
    n_samples = 5000
    test_size = st.slider("Test Split (%)", 10, 40, 20, 5) / 100
    seed      = 42

    st.markdown("### Pipeline Options")
    smote_on         = st.toggle("SMOTE Oversampling",    value=True)
    weight_on        = st.toggle("Domain Sample Weights", value=True)
    log_transform_on = st.toggle(
        "Log Transformation",
        value=False,
        help="Applies log1p to departure_delay_in_minutes and arrival_delay_in_minutes"
    )
    log_passes = 1
    if log_transform_on:
        log_passes = st.slider(
            "Log Strength (passes)",
            min_value=1,
            max_value=3,
            value=1,
            help="1 = log1p(x), 2 = log1p(log1p(x)), 3 = log1p(log1p(log1p(x)))"
        )
        st.markdown("<small style='color:#8892b0'>↳ Applied to departure & arrival delay</small>",
                    unsafe_allow_html=True)

    st.markdown("### Hyperparameter Tuning")
    use_best_params = st.toggle(
        "Best Hyperparameters",
        value=False,
        help="OFF = sklearn defaults  |  ON = pre-tuned best params"
    )
    if use_best_params:
        st.markdown("""<small style='color:#8892b0'>
<b>HGB:</b> max_iter=300, max_depth=7, lr=0.1<br>
<b>MLP:</b> layers=(100,50), alpha=0.001<br>
<b>LDA:</b> solver=lsqr, shrinkage=None
</small>""", unsafe_allow_html=True)

    st.markdown("---")
    run_btn = st.button(" Run Pipeline", use_container_width=True, type="primary")

    st.markdown("### Navigation")
    page = st.radio("Go to", [
        "Overview",
        "Data & SMOTE",
        "Model Metrics",
        "ROC & PR Curves",
        "Confusion Matrices",
        "Feature Importance",
        "LIME Explanations",
    ], label_visibility="collapsed")

# ─────────────────────────────────────────────────────────────────
# RUN PIPELINE
# ─────────────────────────────────────────────────────────────────
if "pipeline_data" not in st.session_state:
    st.session_state.pipeline_data = None

if run_btn or st.session_state.pipeline_data is None:
    with st.spinner("Training models… "):
        st.session_state.pipeline_data = run_pipeline(
            n_samples, test_size, smote_on, weight_on,
            log_transform_on, log_passes, use_best_params, int(seed)
        )

data        = st.session_state.pipeline_data
results     = data["results"]
FEATURES    = data["FEATURES"]
X_test_sc   = data["X_test_sc"]
X_train_sc  = data["X_train_sc"]
y_test      = data["y_test"]
best_models = data["best_models"]

COLORS = ["#4F46E5", "#22C55E", "#F97316"]
MODEL_NAMES = [r["model"] for r in results]

plt.rcParams.update({
    "figure.facecolor": "#ffffff", "axes.facecolor": "#ffffff",
    "axes.edgecolor":   "#3d4663", "axes.labelcolor": "#8892b0",
    "text.color": "#111827", "xtick.color": "#374151",
    "ytick.color": "#374151", "grid.color": "#2d3461",
    "grid.alpha": 0.5,  "legend.facecolor": "#ffffff",
    "legend.edgecolor": "#3d4663",
})

# ══════════════════════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown("# Passenger Satisfaction ML Dashboard")
    st.markdown("*Three-model classification pipeline with SMOTE, sample weighting, and full evaluation suite*")

<<<<<<< HEAD
    log_badge = f"✅ On ({log_passes} pass{'es' if log_passes > 1 else ''})" if log_transform_on else "❌ Off"
    hp_badge  = "✅ Best params" if use_best_params else "⚙️ Default params"
=======
    log_badge = "On" if log_transform_on else "Off"
    hp_badge  = "Best params" if use_best_params else "Default params"
>>>>>>> main
    st.markdown(f"**Log Transform:** {log_badge} &nbsp;|&nbsp; **Hyperparameters:** {hp_badge}")
    st.markdown("---")

    best_r = max(results, key=lambda r: r["f1"])
    st.metric("Best Model", best_r["model"].split("(")[0].strip())
    c1, c2, c3 = st.columns(3)
    c1.metric("Best F1",       f"{best_r['f1']:.4f}")
    c2.metric("Best AUC",      f"{best_r['auc']:.4f}")
    c3.metric("Best Accuracy", f"{best_r['accuracy']:.4f}")

    st.markdown("---")
    st.markdown('<div class="section-header"><b>All Models at a Glance</b></div>', unsafe_allow_html=True)
    metrics = ["accuracy", "precision", "recall", "f1", "auc", "specificity"]
    df_metrics = pd.DataFrame(
        [[r[m] for m in metrics] for r in results],
        columns=[m.title() for m in metrics], index=MODEL_NAMES
    )
    st.dataframe(
        df_metrics.style.format("{:.4f}").background_gradient(cmap="coolwarm", axis=None),
        use_container_width=True
    )

    st.markdown('<div class="section-header"><b>Model Comparison Bar Chart</b></div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(metrics)); w = 0.25
    for i, (r, color) in enumerate(zip(results, COLORS)):
        ax.bar(x + i * w, [r[m] for m in metrics], w, label=r["model"], color=color, alpha=0.85)
    ax.set_xticks(x + w)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylim(0, 1.1); ax.set_ylabel("Score")
    ax.set_title("Model Comparison Across Evaluation Metrics")
    ax.legend(); ax.grid(axis="y")
    st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════
# DATA & SMOTE
# ══════════════════════════════════════════════════════════════════
elif page == "Data & SMOTE":
    st.markdown("# Data Overview & SMOTE")
    df_res = data["df_res"]
    st.markdown('<div class="section-header"><b>Feature Distributions</b></div>', unsafe_allow_html=True)
    feature_sel = st.selectbox("Select feature to explore", FEATURES)
    delay_note = ""
    if log_transform_on and feature_sel in ("departure_delay_in_minutes", "arrival_delay_in_minutes"):
        delay_note = f" *(log1p x{log_passes})*"
    fig, ax = plt.subplots(figsize=(10, 4))
    for sat_val, color, label in [(0, "#F97316", "Unsatisfied"), (1, "#22C55E", "Satisfied")]:
        ax.hist(df_res[df_res["satisfaction"] == sat_val][feature_sel], bins=30, alpha=0.6, color=color, label=label)
    ax.set_xlabel(feature_sel + delay_note); ax.set_ylabel("Count")
    ax.set_title(f"Distribution of {feature_sel}{delay_note} by Satisfaction"); ax.legend()
    st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════
# MODEL METRICS
# ══════════════════════════════════════════════════════════════════
elif page == "Model Metrics":
    st.markdown("# Model Performance Metrics")

    st.markdown("---")
    st.markdown('<div class="section-header"><b>Drift Line Chart</b></div>', unsafe_allow_html=True)

    METRIC_OPTIONS = ["accuracy", "precision", "recall", "f1", "auc", "specificity"]
    selected_metrics = st.multiselect(
        "Select evaluation metrics to display on the drift chart",
        options=METRIC_OPTIONS,
        default=["f1", "auc"],
        help="Solid line = Train, hollow marker = Test"
    )

    if not selected_metrics:
        st.info("Select at least one metric above to display the drift chart.")
    else:
        def compute_metric(metric, y_true, y_pred, y_prob):
            if metric == "accuracy":    return accuracy_score(y_true, y_pred)
            if metric == "precision":   return precision_score(y_true, y_pred, zero_division=0)
            if metric == "recall":      return recall_score(y_true, y_pred, zero_division=0)
            if metric == "f1":          return f1_score(y_true, y_pred, zero_division=0)
            if metric == "auc":         return roc_auc_score(y_true, y_prob)
            if metric == "specificity":
                cm_ = confusion_matrix(y_true, y_pred)
                tn, fp = cm_.ravel()[:2]
                return tn / (tn + fp) if (tn + fp) > 0 else 0.0
            return 0.0

        train_scores = {m: [] for m in selected_metrics}
        test_scores  = {m: [] for m in selected_metrics}

        for r in results:
            model      = r["model_obj"]
            y_tr_pred  = model.predict(X_train_sc)
            y_tr_prob  = model.predict_proba(X_train_sc)[:, 1]
            y_te_pred  = model.predict(X_test_sc)
            y_te_prob  = r["prob"]
            for m in selected_metrics:
                train_scores[m].append(compute_metric(m, data["y_train"], y_tr_pred, y_tr_prob))
                test_scores[m].append(compute_metric(m, y_test, y_te_pred, y_te_prob))

        x = np.arange(len(MODEL_NAMES))
        palette = ["#4F46E5", "#22C55E", "#F97316", "#A855F7", "#0EA5E9", "#EAB308"]
        line_styles  = ["-", "--", "-.", ":", (0,(3,1,1,1)), (0,(5,2))]
        marker_shapes = ["o", "s", "D", "^", "v", "P"]

        fig, ax = plt.subplots(figsize=(10, 5))
        for idx, m in enumerate(selected_metrics):
            col = palette[idx % len(palette)]
            ls  = line_styles[idx % len(line_styles)]
            mk  = marker_shapes[idx % len(marker_shapes)]
            ax.plot(x, train_scores[m], marker=mk, linestyle=ls, color=col,
                    linewidth=2, label=f"{m.upper()} — Train", alpha=0.9)
            ax.plot(x, test_scores[m],  marker=mk, linestyle=ls, color=col,
                    linewidth=2, label=f"{m.upper()} — Test", alpha=0.5, markerfacecolor="none")
            for i in range(len(MODEL_NAMES)):
                ax.fill_between([i], [test_scores[m][i]], [train_scores[m][i]],
                                alpha=0.07, color=col)
        ax.set_xticks(x); ax.set_xticklabels(MODEL_NAMES, rotation=15)
        ax.set_ylabel("Score"); ax.set_ylim(0, 1.1)
        ax.set_title("Train vs Test Performance Drift\n(solid = train, hollow = test)")
        ax.legend(fontsize=8, ncol=2); ax.grid(True)
        st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════
# ROC & PR CURVES
# ══════════════════════════════════════════════════════════════════
elif page == "ROC & PR Curves":
    st.markdown("# ROC & Precision-Recall Curves")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header"><b>ROC Curves</b></div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        for r, color in zip(results, COLORS):
            fpr, tpr, _ = roc_curve(y_test, r["prob"])
            ax.plot(fpr, tpr, label=f"{r['model'].split('(')[0].strip()} (AUC={r['auc']:.3f})",
                    color=color, linewidth=2)
        ax.plot([0,1],[0,1],"w--",linewidth=1,alpha=0.5)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves"); ax.legend(fontsize=8); ax.grid(True)
        st.pyplot(fig); plt.close()

    with col2:
        st.markdown('<div class="section-header"><b>Precision-Recall Curves</b></div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        for r, color in zip(results, COLORS):
            prec, rec, _ = precision_recall_curve(y_test, r["prob"])
            ap = average_precision_score(y_test, r["prob"])
            ax.plot(rec, prec, label=f"{r['model'].split('(')[0].strip()} (AP={ap:.3f})",
                    color=color, linewidth=2)
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curves"); ax.legend(fontsize=8); ax.grid(True)
        st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown('<div class="section-header"><b>Threshold Explorer</b></div>', unsafe_allow_html=True)

    model_sel = st.selectbox("Select model", MODEL_NAMES)
    r_sel = next(r for r in results if r["model"] == model_sel)

    thresh = st.slider("Decision Threshold", 0.1, 0.9, 0.5, 0.01)

    y_pred_thresh = (r_sel["prob"] >= thresh).astype(int)

    precision = precision_score(y_test, y_pred_thresh, zero_division=0)
    recall = recall_score(y_test, y_pred_thresh, zero_division=0)
    f1 = f1_score(y_test, y_pred_thresh, zero_division=0)
    acc = accuracy_score(y_test, y_pred_thresh)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_thresh).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sensitivity", f"{recall:.4f}")
    c2.metric("Specificity", f"{specificity:.4f}")
    c3.metric("F1", f"{f1:.4f}")
    c4.metric("Accuracy", f"{acc:.4f}")

    fig, ax = plt.subplots(figsize=(10, 3))
    fpr, tpr, thresholds = roc_curve(y_test, r_sel["prob"])

    ax.plot(fpr, tpr, color=COLORS[MODEL_NAMES.index(model_sel)], linewidth=2)

    # Current threshold point that user updates
    idx = np.argmin(np.abs(thresholds - thresh))
    ax.scatter([fpr[idx]], [tpr[idx]], color="#ffd166", s=100, zorder=5,
            label=f"Selected threshold={thresh:.2f}")

    # Youden's J to calc best threshold
    sens = tpr
    spec = 1 - fpr
    youden_j = sens + spec - 1
    best_idx = np.argmax(youden_j)
    best_thresh = thresholds[best_idx]

    # Best threshold point to display
    ax.scatter([fpr[best_idx]], [tpr[best_idx]],
            color="cyan", s=100, zorder=5,
            label=f"Best threshold={best_thresh:.2f}")

    ax.plot([0,1],[0,1],"w--",alpha=0.5)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"ROC — {model_sel}")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)
    plt.close()

    st.caption(f"Best threshold (with Youden’s J): {best_thresh:.3f}, maximizes balance between Sensitivity and Specificity")

# ══════════════════════════════════════════════════════════════════
# CONFUSION MATRICES
# ══════════════════════════════════════════════════════════════════
elif page == "Confusion Matrices":
    st.markdown("# Confusion Matrices")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, r, color in zip(axes, results, COLORS):
        sns.heatmap(r["cm"], annot=True, fmt="d", cmap="Purples", ax=ax, linewidths=0.5, cbar=False)
        ax.set_title(r["model"], color=color)
        ax.set_xlabel("Predicted", color="#8892b0"); ax.set_ylabel("Actual", color="#8892b0")
        ax.set_xticklabels(["Unsatisfied", "Satisfied"], rotation=30)
        ax.set_yticklabels(["Unsatisfied", "Satisfied"], rotation=0)
    fig.suptitle("Confusion Matrices — All Models", color="#ccd6f6", fontsize=14)
    plt.tight_layout(); st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════
elif page == "Feature Importance":
    st.markdown("# Feature Importance (Permutation)")

    with st.spinner("Computing permutation importances…"):
        all_imps = []
        for r in results:
            perm = permutation_importance(
                r["model_obj"], X_test_sc, y_test,
                n_repeats=5, random_state=42, scoring="f1"
            )
            all_imps.append(perm.importances_mean)

    mean_imp = np.mean(all_imps, axis=0)
    top_idx = np.argsort(mean_imp)[::-1]

    st.markdown('<div class="section-header"><b>Per-Model Importance (Top 8)</b></div>', unsafe_allow_html=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, r, imp, color in zip(axes, results, all_imps, COLORS):
        idx = np.argsort(imp)[::-1][:8]
        ax.barh([FEATURES[i] for i in idx[::-1]], imp[idx[::-1]], color=color, alpha=0.9)
        ax.set_title(r["model"].split("(")[0].strip(), color=color)
        ax.set_xlabel("Mean Importance")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown('<div class="section-header"><b>Top 5 Features (Average)</b></div>', unsafe_allow_html=True)

    top5_feats = [FEATURES[i] for i in top_idx[:5]]
    top5_vals = mean_imp[top_idx[:5]]
    vals = top5_vals.tolist() + [top5_vals[0]]
    angles = np.linspace(0, 2 * np.pi, 5, endpoint=False).tolist() + [0]

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, vals, color="#4F46E5", linewidth=2)
    ax.fill(angles, vals, alpha=0.25, color="#4F46E5")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(top5_feats, fontsize=10)
    ax.set_title("Top 5 Features (Avg)", y=1.1, color="#ccd6f6")
    ax.set_facecolor("#1a1f35")
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.markdown('<div class="section-header"><b>Ranked Feature Table</b></div>', unsafe_allow_html=True)

    imp_df = pd.DataFrame({
        "Feature": FEATURES,
        "Avg Importance": mean_imp,
        **{r["model"].split("(")[0].strip(): all_imps[i] for i, r in enumerate(results)}
    }).sort_values("Avg Importance", ascending=False).reset_index(drop=True)

    imp_df.index += 1

    st.dataframe(
        imp_df.style.format({c: "{:.4f}" for c in imp_df.columns[1:]})
                    .background_gradient(subset=["Avg Importance"], cmap="coolwarm"),
        use_container_width=True
    )
# ══════════════════════════════════════════════════════════════════
# LIME EXPLANATIONS
# ══════════════════════════════════════════════════════════════════
elif page == "LIME Explanations":
    st.markdown("# LIME — Local Interpretable Model Explanations")

    col_ctrl1, col_ctrl2 = st.columns([1, 2])
    with col_ctrl1:
        instance_idx = st.number_input(
            "Test instance index",
            min_value=0, max_value=len(X_test_sc) - 1,
            value=0, step=1,
            help="Pick which test passenger to explain"
        )
        num_features = st.slider("Features to display", 5, len(FEATURES), 10)

    with col_ctrl2:
        true_label = y_test[instance_idx]
        label_str  = "Satisfied" if true_label == 1 else "Unsatisfied"
        st.markdown(f"""
<div style="background:#E7ECF6;
            border:1px solid #4F46E5;border-radius:12px;padding:20px;margin-top:8px;">
  <p style="color:#334155;margin:0;font-size:0.85rem;letter-spacing:.04em;">TRUE LABEL</p>
  <h2 style="color:#1E293B;margin:6px 0 4px 0;font-weight:700;">{label_str}</h2>
  <p style="color:#475569;margin:0;font-size:0.8rem;">Instance #{instance_idx} of {len(X_test_sc)}</p>
</div>""", unsafe_allow_html=True)

    # Build / cache LIME explainer
    cache_key = f"lime_explainer_{X_train_sc.shape}"
    if cache_key not in st.session_state:
        with st.spinner("Building LIME explainer…"):
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X_train_sc,
                feature_names=FEATURES,
                class_names=["Unsatisfied", "Satisfied"],
                mode="classification",
                random_state=42
            )
            st.session_state[cache_key] = explainer
    explainer = st.session_state[cache_key]

    instance = X_test_sc[instance_idx]

    # One tab per model
    tab_labels = []
    for r in results:
        tab_labels.append(r["model"])
    tabs = st.tabs(tab_labels)

    for tab, r, color in zip(tabs, results, COLORS):
        with tab:
            pred_probs = r["model_obj"].predict_proba(instance.reshape(1, -1))[0]
            pred_label = int(np.argmax(pred_probs))
            pred_str   = " Satisfied" if pred_label == 1 else "Unsatisfied"
            confidence = pred_probs[pred_label]

            mc1, mc2 = st.columns(2)
            mc1.metric("Prediction", pred_str)
            mc2.metric("Confidence", f"{confidence:.1%}")

            with st.spinner(f"Computing LIME for {r['model']}…"):
                exp = explainer.explain_instance(
                    instance,
                    r["model_obj"].predict_proba,
                    num_features=num_features,
                    top_labels=2
                )

            exp_list = exp.as_list(label=pred_label)
            sorted_pairs = sorted(exp_list, key=lambda x: abs(x[1]))
            feat_names_s = [p[0] for p in sorted_pairs]
            weights_s    = [p[1] for p in sorted_pairs]
            bar_colors   = ["#22C55E" if w > 0 else "#F97316" for w in weights_s]

            fig, ax = plt.subplots(figsize=(10, max(4, len(feat_names_s) * 0.55)))
            bars = ax.barh(feat_names_s, weights_s, color=bar_colors, alpha=0.85, edgecolor="#3d4663")
            ax.axvline(0, color="#111827", linewidth=1, alpha=0.6)
            ax.set_xlabel("LIME Weight  (positive → Satisfied, negative → Unsatisfied)")
            ax.set_title(
                f"LIME Explanation — {r['model']}\n"
                f"Instance #{instance_idx}  |  Predicted: {pred_str}  ({confidence:.1%})"
            )
            for bar, w in zip(bars, weights_s):
                xpos = w + (0.001 if w >= 0 else -0.001)
                ha   = "left" if w >= 0 else "right"
                ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                        f"{w:+.4f}", va="center", ha=ha, fontsize=8, color="#111827")
            ax.grid(axis="x", alpha=0.4)
            plt.tight_layout()
            st.pyplot(fig); plt.close()

            st.markdown("---")
            # Class probability mini-bar
            fig2, ax2 = plt.subplots(figsize=(8, 1.8))
            ax2.barh(["Unsatisfied"], [pred_probs[0]], color="#F97316", alpha=0.8, height=0.4)
            ax2.barh(["Satisfied"],   [pred_probs[1]], color="#22C55E", alpha=0.8, height=0.4)
            ax2.set_xlim(0, 1); ax2.set_xlabel("Probability"); ax2.set_title("Class Probabilities")
            ax2.axvline(0.5, color="white", linestyle="--", alpha=0.5)
            for i, (lbl, prob) in enumerate([("Unsatisfied", pred_probs[0]), ("Satisfied", pred_probs[1])]):
                ax2.text(prob + 0.01, i, f"{prob:.1%}", va="center", fontsize=9, color="#ccd6f6")
            ax2.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig2); plt.close()
