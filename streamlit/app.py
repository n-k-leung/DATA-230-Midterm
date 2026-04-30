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
    .main { background-color: #0f1117; }
    .stMetric {
        background: linear-gradient(135deg, #1e2130 0%, #262b3d 100%);
        border: 1px solid #3d4663;
        border-radius: 12px;
        padding: 16px;
    }
    .stMetric label { color: #8892b0 !important; font-size: 0.8rem !important; }
    .stMetric [data-testid="metric-value"] { color: #ccd6f6 !important; font-size: 1.5rem !important; }
    .section-header {
        background: linear-gradient(90deg, #1e2130, transparent);
        border-left: 4px solid #64ffda;
        padding: 8px 16px;
        margin: 24px 0 16px 0;
        border-radius: 0 8px 8px 0;
    }
    div[data-testid="stSidebar"] { background: #0d1117; }
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
# SYNTHETIC DATA
# ─────────────────────────────────────────────────────────────────
@st.cache_data
def generate_synthetic_data(n_samples=5000, seed=42):
    rng = np.random.RandomState(seed)
    n = n_samples
    gender        = rng.choice(["Male", "Female"], n)
    customer_type = rng.choice(["Loyal Customer", "disloyal Customer"], n, p=[0.7, 0.3])
    flight_class  = rng.choice(["Business", "Eco", "Eco Plus"], n, p=[0.45, 0.45, 0.1])
    age            = rng.randint(15, 75, n)
    flight_distance = rng.randint(50, 5000, n)
    dep_delay       = np.where(rng.random(n) < 0.2, rng.randint(1, 300, n), 0)
    arr_delay       = dep_delay + rng.randint(-5, 30, n).clip(0)
    ratings = {col: rng.randint(1, 6, n) for col in [
        "seat_comfort", "inflight_wifi_service", "food_and_drink",
        "inflight_entertainment", "ease_of_online_booking",
        "online_boarding", "online_support"
    ]}
    base = (
        (flight_class == "Business") * 0.3 +
        (customer_type == "Loyal Customer") * 0.2 +
        ratings["seat_comfort"] * 0.05 +
        ratings["inflight_entertainment"] * 0.05 +
        ratings["inflight_wifi_service"] * 0.04 +
        ratings["online_boarding"] * 0.04 +
        (dep_delay < 15) * 0.1 +
        rng.normal(0, 0.15, n)
    )
    satisfaction = (base > 0.6).astype(int)
    return pd.DataFrame({
        "gender": gender, "customer_type": customer_type, "class": flight_class,
        "age": age, "flight_distance": flight_distance,
        "departure_delay_in_minutes": dep_delay,
        "arrival_delay_in_minutes":   arr_delay,
        **ratings, "satisfaction": satisfaction
    })

# ─────────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────────
@st.cache_data
def run_pipeline(n_samples, test_size, smote_on, weight_on, log_transform_on, use_best_params, seed):
    FEATURES = [
        "flight_distance", "departure_delay_in_minutes", "arrival_delay_in_minutes",
        "seat_comfort", "inflight_wifi_service", "food_and_drink", "inflight_entertainment",
        "ease_of_online_booking", "online_boarding", "online_support",
        "age", "gender", "customer_type", "class"
    ]
    TARGET = "satisfaction"

    df = generate_synthetic_data(n_samples, seed)

    le_gender   = LabelEncoder()
    le_customer = LabelEncoder()
    le_class    = LabelEncoder()
    df["gender"]        = le_gender.fit_transform(df["gender"])
    df["customer_type"] = le_customer.fit_transform(df["customer_type"])
    df["class"]         = le_class.fit_transform(df["class"])

    # Log transformation of delay columns
    if log_transform_on:
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
    st.markdown("## ✈️ Dashboard Controls")
    st.markdown("---")

    st.markdown("### 📊 Dataset")
    n_samples = st.slider("Sample Size", 1000, 10000, 5000, 500)
    test_size = st.slider("Test Split (%)", 10, 40, 20, 5) / 100
    seed      = st.number_input("Random Seed", value=42, min_value=0, max_value=9999)

    st.markdown("### ⚙️ Pipeline Options")
    smote_on         = st.toggle("SMOTE Oversampling",    value=True)
    weight_on        = st.toggle("Domain Sample Weights", value=True)
    log_transform_on = st.toggle(
        "Log Transformation",
        value=False,
        help="Applies log1p to departure_delay_in_minutes and arrival_delay_in_minutes"
    )
    if log_transform_on:
        st.markdown("<small style='color:#8892b0'>↳ Applied to departure & arrival delay</small>",
                    unsafe_allow_html=True)

    st.markdown("### 🔧 Hyperparameter Tuning")
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
    run_btn = st.button("🚀 Run Pipeline", use_container_width=True, type="primary")

    st.markdown("### 📌 Navigation")
    page = st.radio("Go to", [
        "🏠 Overview",
        "📊 Data & SMOTE",
        "📈 Model Metrics",
    ], label_visibility="collapsed")

# ─────────────────────────────────────────────────────────────────
# RUN PIPELINE
# ─────────────────────────────────────────────────────────────────
if "pipeline_data" not in st.session_state:
    st.session_state.pipeline_data = None

if run_btn or st.session_state.pipeline_data is None:
    with st.spinner("Training models… ⏳"):
        st.session_state.pipeline_data = run_pipeline(
            n_samples, test_size, smote_on, weight_on,
            log_transform_on, use_best_params, int(seed)
        )

data        = st.session_state.pipeline_data
results     = data["results"]
FEATURES    = data["FEATURES"]
X_test_sc   = data["X_test_sc"]
X_train_sc  = data["X_train_sc"]
y_test      = data["y_test"]
best_models = data["best_models"]

COLORS      = ["#64ffda", "#ff6b9d", "#ffd166"]
MODEL_NAMES = [r["model"] for r in results]

plt.rcParams.update({
    "figure.facecolor": "#0f1117", "axes.facecolor": "#1a1f35",
    "axes.edgecolor":   "#3d4663", "axes.labelcolor": "#8892b0",
    "text.color":       "#ccd6f6", "xtick.color": "#8892b0",
    "ytick.color":      "#8892b0", "grid.color": "#2d3461",
    "grid.alpha": 0.5,  "legend.facecolor": "#1a1f35",
    "legend.edgecolor": "#3d4663",
})

# ══════════════════════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("# ✈️ Passenger Satisfaction ML Dashboard")
    st.markdown("*Three-model classification pipeline with SMOTE, sample weighting, and full evaluation suite*")

    log_badge = "✅ On" if log_transform_on else "❌ Off"
    hp_badge  = "✅ Best params" if use_best_params else "⚙️ Default params"
    st.markdown(f"**Log Transform:** {log_badge} &nbsp;|&nbsp; **Hyperparameters:** {hp_badge}")
    st.markdown("---")

    best_r = max(results, key=lambda r: r["f1"])
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🏆 Best Model",    best_r["model"].split("(")[0].strip())
    c2.metric("🎯 Best F1",       f"{best_r['f1']:.4f}")
    c3.metric("📡 Best AUC",      f"{best_r['auc']:.4f}")
    c4.metric("📦 Train Samples", f"{len(X_train_sc):,}")
    c5.metric("🧪 Test Samples",  f"{len(X_test_sc):,}")

    st.markdown("---")
    st.markdown('<div class="section-header"><b>All Models at a Glance</b></div>', unsafe_allow_html=True)
    metrics = ["accuracy", "precision", "recall", "f1", "auc", "specificity"]
    df_metrics = pd.DataFrame(
        [[r[m] for m in metrics] for r in results],
        columns=[m.title() for m in metrics], index=MODEL_NAMES
    )
    st.dataframe(
        df_metrics.style.format("{:.4f}").background_gradient(cmap="YlGnBu", axis=None),
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
elif page == "📊 Data & SMOTE":
    st.markdown("# 📊 Data Overview & SMOTE")
    df_res = data["df_res"]
    orig   = data["original_counts"]
    res_   = data["resampled_counts"]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header"><b>Class Distribution Before / After SMOTE</b></div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(2); w = 0.35
        ax.bar(x - w/2, orig, w, label="Original",    color="#64ffda", alpha=0.8)
        ax.bar(x + w/2, res_, w, label="After SMOTE", color="#ff6b9d", alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(["Unsatisfied", "Satisfied"])
        ax.legend(); ax.set_title("Class Imbalance Before and After SMOTE")
        for i, (o, r) in enumerate(zip(orig, res_)):
            ax.text(i - w/2, o + 10, str(o), ha="center", va="bottom", fontsize=8)
            ax.text(i + w/2, r + 10, str(r), ha="center", va="bottom", fontsize=8)
        st.pyplot(fig); plt.close()

    with col2:
        st.markdown('<div class="section-header"><b>Satisfaction by Loyalty × Class</b></div>', unsafe_allow_html=True)
        hm = df_res.pivot_table(index="loyalty_group", columns="class_group",
                                values="satisfaction", aggfunc="mean") * 100
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(hm, annot=True, cmap="RdYlGn", fmt=".1f", ax=ax,
                    linewidths=0.5, cbar_kws={"label": "Satisfaction %"})
        ax.set_title("Satisfaction Rate (%) by Loyalty & Class")
        st.pyplot(fig); plt.close()

    st.markdown("---")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-header"><b>Satisfaction by Loyalty</b></div>', unsafe_allow_html=True)
        gm = df_res.groupby("loyalty_group")["satisfaction"].mean() * 100
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(gm.index, gm.values, color=["#64ffda", "#ff6b9d"])
        ax.set_ylim(0, 100); ax.set_ylabel("Satisfaction Rate (%)")
        ax.set_title("Satisfaction by Customer Loyalty")
        for i, v in enumerate(gm.values): ax.text(i, v + 1, f"{v:.1f}%", ha="center")
        st.pyplot(fig); plt.close()

    with col4:
        st.markdown('<div class="section-header"><b>Satisfaction by Class</b></div>', unsafe_allow_html=True)
        cm2 = df_res.groupby("class_group")["satisfaction"].mean() * 100
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(cm2.index, cm2.values, color=["#ffd166", "#a78bfa"])
        ax.set_ylim(0, 100); ax.set_ylabel("Satisfaction Rate (%)")
        ax.set_title("Satisfaction by Cabin Class")
        for i, v in enumerate(cm2.values): ax.text(i, v + 1, f"{v:.1f}%", ha="center")
        st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown('<div class="section-header"><b>Feature Distributions</b></div>', unsafe_allow_html=True)
    feature_sel = st.selectbox("Select feature to explore", FEATURES)
    delay_note = ""
    if log_transform_on and feature_sel in ("departure_delay_in_minutes", "arrival_delay_in_minutes"):
        delay_note = " *(log1p transformed)*"
    fig, ax = plt.subplots(figsize=(10, 4))
    for sat_val, color, label in [(0, "#ff6b9d", "Unsatisfied"), (1, "#64ffda", "Satisfied")]:
        ax.hist(df_res[df_res["satisfaction"] == sat_val][feature_sel], bins=30, alpha=0.6, color=color, label=label)
    ax.set_xlabel(feature_sel + delay_note); ax.set_ylabel("Count")
    ax.set_title(f"Distribution of {feature_sel}{delay_note} by Satisfaction"); ax.legend()
    st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════
# MODEL METRICS
# ══════════════════════════════════════════════════════════════════
elif page == "📈 Model Metrics":
    st.markdown("# 📈 Model Performance Metrics")

    sort_metric = st.selectbox("Sort models by", ["f1", "accuracy", "precision", "recall", "auc", "specificity"])
    sorted_results = sorted(results, key=lambda r: r[sort_metric], reverse=True)

    for r, color in zip(sorted_results, COLORS):
        with st.expander(
            f"{'🥇' if r == sorted_results[0] else '📊'} {r['model']} — F1: {r['f1']:.4f}  |  AUC: {r['auc']:.4f}",
            expanded=(r == sorted_results[0])
        ):
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Accuracy",    f"{r['accuracy']:.4f}")
            c2.metric("Precision",   f"{r['precision']:.4f}")
            c3.metric("Recall",      f"{r['recall']:.4f}")
            c4.metric("F1 Score",    f"{r['f1']:.4f}")
            c5.metric("ROC AUC",     f"{r['auc']:.4f}")
            c6.metric("Specificity", f"{r['specificity']:.4f}")

    st.markdown("---")
    st.markdown('<div class="section-header"><b>Performance Heatmap</b></div>', unsafe_allow_html=True)
    metrics = ["accuracy", "precision", "recall", "f1", "auc", "specificity"]
    df_heat = pd.DataFrame(
        [[r[m] for m in metrics] for r in results],
        columns=[m.title() for m in metrics], index=MODEL_NAMES
    )
    fig, ax = plt.subplots(figsize=(10, 3))
    sns.heatmap(df_heat, annot=True, fmt=".4f", cmap="YlGnBu", ax=ax, vmin=0.5, vmax=1, linewidths=0.5)
    ax.set_title("Model Performance Heatmap")
    st.pyplot(fig); plt.close()

    # ── Drift chart with metric multiselect ──
    st.markdown("---")
    st.markdown('<div class="section-header"><b>Train vs Test Performance Drift</b></div>', unsafe_allow_html=True)

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
        palette      = ["#64ffda", "#ff6b9d", "#ffd166", "#a78bfa", "#fb923c", "#38bdf8"]
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
