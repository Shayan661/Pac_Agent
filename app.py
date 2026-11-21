from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.base import clone

# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="PAC Learning Agent V2",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 PAC Learning Agent V2")
st.caption(
    "Probably Approximately Correct Learning Simulator"
)

# -----------------------------------
# SIDEBAR SETTINGS
# -----------------------------------

st.sidebar.header("⚙️ PAC Configuration")

epsilon = st.sidebar.slider(
    "Epsilon (Maximum Error)",
    0.01, 0.50, 0.10
)

delta = st.sidebar.slider(
    "Delta (Failure Probability)",
    0.01, 0.50, 0.05
)

test_size = st.sidebar.slider(
    "Testing Data Percentage",
    10, 40, 25
)

st.sidebar.write("---")
st.sidebar.write(
    "**PAC Confidence:**",
    f"{(1 - delta) * 100:.1f}%"
)


@st.cache_data
def load_housing_csv(path: str = "housing (2).csv") -> pd.DataFrame | None:
    file_path = Path(path)
    if file_path.exists():
        return pd.read_csv(file_path)
    return None


# -----------------------------------
# DATASET MANAGEMENT
# -----------------------------------

st.header("1. Dataset Management")

dataset_source = st.radio(
    "Choose Dataset Source",
    options=[
        "Housing Dataset (housing (2).csv)",
        "Breast Cancer (Sample)",
        "Upload Custom CSV"
    ],
    horizontal=True
)

uploaded_file = None
if dataset_source == "Upload Custom CSV":
    uploaded_file = st.file_uploader(
        "Upload your CSV dataset",
        type=["csv"]
    )
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("CSV dataset uploaded successfully!")
    else:
        st.info("Upload a CSV file above, or switch to a built-in dataset.")
        st.stop()
elif dataset_source == "Housing Dataset (housing (2).csv)":
    df = load_housing_csv("housing (2).csv")
    if df is None:
        st.error("Could not find `housing (2).csv` in the working directory.")
        st.stop()
    st.success("Loaded `housing (2).csv` (20,640 rows, 10 columns).")
else:
    dataset = load_breast_cancer(as_frame=True)
    df = dataset.frame
    st.info("Using sample Breast Cancer dataset.")

default_target_idx = (
    df.columns.tolist().index("ocean_proximity")
    if "ocean_proximity" in df.columns
    else len(df.columns) - 1
)

target = st.selectbox(
    "Select Target Column",
    options=df.columns,
    index=default_target_idx
)

bin_continuous_target = False
n_bins = 3
if pd.api.types.is_numeric_dtype(df[target]) and df[target].nunique() > 10:
    st.warning(
        f"`{target}` is continuous ({df[target].nunique()} unique values). "
        "Classification requires discrete classes."
    )
    bin_continuous_target = st.checkbox(
        f"Convert `{target}` into quantile classes for classification",
        value=True
    )
    if bin_continuous_target:
        n_bins = st.slider("Number of Quantile Classes", 2, 5, 3)

st.write("Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

st.write(
    f"Rows: {df.shape[0]} | Columns: {df.shape[1]}"
)

# -----------------------------------
# DATA PREPARATION
# -----------------------------------

if st.button("🚀 Train PAC Learning Agent"):

    data = df.dropna(subset=[target]).copy()

    X = data.drop(columns=[target])
    y = data[target]

    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 10:
        if not bin_continuous_target:
            st.error(
                f"Target `{target}` has too many continuous values for classification. "
                "Enable quantile class binning above or pick a categorical target."
            )
            st.stop()
        labels_map = {
            2: ["Low", "High"],
            3: ["Low", "Medium", "High"],
            4: ["Q1 (Lowest)", "Q2", "Q3", "Q4 (Highest)"],
            5: ["Very Low", "Low", "Medium", "High", "Very High"]
        }
        y = pd.qcut(
            y,
            q=n_bins,
            labels=labels_map.get(n_bins, [f"Class_{i}" for i in range(n_bins)]),
            duplicates="drop"
        ).astype(str)

    if X.shape[1] == 0 or y.nunique() < 2:
        st.error("Dataset needs features and at least two target classes.")
        st.stop()

    if y.value_counts().min() < 2:
        st.error("Each target class needs at least two records.")
        st.stop()

    # Identify feature types
    numeric_cols = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_cols = X.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    transformers = []

    if numeric_cols:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        transformers.append(
            ("numeric", numeric_pipeline, numeric_cols)
        )

    if categorical_cols:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        transformers.append(
            ("categorical", categorical_pipeline, categorical_cols)
        )

    preprocessing = ColumnTransformer(transformers)

    # Split data
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size / 100,
            random_state=42,
            stratify=y
        )
    except ValueError:
        st.error(
            "Cannot create a stratified split. "
            "Check your dataset size and class distribution."
        )
        st.stop()

    if len(y_train) < 5 or len(y_test) < 2:
        st.error("Not enough training or testing records.")
        st.stop()

    # -----------------------------------
    # ML ALGORITHMS
    # -----------------------------------

    models = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=min(5, len(y_train))
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=2000
        )
    }

    results = []
    trained_models = {}
    predictions_store = {}

    for name, algorithm in models.items():

        pipeline = Pipeline([
            ("preprocessing", clone(preprocessing)),
            ("model", clone(algorithm))
        ])

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        error = 1 - accuracy

        # One-sided Hoeffding upper confidence bound
        n = len(y_test)

        margin = np.sqrt(
            np.log(1 / delta) / (2 * n)
        )

        upper_error = min(
            1.0, error + margin
        )

        results.append({
            "Algorithm": name,
            "Accuracy": accuracy * 100,
            "Test Error": error * 100,
            "Upper Error Bound": upper_error,
            "PAC-style Criterion": (
                "Satisfied"
                if upper_error <= epsilon
                else "Not Satisfied"
            )
        })

        trained_models[name] = pipeline
        predictions_store[name] = predictions

    results_df = pd.DataFrame(results)

    # -----------------------------------
    # PERFORMANCE DASHBOARD
    # -----------------------------------

    st.divider()
    st.header("2. Model Comparison")

    best_index = results_df["Accuracy"].idxmax()
    best_model = results_df.loc[best_index]

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Best Model",
        best_model["Algorithm"]
    )

    c2.metric(
        "Best Accuracy",
        f'{best_model["Accuracy"]:.2f}%'
    )

    c3.metric(
        "Test Samples",
        len(y_test)
    )

    st.dataframe(
        results_df.round(4),
        use_container_width=True
    )

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.bar(
        results_df["Algorithm"],
        results_df["Accuracy"]
    )

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("ML Algorithm Comparison")
    ax.set_ylim(0, 105)

    plt.xticks(rotation=15)
    st.pyplot(fig)
    plt.close(fig)

    # -----------------------------------
    # EPSILON VS DELTA ANALYSIS
    # -----------------------------------

    st.divider()
    st.header("3. Epsilon vs Delta Analysis")

    selected_model = st.selectbox(
        "Select model for PAC analysis",
        list(trained_models.keys())
    )

    selected_result = results_df[
        results_df["Algorithm"] == selected_model
    ].iloc[0]

    observed_error = selected_result["Test Error"] / 100

    epsilon_values = np.linspace(0.01, 0.50, 50)
    delta_values = np.linspace(0.01, 0.50, 50)

    E, D = np.meshgrid(
        epsilon_values,
        delta_values
    )

    upper_bound = np.minimum(
        1.0,
        observed_error +
        np.sqrt(np.log(1 / D) / (2 * len(y_test)))
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    contour = ax.contourf(
        E, D, upper_bound,
        levels=20,
        cmap="viridis"
    )

    plt.colorbar(
        contour,
        ax=ax,
        label="Upper Error Bound"
    )

    ax.set_xlabel("Epsilon")
    ax.set_ylabel("Delta")
    ax.set_title("PAC Parameter Analysis")

    st.pyplot(fig)
    plt.close(fig)

    current_upper = selected_result["Upper Error Bound"]

    if current_upper <= epsilon:
        st.success(
            f"{selected_model}: PAC-style criterion satisfied "
            f"for the selected epsilon and delta."
        )
    else:
        st.warning(
            f"{selected_model}: PAC-style criterion not satisfied."
        )

    st.write(
        f"Observed error: {observed_error:.4f}"
    )
    st.write(
        f"Upper error bound: {current_upper:.4f}"
    )

    # -----------------------------------
    # TRAINING SAMPLE SIZE ANALYSIS
    # -----------------------------------

    st.divider()
    st.header("4. Training Sample Size Analysis")

    sample_fractions = [0.2, 0.4, 0.6, 0.8, 1.0]

    sample_results = []

    for fraction in sample_fractions:

        sample_count = max(
            2,
            int(len(X_train) * fraction)
        )

        sample_count = min(
            sample_count, len(X_train)
        )

        try:
            xs, _, ys, _ = train_test_split(
                X_train,
                y_train,
                train_size=sample_count,
                random_state=42,
                stratify=y_train
            )
        except ValueError:
            xs = X_train.sample(
                n=sample_count,
                random_state=42
            )
            ys = y_train.loc[xs.index]

        for name, algorithm in models.items():

            temp_model = Pipeline([
                ("preprocessing", clone(preprocessing)),
                ("model", clone(algorithm))
            ])

            temp_model.fit(xs, ys)

            train_accuracy = accuracy_score(
                ys,
                temp_model.predict(xs)
            )

            test_accuracy = accuracy_score(
                y_test,
                temp_model.predict(X_test)
            )

            sample_results.append({
                "Samples": sample_count,
                "Algorithm": name,
                "Training Accuracy": train_accuracy,
                "Testing Accuracy": test_accuracy
            })

    sample_df = pd.DataFrame(sample_results)

    fig, ax = plt.subplots(figsize=(9, 5))

    for name in models:

        subset = sample_df[
            sample_df["Algorithm"] == name
        ]

        ax.plot(
            subset["Samples"],
            subset["Testing Accuracy"] * 100,
            marker="o",
            label=name
        )

    ax.set_xlabel("Training Samples")
    ax.set_ylabel("Testing Accuracy (%)")
    ax.set_title("Training Sample Size vs Accuracy")
    ax.legend()

    st.pyplot(fig)
    plt.close(fig)

    # -----------------------------------
    # LEARNING CURVE
    # -----------------------------------

    st.divider()
    st.header("5. Learning Curve Analysis")

    curve_model = st.selectbox(
        "Choose algorithm",
        list(models.keys()),
        key="curve_model"
    )

    curve_data = sample_df[
        sample_df["Algorithm"] == curve_model
    ]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        curve_data["Samples"],
        curve_data["Training Accuracy"] * 100,
        marker="o",
        label="Training Accuracy"
    )

    ax.plot(
        curve_data["Samples"],
        curve_data["Testing Accuracy"] * 100,
        marker="o",
        label="Testing Accuracy"
    )

    ax.set_xlabel("Training Samples")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Learning Curve: {curve_model}")
    ax.legend()

    st.pyplot(fig)
    plt.close(fig)

    # -----------------------------------
    # CONFUSION MATRIX
    # -----------------------------------

    st.divider()
    st.header("6. Confusion Matrix")

    class_labels = sorted(y.unique().tolist())

    cm = confusion_matrix(
        y_test,
        predictions_store[selected_model],
        labels=class_labels
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.imshow(cm, cmap="Blues")

    ax.set_title(selected_model)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    ax.set_xticks(range(len(class_labels)))
    ax.set_yticks(range(len(class_labels)))
    ax.set_xticklabels(class_labels, rotation=20, ha="right")
    ax.set_yticklabels(class_labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, cm[i, j],
                ha="center",
                va="center"
            )

    st.pyplot(fig)
    plt.close(fig)

    # -----------------------------------
    # DOWNLOAD REPORT
    # -----------------------------------

    st.divider()
    st.header("7. Download Evaluation Report")

    report = results_df.to_csv(index=False)

    st.download_button(
        label="⬇️ Download PAC Evaluation CSV",
        data=report,
        file_name="PAC_Evaluation_Report.csv",
        mime="text/csv"
    )

    st.success(
        "PAC Learning Agent analysis completed!"
    )

else:
    st.info(
        "Select a dataset or upload a CSV, "
        "choose a target column, and click Train PAC Learning Agent."
    ) 