"""Streamlit diabetes classifier with cached models and individual predictions.
Run: streamlit run diabetes_modified.py
Requires diabetes.csv in the same directory as this script.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, precision_score,
                             recall_score, f1_score, roc_curve, auc)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = Path(__file__).resolve().parent / "diabetes.csv"
LABELS = {0: "Non-Diabetes", 1: "Diabetes"}

@st.cache_data
def load_data(path, modified_time):
    data = pd.read_csv(path)
    if "Outcome" not in data.columns:
        raise ValueError("Dataset must contain an 'Outcome' column.")
    if data.isna().any().any():
        raise ValueError("Dataset contains missing values. Please clean it first.")
    if set(data["Outcome"].unique()) != {0, 1}:
        raise ValueError("Outcome must contain both classes 0 and 1.")
    return data

@st.cache_resource(show_spinner="Training model (only for new settings)...")
def train_model(path, modified_time, train_ratio, classifier, parameters):
    df = load_data(path, modified_time)
    x = df.drop(columns=["Outcome"])
    y = df["Outcome"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=train_ratio, random_state=42, stratify=y
    )
    if classifier == "Decision Tree":
        criterion, splitter = parameters
        model = DecisionTreeClassifier(criterion=criterion, splitter=splitter, random_state=42)
    elif classifier == "Logistic Regression":
        c, max_iter = parameters
        model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=max_iter, random_state=42))
    else:
        n_estimators, max_depth, bootstrap = parameters
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                       bootstrap=bootstrap, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train)
    return model, x_test, y_test

def show_evaluation(model, x_test, y_test, selected_charts):
    y_pred = model.predict(x_test)
    scores = [accuracy_score(y_test, y_pred), precision_score(y_test, y_pred, zero_division=0),
              recall_score(y_test, y_pred, zero_division=0), f1_score(y_test, y_pred, zero_division=0)]
    cols = st.columns(4)
    for col, name, score in zip(cols, ["Accuracy", "Precision", "Recall", "F1-score"], scores):
        col.metric(name, f"{score:.3f}")
    if "Confusion Matrix" in selected_charts:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, labels=[0, 1],
                                                 display_labels=list(LABELS.values()),
                                                 cmap="Blues", ax=ax)
        st.pyplot(fig)
        plt.close(fig)
    if "ROC Curve" in selected_charts:
        st.subheader("ROC Curve")
        if y_test.nunique() < 2:
            st.warning("ROC requires both classes in the test set.")
        else:
            probabilities = model.predict_proba(x_test)[:, list(model.classes_).index(1)]
            fpr, tpr, _ = roc_curve(y_test, probabilities)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.3f}")
            ax.plot([0, 1], [0, 1], "k--")
            ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

def show_single_prediction(model, features):
    st.subheader("Single Patient Prediction")
    st.caption("Enter one patient's feature values. This uses the cached model without retraining.")
    # Pre-fill with the first actual dataset record (features only).
    # Users can modify any field before submitting.
    example = features.iloc[0]
    st.caption("Default example: first row of diabetes.csv (editable).")
    with st.form("single_prediction"):
        columns = st.columns(2)
        values = {}
        for index, feature in enumerate(features.columns):
            series = features[feature]
            numeric = pd.to_numeric(series, errors="raise")
            initial = float(example[feature])
            with columns[index % 2]:
                if pd.api.types.is_integer_dtype(series):
                    values[feature] = st.number_input(feature, value=int(round(initial)), step=1)
                else:
                    values[feature] = st.number_input(feature, value=initial, format="%.4f")
        submitted = st.form_submit_button("Predict")
    if submitted:
        sample = pd.DataFrame([values], columns=features.columns)
        prediction = int(model.predict(sample)[0])
        probabilities = model.predict_proba(sample)[0]
        probability_by_class = dict(zip(model.classes_, probabilities))
        st.success(f"Predicted result: {LABELS[prediction]}")
        st.write(f"Non-Diabetes: {probability_by_class.get(0, 0):.2%} | Diabetes: {probability_by_class.get(1, 0):.2%}")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie([probability_by_class.get(0, 0), probability_by_class.get(1, 0)],
               labels=[LABELS[0], LABELS[1]], autopct="%1.1f%%", startangle=90)
        ax.set_title("Prediction Probability Distribution")
        ax.axis("equal")
        st.pyplot(fig)
        plt.close(fig)

def main():
    st.title("Predicting Diabetes Web App")
    st.markdown("Diabetes classification, evaluation and single-patient prediction")
    if not DATA_PATH.is_file():
        st.error(f"Cannot find {DATA_PATH.name}. Place it next to this Python script.")
        st.stop()
    try:
        modified_time = DATA_PATH.stat().st_mtime_ns
        df = load_data(str(DATA_PATH), modified_time)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        st.error(f"Cannot load dataset: {exc}")
        st.stop()
    st.sidebar.header("Model Selection Panel")
    classifier = st.sidebar.selectbox("Classifier", ["Decision Tree", "Logistic Regression", "Random Forest"])
    train_percent = st.sidebar.slider("Training data (%)", 60, 90, 70, 5)
    st.sidebar.caption(f"Train: {train_percent}% | Test: {100-train_percent}%")
    if classifier == "Decision Tree":
        criterion = st.sidebar.radio("Criterion", ["gini", "entropy"])
        splitter = st.sidebar.radio("Splitter", ["best", "random"])
        parameters = (criterion, splitter)
    elif classifier == "Logistic Regression":
        c = st.sidebar.number_input("C (regularization)", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
        max_iter = st.sidebar.slider("Maximum iterations", 100, 1000, 500, 50)
        parameters = (float(c), int(max_iter))
    else:
        n_estimators = st.sidebar.slider("Number of trees", 50, 500, 100, 10)
        max_depth = st.sidebar.slider("Maximum depth", 1, 30, 5)
        bootstrap = st.sidebar.checkbox("Bootstrap", value=True)
        parameters = (n_estimators, max_depth, bootstrap)
    selected_charts = st.sidebar.multiselect("Charts", ["Confusion Matrix", "ROC Curve"],
                                              default=["Confusion Matrix", "ROC Curve"])
    if st.sidebar.checkbox("Show raw data"):
        st.subheader("Diabetes Raw Dataset")
        st.dataframe(df)
    try:
        model, x_test, y_test = train_model(str(DATA_PATH), modified_time,
                                             train_percent / 100, classifier, parameters)
    except (ValueError, RuntimeError) as exc:
        st.error(f"Model training failed: {exc}")
        st.stop()
    st.subheader(f"{classifier} Results")
    st.caption(f"Training samples: {len(df)-len(x_test)} | Testing samples: {len(x_test)}")
    show_evaluation(model, x_test, y_test, selected_charts)
    st.divider()
    show_single_prediction(model, df.drop(columns=["Outcome"]))
    st.info("For educational use only; this model is not a medical diagnosis.")

if __name__ == "__main__":
    main()
