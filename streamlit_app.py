import streamlit as st
import os
import pandas as pd
import joblib
from src.config_manager import ConfigManager
from sklearn.metrics import (
                accuracy_score, precision_score, recall_score,
                f1_score, matthews_corrcoef,roc_auc_score,  confusion_matrix
            )

st.set_page_config(
    page_title="ML Classification",
    layout="wide"
)

st.title("Banking Customer Churn - ML Classification")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "outputs", "models")
CONFIG_DIR = os.path.join(BASE_DIR,"config")
config_manager = ConfigManager(CONFIG_DIR)
# -------------------------------------------------
# Load Models and Scaler (cached)
# -------------------------------------------------
#@st.cache_resource
def load_artifacts(output_dir):

    models = {
        "Logistic Regression": joblib.load(os.path.join(output_dir, "logisticregression.pkl")),
        "Decision Tree": joblib.load(os.path.join(output_dir, "decisiontree.pkl")),
        "KNN": joblib.load(os.path.join(output_dir, "knn.pkl")),
        "Random Forest": joblib.load(os.path.join(output_dir, "randomforest.pkl")),
        "XGBoost": joblib.load(os.path.join(output_dir, "xgboost.pkl")),
        "Gaussian": joblib.load(os.path.join(output_dir, "gaussiannb.pkl")),
    }

    # if XGBOOST_AVAILABLE:
    # models["XGBoost"] = joblib.load(
    #     os.path.join(os.path.join(output_dir, "xgboost.pkl"))
    # )

    scaler = joblib.load(os.path.join(output_dir,"StandardScaler.pkl"))
    return models, scaler

models, scaler = load_artifacts(MODEL_DIR)

tab_dataset,tab_single,tab_compare = st.tabs([
    " Dataset",
    " Single Model",
    " Model Comparison"
])

with tab_dataset:
    st.header(" Dataset Upload")
    MAX_FILE_SIZE =  10 
    st.markdown(f" **Upload CSV file (Max size: {MAX_FILE_SIZE} KB)**")
    uploaded_file = st.file_uploader(
        " ",
        type=["csv"],
        help=f"Max file size: {MAX_FILE_SIZE} KB"
    )
    MAX_FILE_SIZE = MAX_FILE_SIZE *  1024
    if uploaded_file:
        if uploaded_file.size > MAX_FILE_SIZE:
            st.error(f"File too large. Please upload a file under {MAX_FILE_SIZE} KB.")
            st.stop()
        df_initial = pd.read_csv(uploaded_file)
        drop_cols = config_manager.get_dataset_config()["drop_columns"]
        df = df_initial.drop(columns=[c for c in drop_cols if c in df_initial.columns], errors='ignore')

        if 'Gender' in df.columns:
            df['Gender'] = df['Gender'].map({
                "Male": 1,
                "Female": 0,
                "M": 1,
                "F": 0,
                "O":3,
                "Other":3,
                "U":3,
                "Unknown":3
            })
        
        if 'gender' in df.columns:
            df['gender'] = df['gender'].map({
                "Male": 1,
                "Female": 0,
                "M": 1,
                "F": 0,
                "O":3,
                "Other":3,
                "U":3,
                "Unknown":3
            })

        st.success("Dataset loaded successfully!")
        st.write("Shape:", df.shape)
        # st.write("Columns:", list(df.columns))

        st.dataframe(df.head())

        if df.shape[1] < 12:
            st.warning("Dataset has less than 12 features (assignment requirement)")
    else:
        SAMPLE_CSV_PATH = "data/Sample_download.csv"
        if os.path.exists(SAMPLE_CSV_PATH):
            with open(SAMPLE_CSV_PATH, "rb") as f:
                st.download_button(
                    label="Download Sample Dataset (CSV)",
                    data=f,
                    file_name="Sample_download.csv",
                    mime="text/csv"
                )
        else:
            st.error("Sample CSV file not found.")

with tab_single:
    st.header("Single Model Evaluation")
    if uploaded_file is None:
        st.info("Please upload a dataset first in the Dataset tab before reviewing Single Model.")
    else:
        

        if not models:
            st.error("No models loaded")
            st.stop()

        model_name = st.selectbox(
            "Select a model",
            list(models.keys())
        )

        if uploaded_file:
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]

            if scaler:
                X = scaler.transform(X)

            if st.button("Run Evaluation"):
                model = models[model_name]
                y_pred = model.predict(X)

                col1, col2, col3 = st.columns(3)
                col1.metric("Accuracy", accuracy_score(y, y_pred))
                col2.metric("Precision", precision_score(y, y_pred, average="weighted"))
                col3.metric("Recall", recall_score(y, y_pred, average="weighted"))

                col4, col5, col6 = st.columns(3)
                col4.metric("F1 Score", f1_score(y, y_pred, average="weighted"))
                col5.metric("MCC", matthews_corrcoef(y, y_pred))
                col6.metric("Samples", len(y))

                st.subheader("Confusion Matrix")
                
                targetColumn = config_manager.get_dataset_config()["target_column"]
                
                
                cm = confusion_matrix(y, y_pred)
                # st.write(cm)
                cm_df = pd.DataFrame(
                    cm,
                    index=[f"Actual {targetColumn} No", f"Actual {targetColumn} Yes"],
                    columns=[f"Predicted {targetColumn} No", f"Predicted {targetColumn} Yes"]
                )
                st.table(cm_df)
            

with tab_compare:
    st.header(" Model Comparison")
    if uploaded_file is None:
        st.info(" Please upload a dataset first in the Dataset tab before Checking Comparison.")
    else:
        

        if uploaded_file and st.button("Compare All Models"):
            results = []

            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]

            if scaler:
                X = scaler.transform(X)

            for name, model in models.items():
                y_pred = model.predict(X)

                # AUC handling (binary vs multiclass)
                try:
                    y_proba = model.predict_proba(X)
                    if len(set(y)) == 2:
                        auc = roc_auc_score(y, y_proba[:, 1])
                    else:
                        auc = roc_auc_score(y, y_proba, multi_class="ovr")
                except:
                    auc = None

                results.append({
                    "Model": name,
                    "Accuracy": accuracy_score(y, y_pred),
                    "AUC": auc,
                    "Precision": precision_score(y, y_pred, average="weighted", zero_division=0),
                    "Recall": recall_score(y, y_pred, average="weighted", zero_division=0),
                    "F1": f1_score(y, y_pred, average="weighted", zero_division=0),
                    "MCC": matthews_corrcoef(y, y_pred)
                })

            results_df = pd.DataFrame(results)
            # st.dataframe(results_df)
            st.dataframe(
                results_df.style.highlight_max(
                    subset=["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
                    color="lightgreen"
                ),
                use_container_width=True
            )

            st.subheader("Best Model per Metric")
            for metric in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]:
                best_row = results_df.loc[results_df[metric].idxmax()]
                st.success(f"{metric}: **{best_row['Model']}** ({best_row[metric]:.4f})")

            st.download_button(
                " Download Comparison Results",
                data=results_df.to_csv(index=False),
                file_name="model_comparison.csv",
                mime="text/csv"
            )

