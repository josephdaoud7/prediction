import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from google.cloud import bigquery
import joblib

# --- Configuration ---
PROJECT_ID = "sevenrooms-datawarehouse"  # Your Google Cloud project ID
MODEL_FILENAME = "hot_cold_slot_model_v004.joblib"

# Initialize BigQuery Client
# This assumes you have authenticated with Google Cloud CLI (`gcloud auth application-default login`)
client = bigquery.Client(project=PROJECT_ID)


def fetch_training_data():
    """
    Executes the slot-level training query in BigQuery and returns a Pandas DataFrame.
    """
    print("Fetching slot-level training data from BigQuery (v004)...")
    try:
        with open("hot_cold_shift_training_data_v004.sql", "r") as f:
            training_sql = f.read()
        df = client.query(training_sql).to_dataframe()
        print(f"Successfully fetched {len(df)} rows for slot-level training.")
        return df
    except Exception as e:
        print(f"An error occurred while fetching slot-level training data: {e}")
        return None


def fetch_prediction_data():
    """
    Executes the slot-level prediction query in BigQuery and returns a Pandas DataFrame.
    """
    print("\nFetching slot-level data for prediction from BigQuery (v004)...")
    try:
        with open("hot_cold_shift_prediction_data_v004.sql", "r") as f:
            prediction_sql = f.read()
        df = client.query(prediction_sql).to_dataframe()
        print(f"Successfully fetched {len(df)} future slots to predict.")
        return df
    except Exception as e:
        print(f"An error occurred while fetching slot-level prediction data: {e}")
        return None


def _coerce_numerics(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def train_model(df: pd.DataFrame):
    """
    Trains a LightGBM model at the slot level and saves it to a file.
    """
    if df is None or df.empty:
        print("Training data is empty. Skipping training.")
        return None

    print("\nPreprocessing data for training...")

    numeric_columns = [
        "util_capacity",
        "slot_capacity",
        "covers_as_of_date",
        "day_prior",
        "day_of_week",
        "month",
        "year",
        "slot_index",
        "slot_start_time_min",
        "interval_minutes",
    ]
    df = _coerce_numerics(df, numeric_columns)
    df.dropna(subset=["slot_capacity", "covers_as_of_date"], inplace=True)
    print(f"Data preprocessed. {len(df)} rows remaining after cleaning.")

    features = [
        "day_prior",
        "day_of_week",
        "month",
        "year",
        "slot_index",
        "slot_start_time_min",
        "interval_minutes",
        "util_capacity",
        "slot_capacity",
        "covers_as_of_date",
    ]
    target = "is_hot"

    X = df[features]
    y = df[target]

    # Split data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Handle potential class imbalance
    try:
        class_counts = y_train.value_counts()
        scale_pos_weight = class_counts.get(0, 1) / max(class_counts.get(1, 1), 1)
        print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")
    except Exception:
        print("Warning: Could not compute scale_pos_weight; defaulting to 1")
        scale_pos_weight = 1

    lgbm = lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        colsample_bytree=0.8,
        subsample=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        scale_pos_weight=scale_pos_weight,
    )

    lgbm.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(100, verbose=True)],
    )

    # Evaluate
    print("\n--- Model Evaluation (Validation Set) ---")
    y_pred_val = lgbm.predict(X_val)
    y_proba_val = lgbm.predict_proba(X_val)[:, 1]
    print(f"Validation Accuracy: {accuracy_score(y_val, y_pred_val):.4f}")
    print(f"Validation AUC Score: {roc_auc_score(y_val, y_proba_val):.4f}")
    print("\nClassification Report (Validation Set):")
    print(classification_report(y_val, y_pred_val))

    # Save model
    print(f"\nSaving model to {MODEL_FILENAME}...")
    joblib.dump(lgbm, MODEL_FILENAME)
    print("Model saved successfully.")

    return lgbm


def predict_future_slots(prediction_df: pd.DataFrame):
    """
    Loads a trained model and predicts hot probability per future slot.
    """
    if prediction_df is None or prediction_df.empty:
        print("Prediction data is empty. Skipping prediction.")
        return None

    print("\nPreprocessing data for prediction...")
    numeric_columns = [
        "util_capacity",
        "slot_capacity",
        "covers_as_of_date",
        "day_prior",
        "day_of_week",
        "month",
        "year",
        "slot_index",
        "slot_start_time_min",
        "interval_minutes",
    ]
    prediction_df = _coerce_numerics(prediction_df, numeric_columns)
    prediction_df.fillna(0, inplace=True)

    print("\nLoading trained model for prediction...")
    try:
        model = joblib.load(MODEL_FILENAME)
    except FileNotFoundError:
        print(
            f"Error: Model file '{MODEL_FILENAME}' not found. Please train the model first."
        )
        return None

    print("Generating predictions for future slots...")

    features = [
        "day_prior",
        "day_of_week",
        "month",
        "year",
        "slot_index",
        "slot_start_time_min",
        "interval_minutes",
        "util_capacity",
        "slot_capacity",
        "covers_as_of_date",
    ]
    X_predict = prediction_df[features]

    probabilities = model.predict_proba(X_predict)[:, 1]
    prediction_df["hot_slot_probability"] = probabilities

    print("Predictions generated successfully.")
    return prediction_df


if __name__ == "__main__":
    # --- Step 1: Fetch training data and train the model ---
    training_df = fetch_training_data()
    if training_df is not None and not training_df.empty:
        out_path = "hot_slot_training_v004.csv"
        training_df.to_csv(out_path, index=False)
        print(f"Saved training dataset to '{out_path}' ({len(training_df)} rows)")
    train_model(training_df)

    # --- Step 2: Fetch data for future slots and make predictions ---
    future_slots_df = fetch_prediction_data()
    predictions = predict_future_slots(future_slots_df)

    if predictions is not None:
        print("\n--- Sample of Slot Predictions ---")
        print(
            predictions.sort_values(by="hot_slot_probability", ascending=False)[
                [
                    "shift_date",
                    "venue_key_id",
                    "persistent_id",
                    "shift_category",
                    "slot_index",
                    "slot_start_timestamp",
                    "slot_capacity",
                    "covers_as_of_date",
                    "hot_slot_probability",
                ]
            ].head(10)
        )

        # Save predictions to a CSV file for analysis
        predictions.to_csv("hot_slot_predictions.csv", index=False)
        print("\nPredictions saved to 'hot_slot_predictions.csv'")
