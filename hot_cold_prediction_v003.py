import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from google.cloud import bigquery
import joblib

# --- Configuration ---
PROJECT_ID = "sevenrooms-datawarehouse"  # Your Google Cloud project ID
VENUE_ID = 5452140763283456
MODEL_FILENAME = "hot_cold_shift_model.joblib"

# Initialize BigQuery Client
# This assumes you have authenticated with Google Cloud CLI (`gcloud auth application-default login`)
client = bigquery.Client(project=PROJECT_ID)


def fetch_training_data():
    """
    Executes the training query in BigQuery and returns a Pandas DataFrame.
    """
    print("Fetching training data from BigQuery...")
    try:
        # NOTE: Ensure you have the latest SQL file named correctly in your directory
        with open("hot_cold_shift_training_data_v003.sql", "r") as f:
            training_sql = f.read()
        df = client.query(training_sql).to_dataframe()
        print(f"Successfully fetched {len(df)} rows for training.")
        return df
    except Exception as e:
        print(f"An error occurred while fetching training data: {e}")
        return None


def fetch_prediction_data():
    """
    Executes the prediction query in BigQuery and returns a Pandas DataFrame.
    """
    print("\nFetching data for prediction from BigQuery...")
    try:
        # NOTE: Ensure you have the latest SQL file named correctly in your directory
        with open("hot_cold_shift_prediction_data_v003.sql", "r") as f:
            prediction_sql = f.read()
        df = client.query(prediction_sql).to_dataframe()
        print(f"Successfully fetched {len(df)} future shifts to predict.")
        return df
    except Exception as e:
        print(f"An error occurred while fetching prediction data: {e}")
        return None


def train_model(df):
    """
    Trains a LightGBM model on the provided DataFrame and saves it to a file.
    """
    if df is None or df.empty:
        print("Training data is empty. Skipping training.")
        return None

    print("\nPreprocessing data for training...")

    # Convert data types
    df["util_capacity"] = pd.to_numeric(df["util_capacity"], errors="coerce")
    df["cumulative_covers_as_of_snapshot"] = pd.to_numeric(
        df["cumulative_covers_as_of_snapshot"], errors="coerce"
    )
    df.dropna(
        subset=["util_capacity", "cumulative_covers_as_of_snapshot"], inplace=True
    )
    print(f"Data preprocessed. {len(df)} rows remaining after cleaning.")

    print("\nStarting model training...")

    # Define features (X) and target (y)
    features = [
        "day_prior",
        "util_capacity",
        "day_of_week",
        "month",
        "year",
        "cumulative_covers_as_of_snapshot",
        "is_holiday",
    ]
    target = "is_hot"

    X = df[features]
    y = df[target]

    # Split data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Calculate scale_pos_weight for handling class imbalance
    try:
        scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
        print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")
    except IndexError:
        print(
            "Warning: Cannot calculate scale_pos_weight, only one class present in training data."
        )
        scale_pos_weight = 1

    # Initialize and train the LightGBM Classifier
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

    # Use early stopping to prevent overfitting
    lgbm.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(100, verbose=True)],
    )

    # --- Evaluate the model on the VALIDATION set (unseen data) ---
    print("\n--- Model Evaluation (Validation Set) ---")
    y_pred_val = lgbm.predict(X_val)
    y_proba_val = lgbm.predict_proba(X_val)[:, 1]

    print(f"Validation Accuracy: {accuracy_score(y_val, y_pred_val):.4f}")
    print(f"Validation AUC Score: {roc_auc_score(y_val, y_proba_val):.4f}")
    print("\nClassification Report (Validation Set):")
    print(classification_report(y_val, y_pred_val))

    # =================================================================
    # NEW SECTION: Evaluate the model on the TRAINING set
    # =================================================================
    print("\n--- Model Evaluation (Training Set) ---")
    y_pred_train = lgbm.predict(X_train)
    y_proba_train = lgbm.predict_proba(X_train)[:, 1]

    print(f"Training Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
    print(f"Training AUC Score: {roc_auc_score(y_train, y_proba_train):.4f}")
    print("\nClassification Report (Training Set):")
    print(classification_report(y_train, y_pred_train))
    # =================================================================

    # --- Save the trained model ---
    print(f"\nSaving model to {MODEL_FILENAME}...")
    joblib.dump(lgbm, MODEL_FILENAME)
    print("Model saved successfully.")

    return lgbm


def predict_future_shifts(prediction_df):
    """
    Loads a trained model and uses it to predict on new data.
    """
    if prediction_df is None or prediction_df.empty:
        print("Prediction data is empty. Skipping prediction.")
        return None

    print("\nPreprocessing data for prediction...")
    prediction_df["util_capacity"] = pd.to_numeric(
        prediction_df["util_capacity"], errors="coerce"
    )
    prediction_df["cumulative_covers_as_of_snapshot"] = pd.to_numeric(
        prediction_df["cumulative_covers_as_of_snapshot"], errors="coerce"
    )
    prediction_df.fillna(0, inplace=True)

    print("\nLoading trained model for prediction...")
    try:
        model = joblib.load(MODEL_FILENAME)
    except FileNotFoundError:
        print(
            f"Error: Model file '{MODEL_FILENAME}' not found. Please train the model first."
        )
        return None

    print("Generating predictions for future shifts...")

    features = [
        "day_prior",
        "util_capacity",
        "day_of_week",
        "month",
        "year",
        "cumulative_covers_as_of_snapshot",
        "is_holiday",
    ]
    X_predict = prediction_df[features]

    # Predict the probability of being 'hot' (class 1)
    probabilities = model.predict_proba(X_predict)[:, 1]

    # Add predictions to the DataFrame
    prediction_df["hot_shift_probability"] = probabilities

    print("Predictions generated successfully.")
    return prediction_df


if __name__ == "__main__":
    # --- Step 1: Fetch training data and train the model ---
    training_df = fetch_training_data()
    train_model(training_df)

    # --- Step 2: Fetch data for future shifts and make predictions ---
    future_shifts_df = fetch_prediction_data()
    predictions = predict_future_shifts(future_shifts_df)

    if predictions is not None:
        print("\n--- Sample of Predictions ---")
        # Display the top 10 shifts most likely to be 'hot'
        print(
            predictions.sort_values(by="hot_shift_probability", ascending=False).head(
                10
            )
        )

        # Save predictions to a CSV file for analysis
        predictions.to_csv("hot_shift_predictions.csv", index=False)
        print("\nPredictions saved to 'hot_shift_predictions.csv'")
