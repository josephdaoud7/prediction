import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from google.cloud import bigquery
import joblib

# --- Configuration ---
PROJECT_ID = "sevenrooms-datawarehouse"  # Your Google Cloud project ID
MODEL_FILENAME = "hot_cold_shift_model.joblib"

# Initialize BigQuery Client
# This assumes you have authenticated with Google Cloud CLI (`gcloud auth application-default login`)
client = bigquery.Client(project=PROJECT_ID)


def prepare_features(df):
    df_encoded = df.copy()

    day_names = {
        1: "sunday",
        2: "monday",
        3: "tuesday",
        4: "wednesday",
        5: "thursday",
        6: "friday",
        7: "saturday",
    }
    df_encoded["day_of_week"] = df_encoded["day_of_week"].map(day_names)

    month_names = {
        1: "january",
        2: "february",
        3: "march",
        4: "april",
        5: "may",
        6: "june",
        7: "july",
        8: "august",
        9: "september",
        10: "october",
        11: "november",
        12: "december",
    }
    df_encoded["month"] = df_encoded["month"].map(month_names)

    return df_encoded


def fetch_training_data():
    """
    Executes the training query in BigQuery and returns a Pandas DataFrame.
    """
    print("Fetching training data from BigQuery...")
    try:
        with open("hot_cold_shift_training_data_v002.sql", "r") as f:
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
        with open("hot_cold_shift_prediction_data_v002.sql", "r") as f:
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

    # convert types
    df["util_capacity"] = pd.to_numeric(df["util_capacity"], errors="coerce")
    df["covers_as_of_date"] = pd.to_numeric(df["covers_as_of_date"], errors="coerce")

    # drop rows where conversion might have failed (created NaNs)
    df.dropna(subset=["util_capacity", "covers_as_of_date"], inplace=True)
    print(f"Data preprocessed. {len(df)} rows remaining after cleaning.")

    print("Applying one-hot encoding to categorical features...")
    df_encoded = prepare_features(df)
    print(f"Features after encoding: {list(df_encoded.columns)}")

    print("\nStarting model training...")

    feature_columns = []
    for col in df_encoded.columns:
        if col not in [
            "shift_date",
            "venue_key_id",
            "persistent_id",
            "as_of_date",
            "is_hot",
            "shift_category",
        ]:
            if pd.api.types.is_numeric_dtype(
                df_encoded[col]
            ) or pd.api.types.is_string_dtype(df_encoded[col]):
                feature_columns.append(col)
            else:
                print(f"Excluding column: {col} (dtype: {df_encoded[col].dtype})")

    target = "is_hot"

    # Initialize LightGBM Classifier
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
    )

    # Identify categorical columns
    categorical_features = []
    for col in feature_columns:
        if pd.api.types.is_string_dtype(df_encoded[col]):
            categorical_features.append(col)

    if categorical_features:
        print(f"Categorical features: {categorical_features}")
        # Convert categorical columns to category dtype for LightGBM
        for col in categorical_features:
            df_encoded[col] = df_encoded[col].astype("category")

    # Create feature matrix and target
    X = df_encoded[feature_columns]
    y = df_encoded[target]

    # Get the indices of categorical features for LightGBM
    categorical_feature_indices = []
    for col in categorical_features:
        if col in X.columns:
            categorical_feature_indices.append(X.columns.get_loc(col))

    print(f"Categorical feature indices: {categorical_feature_indices}")
    print(f"Training with {len(feature_columns)} features: {feature_columns}")

    # Split data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Use early stopping to prevent overfitting
    lgbm.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(100, verbose=True)],
        categorical_feature=categorical_feature_indices,
    )

    # --- Evaluate the model ---
    print("\n--- Model Evaluation ---")
    y_pred_val = lgbm.predict(X_val)
    y_proba_val = lgbm.predict_proba(X_val)[:, 1]

    print(f"Validation Accuracy: {accuracy_score(y_val, y_pred_val):.4f}")
    print(f"Validation AUC Score: {roc_auc_score(y_val, y_proba_val):.4f}")
    print("\nClassification Report (Validation Set):")
    print(classification_report(y_val, y_pred_val))

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
    # --- FIX: Convert data types to match training data ---
    prediction_df["util_capacity"] = pd.to_numeric(
        prediction_df["util_capacity"], errors="coerce"
    )
    # Rename the column to match training data
    if "covers_as_of_date" in prediction_df.columns:
        prediction_df = prediction_df.rename(
            columns={"covers_as_of_date": "covers_as_of_date"}
        )

    prediction_df["covers_as_of_date"] = pd.to_numeric(
        prediction_df["covers_as_of_date"], errors="coerce"
    )
    # Fill any potential missing values after conversion with 0
    prediction_df.fillna(0, inplace=True)

    # --- NEW: Apply the same feature preparation as training ---
    print("Applying descriptive categorical encoding to prediction data...")
    prediction_df_encoded = prepare_features(prediction_df)
    print(f"Prediction features after encoding: {list(prediction_df_encoded.columns)}")

    # Convert categorical columns to category dtype for LightGBM (same as training)
    categorical_features = []
    for col in prediction_df_encoded.columns:
        if pd.api.types.is_string_dtype(prediction_df_encoded[col]):
            categorical_features.append(col)

    if categorical_features:
        print(f"Categorical features in prediction data: {categorical_features}")
        for col in categorical_features:
            prediction_df_encoded[col] = prediction_df_encoded[col].astype("category")

    print("\nLoading trained model for prediction...")
    try:
        model = joblib.load(MODEL_FILENAME)
    except FileNotFoundError:
        print(
            f"Error: Model file '{MODEL_FILENAME}' not found. Please train the model first."
        )
        return None

    print("Generating predictions for future shifts...")

    model_features = []
    for col in prediction_df_encoded.columns:
        if col not in [
            "shift_date",
            "venue_key_id",
            "persistent_id",
            "as_of_date",
            "shift_category",
        ]:
            if pd.api.types.is_numeric_dtype(
                prediction_df_encoded[col]
            ) or pd.api.types.is_string_dtype(prediction_df_encoded[col]):
                model_features.append(col)
            else:
                print(
                    f"Excluding column: {col} (dtype: {prediction_df_encoded[col].dtype})"
                )

    # check if missing feature
    missing_features = set(model.feature_name_) - set(model_features)
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
        # Add missing features with 0 values
        for feature in missing_features:
            prediction_df_encoded[feature] = 0

    # Ensure categorical features are properly encoded
    for col in model.feature_name_:
        if col in prediction_df_encoded.columns and pd.api.types.is_string_dtype(
            prediction_df_encoded[col]
        ):
            prediction_df_encoded[col] = prediction_df_encoded[col].astype("category")

    # Ensure columns are in the same order as training
    X_predict = prediction_df_encoded[model.feature_name_]

    # Predict the probability of being 'hot' (class 1)
    probabilities = model.predict_proba(X_predict)[:, 1]

    # Add predictions to the original DataFrame (not the encoded one)
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
