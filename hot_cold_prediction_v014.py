import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score, 
    precision_recall_curve, roc_curve, confusion_matrix
)
from google.cloud import bigquery
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
PROJECT_ID = "sevenrooms-datawarehouse"
MODEL_FILENAME = "hot_cold_shift_model_v014.joblib"

# Test mode flag - set to True to use sample data instead of BigQuery
TEST_MODE = True  # Enable test mode by default for demonstration

# Initialize BigQuery Client
if not TEST_MODE:
    try:
        client = bigquery.Client(project=PROJECT_ID)
        print("✓ BigQuery client initialized successfully")
    except Exception as e:
        print(f"ERROR: Failed to initialize BigQuery client: {e}")
        print("Setting TEST_MODE = True to use sample data instead")
        TEST_MODE = True


def create_sample_data():
    """Create sample training data for testing when BigQuery is not available."""
    print("Creating sample training data for testing...")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
    sample_dates = np.random.choice(dates, n_samples)
    
    # Convert to proper datetime objects
    sample_dates = [pd.to_datetime(d) for d in sample_dates]
    
    # Create snapshot timestamps (days before shift date)
    days_prior = np.random.randint(0, 30, n_samples)
    snapshot_timestamps = [sample_dates[i] - pd.Timedelta(days=int(days_prior[i])) for i in range(n_samples)]
    
    data = {
        'shift_date': sample_dates,
        'venue_key_id': [14301284] * n_samples,
        'persistent_id': np.random.choice(['dinner', 'lunch', 'brunch'], n_samples),
        'snapshot_timestamp': snapshot_timestamps,
        'shift_category': np.random.choice(['dinner', 'lunch', 'brunch'], n_samples),
        'day_prior': days_prior,
        'util_capacity': np.random.randint(50, 200, n_samples),
        'day_of_week': [d.weekday() + 1 for d in sample_dates],
        'month': [d.month for d in sample_dates],
        'year': [d.year for d in sample_dates],
        'covers_as_of_date': np.random.randint(0, 150, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create realistic is_hot based on utilization
    df['utilization'] = df['covers_as_of_date'] / df['util_capacity']
    df['is_hot'] = (df['utilization'] > 0.8).astype(int)
    df = df.drop('utilization', axis=1)
    
    print(f"✓ Created {len(df)} sample training rows")
    print(f"✓ Hot shifts: {df['is_hot'].sum()} ({100*df['is_hot'].mean():.1f}%)")
    
    return df


def prepare_features(df):
    """Apply consistent feature encoding for both training and prediction data."""
    df_encoded = df.copy()

    day_names = {
        1: "sunday", 2: "monday", 3: "tuesday", 4: "wednesday",
        5: "thursday", 6: "friday", 7: "saturday",
    }
    df_encoded["day_of_week"] = df_encoded["day_of_week"].map(day_names)

    month_names = {
        1: "january", 2: "february", 3: "march", 4: "april",
        5: "may", 6: "june", 7: "july", 8: "august",
        9: "september", 10: "october", 11: "november", 12: "december",
    }
    df_encoded["month"] = df_encoded["month"].map(month_names)

    return df_encoded


def fetch_training_data():
    """Fetch training data from BigQuery or create sample data in test mode."""
    global TEST_MODE
    
    if TEST_MODE:
        return create_sample_data()
    
    print("Fetching training data from BigQuery...")
    try:
        # Check if SQL file exists
        sql_file = "hot_cold_shift_training_data_v014.sql"
        try:
            with open(sql_file, "r") as f:
                training_sql = f.read()
            print(f"✓ Successfully read SQL file: {sql_file}")
        except FileNotFoundError:
            print(f"ERROR: SQL file not found: {sql_file}")
            print("Make sure the file exists in the current directory.")
            print("Switching to TEST_MODE with sample data...")
            TEST_MODE = True
            return create_sample_data()
        
        # Execute query
        print("Executing BigQuery...")
        query_job = client.query(training_sql)
        df = query_job.to_dataframe()
        
        if len(df) == 0:
            print("WARNING: Query returned 0 rows. Check your SQL query and data.")
            print("Switching to TEST_MODE with sample data...")
            TEST_MODE = True
            return create_sample_data()
        
        print(f"✓ Successfully fetched {len(df)} rows for training.")
        
        # Basic data validation
        required_columns = ['shift_date', 'util_capacity', 'covers_as_of_date', 'is_hot']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"ERROR: Missing required columns: {missing_cols}")
            print(f"Available columns: {list(df.columns)}")
            print("Switching to TEST_MODE with sample data...")
            TEST_MODE = True
            return create_sample_data()
            
        print(f"✓ Data validation passed")
        return df
        
    except Exception as e:
        print(f"ERROR fetching training data: {e}")
        print(f"Error type: {type(e).__name__}")
        print("Switching to TEST_MODE with sample data...")
        TEST_MODE = True
        return create_sample_data()


def fetch_prediction_data():
    """Fetch prediction data from BigQuery."""
    print("\nFetching data for prediction from BigQuery...")
    try:
        with open("hot_cold_shift_prediction_data_v014.sql", "r") as f:
            prediction_sql = f.read()
        df = client.query(prediction_sql).to_dataframe()
        print(f"Successfully fetched {len(df)} future shifts to predict.")
        return df
    except Exception as e:
        print(f"Error fetching prediction data: {e}")
        return None


def fetch_historical_test_data():
    """Fetch historical data for accuracy testing."""
    global TEST_MODE
    
    if TEST_MODE:
        print("\nCreating sample historical test data...")
        # Create a smaller sample for testing
        sample_df = create_sample_data()
        # Take only 200 rows for historical testing
        test_df = sample_df.sample(n=min(200, len(sample_df)), random_state=42)
        print(f"✓ Created {len(test_df)} sample historical test rows")
        return test_df
    
    print("\nFetching historical test data from BigQuery...")
    try:
        # Modify training query to get recent historical data
        with open("hot_cold_shift_training_data_v014.sql", "r") as f:
            training_sql = f.read()
        
        # Get data from last 90 days for testing
        test_sql = training_sql.replace(
            "WHERE DATE(s.DATE) < CURRENT_DATE()",
            "WHERE DATE(s.DATE) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY) AND DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)"
        )
        
        df = client.query(test_sql).to_dataframe()
        print(f"Successfully fetched {len(df)} rows for historical testing.")
        return df
    except Exception as e:
        print(f"Error fetching historical test data: {e}")
        print("Switching to sample data for historical testing...")
        TEST_MODE = True
        return fetch_historical_test_data()  # This will use TEST_MODE path


def create_accuracy_charts(y_true, y_pred, y_prob, model_name="Model"):
    """Create comprehensive accuracy visualization charts."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{model_name} - Accuracy Analysis', fontsize=16, fontweight='bold')
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
    axes[0,0].set_title('Confusion Matrix')
    axes[0,0].set_xlabel('Predicted')
    axes[0,0].set_ylabel('Actual')
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    axes[0,1].plot(fpr, tpr, color='darkorange', lw=2, 
                   label=f'ROC curve (AUC = {roc_auc:.3f})')
    axes[0,1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0,1].set_xlim([0.0, 1.0])
    axes[0,1].set_ylim([0.0, 1.05])
    axes[0,1].set_xlabel('False Positive Rate')
    axes[0,1].set_ylabel('True Positive Rate')
    axes[0,1].set_title('ROC Curve')
    axes[0,1].legend(loc="lower right")
    
    # 3. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    axes[0,2].plot(recall, precision, color='red', lw=2)
    axes[0,2].set_xlabel('Recall')
    axes[0,2].set_ylabel('Precision')
    axes[0,2].set_title('Precision-Recall Curve')
    axes[0,2].grid(True)
    
    # 4. Probability Distribution
    axes[1,0].hist(y_prob[y_true == 0], bins=30, alpha=0.7, label='Not Hot', color='blue')
    axes[1,0].hist(y_prob[y_true == 1], bins=30, alpha=0.7, label='Hot', color='red')
    axes[1,0].set_xlabel('Predicted Probability')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].set_title('Probability Distribution by Class')
    axes[1,0].legend()
    
    # 5. Calibration Plot
    prob_true, prob_pred = np.histogram(y_prob, bins=10, range=(0, 1))
    prob_true = prob_true.astype(float)
    
    bin_boundaries = np.linspace(0, 1, 11)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    accuracies = []
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            accuracies.append(accuracy_in_bin)
        else:
            accuracies.append(0)
    
    bin_centers = (bin_lowers + bin_uppers) / 2
    axes[1,1].plot(bin_centers, accuracies, 'o-', label='Model')
    axes[1,1].plot([0, 1], [0, 1], '--', label='Perfect Calibration')
    axes[1,1].set_xlabel('Mean Predicted Probability')
    axes[1,1].set_ylabel('Fraction of Positives')
    axes[1,1].set_title('Calibration Plot')
    axes[1,1].legend()
    
    # 6. Feature Importance (if model has this attribute)
    try:
        if hasattr(model, 'feature_importance'):
            importances = model.feature_importance()
            feature_names = model.feature_name_
            
            # Get top 10 features
            indices = np.argsort(importances)[::-1][:10]
            top_features = [feature_names[i] for i in indices]
            top_importances = [importances[i] for i in indices]
            
            axes[1,2].barh(range(len(top_features)), top_importances)
            axes[1,2].set_yticks(range(len(top_features)))
            axes[1,2].set_yticklabels(top_features)
            axes[1,2].set_xlabel('Importance')
            axes[1,2].set_title('Top 10 Feature Importances')
    except:
        axes[1,2].text(0.5, 0.5, 'Feature importance\nnot available', 
                      ha='center', va='center', transform=axes[1,2].transAxes)
        axes[1,2].set_title('Feature Importance')
    
    plt.tight_layout()
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_accuracy_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig


def calculate_accuracy_metrics(y_true, y_pred, y_prob):
    """Calculate comprehensive accuracy metrics."""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'AUC Score': roc_auc_score(y_true, y_prob),
        'Precision (Hot)': confusion_matrix(y_true, y_pred)[1,1] / (confusion_matrix(y_true, y_pred)[1,1] + confusion_matrix(y_true, y_pred)[0,1]) if confusion_matrix(y_true, y_pred)[1,1] + confusion_matrix(y_true, y_pred)[0,1] > 0 else 0,
        'Recall (Hot)': confusion_matrix(y_true, y_pred)[1,1] / (confusion_matrix(y_true, y_pred)[1,1] + confusion_matrix(y_true, y_pred)[1,0]) if confusion_matrix(y_true, y_pred)[1,1] + confusion_matrix(y_true, y_pred)[1,0] > 0 else 0,
        'True Positives': int(confusion_matrix(y_true, y_pred)[1,1]),
        'False Positives': int(confusion_matrix(y_true, y_pred)[0,1]),
        'True Negatives': int(confusion_matrix(y_true, y_pred)[0,0]),
        'False Negatives': int(confusion_matrix(y_true, y_pred)[1,0])
    }
    
    # Calculate F1 Score
    precision = metrics['Precision (Hot)']
    recall = metrics['Recall (Hot)']
    metrics['F1 Score'] = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return metrics


def train_model(df):
    """Train the LightGBM model with enhanced evaluation."""
    if df is None or df.empty:
        print("Training data is empty. Skipping training.")
        return None

    print("\nPreprocessing data for training...")
    
    # Data preprocessing
    df["util_capacity"] = pd.to_numeric(df["util_capacity"], errors="coerce")
    df["covers_as_of_date"] = pd.to_numeric(df["covers_as_of_date"], errors="coerce")
    df.dropna(subset=["util_capacity", "covers_as_of_date"], inplace=True)
    print(f"Data preprocessed. {len(df)} rows remaining after cleaning.")

    # Feature encoding
    df_encoded = prepare_features(df)
    print(f"Features after encoding: {list(df_encoded.columns)}")

    # Feature selection
    feature_columns = []
    for col in df_encoded.columns:
        if col not in ["shift_date", "venue_key_id", "persistent_id", "snapshot_timestamp", "is_hot", "shift_category"]:
            if pd.api.types.is_numeric_dtype(df_encoded[col]) or pd.api.types.is_string_dtype(df_encoded[col]):
                feature_columns.append(col)

    target = "is_hot"

    # Model configuration
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

    # Handle categorical features
    categorical_features = [col for col in feature_columns if pd.api.types.is_string_dtype(df_encoded[col])]
    if categorical_features:
        for col in categorical_features:
            df_encoded[col] = df_encoded[col].astype("category")

    # Create feature matrix and target
    X = df_encoded[feature_columns]
    y = df_encoded[target]

    # Categorical feature indices
    categorical_feature_indices = [X.columns.get_loc(col) for col in categorical_features if col in X.columns]

    print(f"Training with {len(feature_columns)} features: {feature_columns}")

    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Model training
    lgbm.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(100, verbose=True)],
        categorical_feature=categorical_feature_indices,
    )

    # Model evaluation
    print("\n" + "="*50)
    print("MODEL EVALUATION - VALIDATION SET")
    print("="*50)
    
    y_pred_val = lgbm.predict(X_val)
    y_proba_val = lgbm.predict_proba(X_val)[:, 1]
    
    # Calculate and display metrics
    val_metrics = calculate_accuracy_metrics(y_val, y_pred_val, y_proba_val)
    
    for metric, value in val_metrics.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
    
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred_val))
    
    # Create validation accuracy charts
    create_accuracy_charts(y_val, y_pred_val, y_proba_val, "Validation Set")

    # Save the model
    print(f"\nSaving model to {MODEL_FILENAME}...")
    joblib.dump(lgbm, MODEL_FILENAME)
    print("Model saved successfully.")

    # Store the model reference globally for later use
    global model
    model = lgbm
    
    return lgbm


def test_on_historical_data(trained_model):
    """Test the model on historical data for accuracy assessment."""
    print("\n" + "="*60)
    print("HISTORICAL ACCURACY TESTING")
    print("="*60)
    
    # Fetch historical test data
    historical_df = fetch_historical_test_data()
    if historical_df is None or historical_df.empty:
        print("No historical test data available.")
        return None
    
    # Preprocess historical data
    historical_df["util_capacity"] = pd.to_numeric(historical_df["util_capacity"], errors="coerce")
    historical_df["covers_as_of_date"] = pd.to_numeric(historical_df["covers_as_of_date"], errors="coerce")
    historical_df.dropna(subset=["util_capacity", "covers_as_of_date", "is_hot"], inplace=True)
    
    print(f"Testing on {len(historical_df)} historical data points...")
    
    # Prepare features
    historical_df_encoded = prepare_features(historical_df)
    
    # Handle categorical features
    categorical_features = [col for col in historical_df_encoded.columns if pd.api.types.is_string_dtype(historical_df_encoded[col])]
    for col in categorical_features:
        if col in historical_df_encoded.columns:
            historical_df_encoded[col] = historical_df_encoded[col].astype("category")
    
    # Feature selection (match training features)
    model_features = trained_model.feature_name_
    
    # Ensure all required features exist
    missing_features = set(model_features) - set(historical_df_encoded.columns)
    if missing_features:
        print(f"Adding missing features with zero values: {missing_features}")
        for feature in missing_features:
            historical_df_encoded[feature] = 0
    
    # Create feature matrix
    X_historical = historical_df_encoded[model_features]
    y_historical = historical_df_encoded["is_hot"]
    
    # Make predictions
    y_pred_historical = trained_model.predict(X_historical)
    y_proba_historical = trained_model.predict_proba(X_historical)[:, 1]
    
    # Calculate metrics
    historical_metrics = calculate_accuracy_metrics(y_historical, y_pred_historical, y_proba_historical)
    
    print("\nHistorical Test Results:")
    for metric, value in historical_metrics.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
    
    print(f"\nTotal Historical Shifts Analyzed: {len(y_historical)}")
    print(f"Actually Hot Shifts: {sum(y_historical)} ({100*sum(y_historical)/len(y_historical):.1f}%)")
    print(f"Predicted Hot Shifts: {sum(y_pred_historical)} ({100*sum(y_pred_historical)/len(y_historical):.1f}%)")
    
    # Create historical accuracy charts
    create_accuracy_charts(y_historical, y_pred_historical, y_proba_historical, "Historical Test Set")
    
    # Time-based analysis
    print("\n" + "-"*40)
    print("TIME-BASED ACCURACY ANALYSIS")
    print("-"*40)
    
    # Add predictions to historical dataframe for analysis
    historical_df['predicted_hot'] = y_pred_historical
    historical_df['hot_probability'] = y_proba_historical
    historical_df['correct_prediction'] = (historical_df['is_hot'] == historical_df['predicted_hot'])
    
    # Accuracy by day of week
    dow_accuracy = historical_df.groupby('day_of_week').agg({
        'correct_prediction': 'mean',
        'is_hot': ['count', 'sum'],
        'predicted_hot': 'sum'
    }).round(4)
    
    print("\nAccuracy by Day of Week:")
    print(dow_accuracy)
    
    # Accuracy by time horizon (days prior)
    if 'day_prior' in historical_df.columns:
        horizon_accuracy = historical_df.groupby(pd.cut(historical_df['day_prior'], bins=[0, 7, 30, 90, 365])).agg({
            'correct_prediction': 'mean',
            'is_hot': ['count', 'sum'],
            'predicted_hot': 'sum'
        }).round(4)
        
        print("\nAccuracy by Time Horizon:")
        print(horizon_accuracy)
    
    # Save historical test results
    historical_results = historical_df[['shift_date', 'venue_key_id', 'persistent_id', 'is_hot', 
                                      'predicted_hot', 'hot_probability', 'correct_prediction']].copy()
    historical_results.to_csv("historical_accuracy_test_results.csv", index=False)
    print(f"\nHistorical test results saved to 'historical_accuracy_test_results.csv'")
    
    return historical_metrics


def predict_future_shifts(prediction_df):
    """Make predictions on future shifts."""
    if prediction_df is None or prediction_df.empty:
        print("Prediction data is empty. Skipping prediction.")
        return None

    print("\nPreprocessing data for prediction...")
    
    # Data preprocessing
    prediction_df["util_capacity"] = pd.to_numeric(prediction_df["util_capacity"], errors="coerce")
    prediction_df["covers_as_of_date"] = pd.to_numeric(prediction_df["covers_as_of_date"], errors="coerce")
    prediction_df.fillna(0, inplace=True)

    # Feature preparation
    prediction_df_encoded = prepare_features(prediction_df)

    # Handle categorical features
    categorical_features = [col for col in prediction_df_encoded.columns if pd.api.types.is_string_dtype(prediction_df_encoded[col])]
    for col in categorical_features:
        if col in prediction_df_encoded.columns:
            prediction_df_encoded[col] = prediction_df_encoded[col].astype("category")

    # Load trained model
    try:
        trained_model = joblib.load(MODEL_FILENAME)
    except FileNotFoundError:
        print(f"Error: Model file '{MODEL_FILENAME}' not found. Please train the model first.")
        return None

    print("Generating predictions for future shifts...")

    # Ensure all required features exist
    missing_features = set(trained_model.feature_name_) - set(prediction_df_encoded.columns)
    if missing_features:
        print(f"Adding missing features with zero values: {missing_features}")
        for feature in missing_features:
            prediction_df_encoded[feature] = 0

    # Feature matrix
    X_predict = prediction_df_encoded[trained_model.feature_name_]

    # Make predictions
    probabilities = trained_model.predict_proba(X_predict)[:, 1]
    predictions = trained_model.predict(X_predict)

    # Add results to original dataframe
    prediction_df["hot_shift_probability"] = probabilities
    prediction_df["predicted_hot"] = predictions

    print("Predictions generated successfully.")
    return prediction_df


if __name__ == "__main__":
    print("="*60)
    print("HOT/COLD SHIFT PREDICTION MODEL v014")
    print("Enhanced with Historical Accuracy Testing")
    print("="*60)
    
    # Step 1: Fetch and train on training data
    print("\n1. TRAINING PHASE")
    training_df = fetch_training_data()
    
    if training_df is None:
        print("ERROR: Failed to fetch training data. Please check:")
        print("1. BigQuery credentials are properly configured")
        print("2. hot_cold_shift_training_data_v014.sql file exists")
        print("3. BigQuery permissions are correct")
        print("4. SQL query is valid")
        exit(1)
    
    if training_df.empty:
        print("ERROR: Training data is empty. Please check:")
        print("1. SQL query returns data")
        print("2. Venue ID and date filters are correct")
        print("3. Database contains relevant historical data")
        exit(1)
    
    print(f"✓ Successfully loaded {len(training_df)} training rows")
    print(f"✓ Date range: {training_df['shift_date'].min()} to {training_df['shift_date'].max()}")
    
    trained_model = train_model(training_df)
    
    if trained_model is None:
        print("ERROR: Model training failed. Check the error messages above.")
        exit(1)
    
    # Step 2: Test on historical data for accuracy assessment
    print("\n2. HISTORICAL ACCURACY TESTING")
    historical_metrics = test_on_historical_data(trained_model)
    
    # Step 3: Make predictions on future data
    print("\n3. FUTURE PREDICTIONS")
    future_shifts_df = fetch_prediction_data()
    predictions = predict_future_shifts(future_shifts_df)
    
    if predictions is not None:
        print("\n" + "="*40)
        print("PREDICTION RESULTS")
        print("="*40)
        
        # Display summary statistics
        total_shifts = len(predictions)
        predicted_hot = sum(predictions['predicted_hot'])
        avg_probability = predictions['hot_shift_probability'].mean()
        
        print(f"\nPrediction Summary:")
        print(f"Total future shifts analyzed: {total_shifts}")
        print(f"Predicted hot shifts: {predicted_hot} ({100*predicted_hot/total_shifts:.1f}%)")
        print(f"Average hot probability: {avg_probability:.3f}")
        
        # Show top hot predictions
        print(f"\nTop 10 shifts most likely to be HOT:")
        top_predictions = predictions.sort_values(by="hot_shift_probability", ascending=False).head(10)
        print(top_predictions[['shift_date', 'persistent_id', 'hot_shift_probability', 'predicted_hot']])
        
        # Save predictions
        predictions.to_csv("hot_shift_predictions_v014.csv", index=False)
        print(f"\nFuture predictions saved to 'hot_shift_predictions_v014.csv'")
        
        # Create prediction distribution chart
        plt.figure(figsize=(10, 6))
        plt.hist(predictions['hot_shift_probability'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Hot Shift Probability')
        plt.ylabel('Number of Shifts')
        plt.title('Distribution of Hot Shift Probabilities - Future Predictions')
        plt.axvline(x=0.5, color='red', linestyle='--', label='Decision Threshold (0.5)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('future_predictions_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("Generated files:")
    print(f"- {MODEL_FILENAME} (trained model)")
    print("- validation_set_accuracy_analysis.png")
    print("- historical_test_set_accuracy_analysis.png") 
    print("- historical_accuracy_test_results.csv")
    print("- hot_shift_predictions_v014.csv")
    print("- future_predictions_distribution.png")