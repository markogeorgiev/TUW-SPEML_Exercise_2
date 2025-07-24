import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, recall_score, roc_curve
from scipy.spatial.distance import cdist

def preprocess_data(synthetic_df, test_df, sentinel_numeric=-9999):
    test_labels_raw = test_df["is_member"]
    test_labels = test_labels_raw.map({"yes": 1, "no": 0}).values
    test_df = test_df.drop(columns=["is_member"])

    common_cols = synthetic_df.columns.intersection(test_df.columns).tolist()
    if not common_cols:
        raise ValueError("No common columns found between synthetic and test data!")

    synthetic_df = synthetic_df[common_cols].copy()
    test_df = test_df[common_cols].copy()

    categorical_cols = test_df.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = test_df.select_dtypes(include=[np.number]).columns.tolist()

    test_df[categorical_cols] = test_df[categorical_cols].fillna("missing")
    test_df[numeric_cols] = test_df[numeric_cols].fillna(sentinel_numeric)

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ],
        remainder='passthrough'
    )
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    pipeline.fit(test_df)  # fit only on test data

    synthetic_df[categorical_cols] = synthetic_df[categorical_cols].fillna("missing")
    synthetic_df[numeric_cols] = synthetic_df[numeric_cols].fillna(sentinel_numeric)

    synthetic_encoded = pipeline.transform(synthetic_df)
    test_encoded = pipeline.transform(test_df)

    if hasattr(synthetic_encoded, "toarray"):
        synthetic_encoded = synthetic_encoded.toarray()
    if hasattr(test_encoded, "toarray"):
        test_encoded = test_encoded.toarray()

    return synthetic_encoded, test_encoded, test_labels

def compute_min_distances(test_encoded, synthetic_encoded):
    distances = cdist(test_encoded, synthetic_encoded, metric='euclidean')
    min_distances = distances.min(axis=1)
    return min_distances

def scale_distances(distances):
    scaler = MinMaxScaler()
    return scaler.fit_transform(distances.reshape(-1, 1)).ravel()

def main():
    parser = argparse.ArgumentParser(description="Distance-Based Membership Inference Attack")
    parser.add_argument("-s", "--synthetic", required=True, help="Path to synthetic data CSV")
    parser.add_argument("-t", "--test", required=True, help="Path to test data CSV (with 'is_member')")
    args = parser.parse_args()

    synthetic_df = pd.read_csv(args.synthetic)
    test_df = pd.read_csv(args.test)

    synthetic_encoded, test_encoded, test_labels = preprocess_data(synthetic_df, test_df)
    raw_distances = compute_min_distances(test_encoded, synthetic_encoded)
    scaled_distances = scale_distances(raw_distances)

    fpr, tpr, thresholds = roc_curve(test_labels, scaled_distances)
    optimal_idx = np.argmax(tpr - fpr)
    epsilon_star = (thresholds[optimal_idx])
    
    predictions = (scaled_distances >= epsilon_star).astype(int)
   
    acc = accuracy_score(test_labels, predictions)
    auc = roc_auc_score(test_labels, scaled_distances)
    recall = recall_score(test_labels, predictions)
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC AUC: {auc:.4f}")
    print(f"Recall: {recall:.4f}")

    # Risk score: λ = 1 - distance
    risk_scores = 1 - scaled_distances
    test_df["prediction"] = predictions
    test_df["predicted_is_correct"] = predictions == test_labels
    test_df["risk_score"] = risk_scores

    print("Results:")
    print(test_df[["prediction","predicted_is_correct", "risk_score"]])

if __name__ == "__main__":
    main()
