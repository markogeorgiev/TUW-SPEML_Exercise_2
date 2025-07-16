import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from scipy.spatial.distance import cdist

def preprocess_data(synthetic_df, test_df):
    test_labels_raw = test_df["is_member"]
    test_labels = test_labels_raw.map({"yes": 1, "no": 0}).values
    test_df = test_df.drop(columns=["is_member"])

    # Keep only shared columns
    common_cols = synthetic_df.columns.intersection(test_df.columns).tolist()
    if not common_cols:
        raise ValueError("No common columns found between synthetic and test data!")

    synthetic_df = synthetic_df[common_cols].copy()
    test_df = test_df[common_cols].copy()

    combined = pd.concat([synthetic_df, test_df], axis=0)

    categorical_cols = combined.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = combined.select_dtypes(exclude=["object"]).columns.tolist()

    for col in categorical_cols:
        combined[col] = combined[col].fillna("missing")
    for col in numeric_cols:
        combined[col] = combined[col].fillna(0)

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ],
        remainder='passthrough'
    )
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    encoded_data = pipeline.fit_transform(combined)
    if hasattr(encoded_data, "toarray"):
        encoded_data = encoded_data.toarray()

    synthetic_encoded = encoded_data[:len(synthetic_df)]
    test_encoded = encoded_data[len(synthetic_df):]

    return synthetic_encoded, test_encoded, test_labels

def compute_min_distances(test_encoded, synthetic_encoded):
    distances = cdist(test_encoded, synthetic_encoded, metric='euclidean')
    min_distances = distances.min(axis=1)
    return min_distances

def scale_distances(distances):
    scaler = MinMaxScaler()
    return scaler.fit_transform(distances.reshape(-1, 1)).ravel()

def find_optimal_threshold(distances, labels):
    best_score = -np.inf
    best_thresh = 0
    for eps in np.linspace(0, 1, 1000):
        preds = (distances < eps).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        tpr = tp / (tp + fn + 1e-10)
        fpr = fp / (fp + tn + 1e-10)
        score = tpr - fpr
        if score > best_score:
            best_score = score
            best_thresh = eps
    return best_thresh

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
    eps_star = find_optimal_threshold(scaled_distances, test_labels)
    predictions = (scaled_distances < eps_star).astype(int)

    acc = accuracy_score(test_labels, predictions)
    auc = roc_auc_score(test_labels, scaled_distances)
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC AUC: {auc:.4f}")

    # Risk score: λ = 1 - distance
    risk_scores = 1 - scaled_distances
    test_df["predicted_is_member"] = predictions
    test_df["risk_score"] = risk_scores

    print("Results:")
    print(test_df[["predicted_is_member", "risk_score"]])

if __name__ == "__main__":
    main()
