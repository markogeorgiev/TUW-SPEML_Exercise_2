import argparse
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve

def preprocess_data(synthetic_df, test_df):
    test_labels_raw = test_df["is_member"]
    test_labels = test_labels_raw.map({"yes": 1, "no": 0}).values
    test_df = test_df.drop(columns=["is_member"])

    common_cols = synthetic_df.columns.intersection(test_df.columns).tolist()
    if not common_cols:
        raise ValueError("No common columns found between synthetic and test data!")

    synthetic_df = synthetic_df[common_cols].copy()
    test_df = test_df[common_cols].copy()

    combined = pd.concat([synthetic_df, test_df], axis=0)
    categorical_cols = combined.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = combined.select_dtypes(exclude=['object']).columns.tolist()

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

    synthetic_encoded = encoded_data[:len(synthetic_df)]
    test_encoded = encoded_data[len(synthetic_df):]

    return synthetic_encoded, test_encoded, test_labels


def compute_min_distances(test_records, synthetic_data):
    distances = euclidean_distances(test_records, synthetic_data)
    return np.min(distances, axis=1)

def main():
    parser = argparse.ArgumentParser(description="Distance-based MIA with Categorical Encoding")
    parser.add_argument("-s", "--synthetic", required=True, help="Path to synthetic CSV")
    parser.add_argument("-t", "--test", required=True, help="Path to test CSV with 'is_member'")
    args = parser.parse_args()

    synthetic_df = pd.read_csv(args.synthetic)
    test_df = pd.read_csv(args.test)

    synthetic_encoded, test_encoded, test_labels = preprocess_data(synthetic_df, test_df)

    min_distances = compute_min_distances(test_encoded, synthetic_encoded)
    scaled_distances = (min_distances - min_distances.min()) / (min_distances.max() - min_distances.min())

    fpr, tpr, thresholds = roc_curve(test_labels, -scaled_distances)
    optimal_idx = np.argmax(tpr - fpr)
    epsilon_star = thresholds[optimal_idx]
    predictions = (-(scaled_distances) >= epsilon_star).astype(int)

    auc = roc_auc_score(test_labels, -scaled_distances)
    asr = accuracy_score(test_labels, predictions)

    print("Results:")
    print(f"AUC: {auc:.4f}")
    print(f"Attack Success Rate: {asr:.4f}")
    print(f"Optimal ε*: {epsilon_star:.4f}")
    print("\nPredictions:")
    for i, (dist, pred, true) in enumerate(zip(min_distances, predictions, test_labels)):
        print(f"  Record {i+1}: distance = {dist:.4f}, predicted = {pred}, true = {true}")

if __name__ == "__main__":
    main()
