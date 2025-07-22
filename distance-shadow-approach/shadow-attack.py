import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, precision_score, recall_score
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

def preprocess_data(synthetic_df, test_df, sentinel_numeric=-9999):
    # Extract and encode test labels
    test_labels_raw = test_df["is_member"]
    test_labels = test_labels_raw.map({"yes": 1, "no": 0}).values
    test_df = test_df.drop(columns=["is_member"])

    # Align columns: keep only intersection
    common_cols = synthetic_df.columns.intersection(test_df.columns).tolist()
    if not common_cols:
        raise ValueError("No common columns found between synthetic and test data!")

    synthetic_df = synthetic_df[common_cols].copy()
    test_df = test_df[common_cols].copy()

    # Identify categorical and numeric columns based on test_df only
    categorical_cols = test_df.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = test_df.select_dtypes(include=[np.number]).columns.tolist()

    # Fill missing values in test_df ONLY
    test_df[categorical_cols] = test_df[categorical_cols].fillna("missing")
    test_df[numeric_cols] = test_df[numeric_cols].fillna(sentinel_numeric)

    # Fit encoder only on test data (to avoid info leak)
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ],
        remainder='passthrough'
    )
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    pipeline.fit(test_df)  # fit only on test data

    # Now apply the same missing value strategy to synthetic data
    synthetic_df[categorical_cols] = synthetic_df[categorical_cols].fillna("missing")
    synthetic_df[numeric_cols] = synthetic_df[numeric_cols].fillna(sentinel_numeric)

    # Transform both datasets
    synthetic_encoded = pipeline.transform(synthetic_df)
    test_encoded = pipeline.transform(test_df)

    # Convert sparse to dense if necessary
    if hasattr(synthetic_encoded, "toarray"):
        synthetic_encoded = synthetic_encoded.toarray()
    if hasattr(test_encoded, "toarray"):
        test_encoded = test_encoded.toarray()

    return synthetic_encoded, test_encoded, test_labels


def flatten_encoded_matrix(matrix):
    summary = {
        "mean": matrix.mean(axis=0),
        "median": np.median(matrix, axis=0),
        "var": matrix.var(axis=0)
    }
    # Concatenate all flattened summaries
    return np.concatenate([summary["mean"], summary["median"], summary["var"]])

def shadow_model_attack_all(synthetic_encoded, test_encoded, test_labels):
    print("🧪 Running shadow model attack on all test records...")

    preds = []
    confidence = []

    for i in range(len(test_encoded)):
        target = test_encoded[i:i+1]
        rest = np.delete(test_encoded, i, axis=0)

        # Simulate synthetic data generation (sampling with replacement)
        with_target = np.vstack([rest, target])
        shadow_plus = with_target[np.random.choice(with_target.shape[0], synthetic_encoded.shape[0], replace=True)]

        shadow_minus = rest[np.random.choice(rest.shape[0], synthetic_encoded.shape[0], replace=True)]

        # Flatten
        f_plus = flatten_encoded_matrix(shadow_plus)
        f_minus = flatten_encoded_matrix(shadow_minus)
        X_attack = np.vstack([f_plus, f_minus])
        y_attack = np.array([1, 0])

        # Flatten real synthetic data
        real_flat = flatten_encoded_matrix(synthetic_encoded).reshape(1, -1)

        classifiers = [
            RandomForestClassifier(),
            LogisticRegression(max_iter=1000),
            GaussianNB(),
            SVC(probability=True),
            KNeighborsClassifier(n_neighbors=1)
        ]

        confidences = []
        for clf in classifiers:
            clf.fit(X_attack, y_attack)
            prob = clf.predict_proba(real_flat)[0][1]
            confidences.append(prob)

        avg_conf = np.mean(confidences)
        pred = int(avg_conf > 0.5)

        preds.append(pred)
        confidence.append(avg_conf)

        print(f"Record {i}: pred={pred}, true={test_labels[i]}, confidence_score={avg_conf:.4f}")

    return preds, confidence

def main():
    parser = argparse.ArgumentParser(description="Shadow-Based Membership Inference Attack")
    parser.add_argument("-s", "--synthetic", required=True, help="Path to synthetic data CSV")
    parser.add_argument("-t", "--test", required=True, help="Path to test data CSV (with 'is_member')")
    args = parser.parse_args()

    synthetic_df = pd.read_csv(args.synthetic)
    test_df = pd.read_csv(args.test)

    synthetic_encoded, test_encoded, test_labels = preprocess_data(synthetic_df, test_df)

    preds, confidence = shadow_model_attack_all(synthetic_encoded, test_encoded, test_labels)

    # Add results back to original test_df
    test_df["predicted_is_member"] = preds
    test_df["confidence_score"] = confidence
    test_df["true_label"] = test_labels

    # Print results
    acc = accuracy_score(test_labels, preds)
    auc = roc_auc_score(test_labels, confidence)
    precision = precision_score(test_labels, preds)
    recall = recall_score(test_labels, preds)
    print(f"\n✅ Shadow Model Attack Complete.")
    print(f"🎯 Accuracy: {acc:.4f}")
    print(f"📈 ROC AUC: {auc:.4f}")
    print(f"🔍 Precision:    {precision:.4f}")
    print(f"📢 Recall:       {recall:.4f}")

    # test_df.to_csv("shadow_attack_results.csv", index=False)
    # print("📁 Results saved to shadow_attack_results.csv")

if __name__ == "__main__":
    main()
