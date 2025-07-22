import argparse
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist

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
    categorical_cols = combined.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = combined.select_dtypes(exclude=["object"]).columns.tolist()

    for col in categorical_cols:
        combined[col] = combined[col].fillna("missing")
    for col in numeric_cols:
        combined[col] = combined[col].fillna(0)

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ],
        remainder="passthrough"
    )
    pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
    encoded_data = pipeline.fit_transform(combined)

    if hasattr(encoded_data, "toarray"):
        encoded_data = encoded_data.toarray()

    synthetic_encoded = encoded_data[:len(synthetic_df)]
    test_encoded = encoded_data[len(synthetic_df):]

    return synthetic_encoded, test_encoded, test_labels

def harmonic_mean(distances):
    return len(distances) / np.sum(1.0 / (distances + 1e-10))  # Avoid div-by-zero

def reconstruction_attack(synthetic_data, tau=0.05, k=5):
    N = len(synthetic_data)
    N_recon = int(N * tau)

    knn = NearestNeighbors(n_neighbors=k+1, metric='euclidean')
    knn.fit(synthetic_data)

    distances, indices = knn.kneighbors(synthetic_data)
    distances = distances[:, 1:]  # Remove self
    indices = indices[:, 1:]

    harmonic_scores = np.array([harmonic_mean(d) for d in distances])
    neighbors_dict = {i: set(indices[i]) for i in range(N)}

    ranked = np.argsort(harmonic_scores)
    reconstructed = set()
    discarded = set()

    for idx in ranked:
        if idx in discarded:
            continue
        reconstructed.add(idx)
        discarded.update(neighbors_dict[idx])
        if len(reconstructed) >= N_recon:
            break

    return list(reconstructed), harmonic_scores

def match_to_test(reconstructed_indices, synthetic_encoded, test_encoded, test_labels):
    matched_labels = []
    matched_distances = []

    for idx in reconstructed_indices:
        syn_sample = synthetic_encoded[idx].reshape(1, -1)
        dists = cdist(syn_sample, test_encoded, metric="euclidean")
        nearest_idx = dists.argmin()
        matched_labels.append(test_labels[nearest_idx])
        matched_distances.append(dists.min())

    return matched_labels, matched_distances

def main():
    parser = argparse.ArgumentParser(description="Reconstruction Attack via Synthetic Clustering")
    parser.add_argument("-s", "--synthetic", required=True, help="Path to synthetic data CSV")
    parser.add_argument("-t", "--test", required=True, help="Path to test data CSV (with 'is_member')")
    parser.add_argument("--tau", type=float, default=0.05, help="Proportion of reconstructed records (default: 0.05)")
    parser.add_argument("--k", type=int, default=5, help="Number of nearest neighbors (default: 5)")
    args = parser.parse_args()

    # Load data
    synthetic_df = pd.read_csv(args.synthetic)
    test_df = pd.read_csv(args.test)

    synthetic_encoded, test_encoded, test_labels = preprocess_data(synthetic_df, test_df)

    recon_indices, harmonic_scores = reconstruction_attack(synthetic_encoded, tau=args.tau, k=args.k)

    print(f"Selected {len(recon_indices)} reconstructed synthetic samples.")

    matched_labels, matched_distances = match_to_test(recon_indices, synthetic_encoded, test_encoded, test_labels)

    num_matches = len(matched_labels)
    num_members = sum(matched_labels)
    percent_members = 100 * num_members / num_matches

    print("Results:")
    print(f" - Reconstructed samples matched: {num_matches}")
    print(f" - True training members among them: {num_members} ({percent_members:.2f}%)")
    # print(matched_labels)

if __name__ == "__main__":
    main()
