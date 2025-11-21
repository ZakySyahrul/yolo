import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def cluster_traffic(df):
    """
    Menambahkan kolom 'density_cluster' ke df
    tanpa menghapus kolom fitur lama.
    """

    data = df.copy()

    features = ["total", "bbox_ratio", "spread_x", "spread_y"]

    for f in features:
        data[f] = data[f].fillna(0)

    X = data[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    data["cluster"] = labels

    # urutkan cluster berdasarkan rata-rata total kendaraan
    order = data.groupby("cluster")["total"].mean().sort_values().index.tolist()

    mapping = {
        order[0]: "Lancar",
        order[1]: "Ramai",
        order[2]: "Padat"
    }

    data["density_cluster"] = data["cluster"].map(mapping)

    return data
