import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def load_data(path):
    df = pd.read_csv(path, encoding='ISO-8859-1')
    df = df[df['InvoiceNo'].str.startswith('C') == False]
    df.dropna(subset=['CustomerID'], inplace=True)
    return df

def compute_rfm(df):
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalSum': 'sum'
    }).reset_index()
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    return rfm

def preprocess_rfm(rfm):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
    return scaled

def apply_kmeans(data, n_clusters=4):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data)
    return clusters

def reduce_pca(data):
    pca = PCA(n_components=2)
    components = pca.fit_transform(data)
    return components
