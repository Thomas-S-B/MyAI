from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px
from gensim.models import Word2Vec
import pandas as pd

# Lade das Word2Vec-Modell
model = Word2Vec.load("word2vec_model.model")

# Reduziere die Dimension auf 3
pca = PCA(n_components=3)
word_vectors = model.wv[model.wv.index_to_key]
word_vectors_3d = pca.fit_transform(word_vectors)

# Wende K-Means-Clustering an
num_clusters = 2  # Anzahl der Cluster
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(word_vectors_3d)

dimensionen = 10

# Daten für Plotly vorbereiten
data = pd.DataFrame({
    'x': word_vectors_3d[:dimensionen, 0],
    'y': word_vectors_3d[:dimensionen, 1],
    'z': word_vectors_3d[:dimensionen, 2],
    'word': model.wv.index_to_key[:dimensionen],
    'cluster': cluster_labels[:dimensionen]  # Clusterzugehörigkeit
})

# Interaktiven 3D-Plot erstellen
fig = px.scatter_3d(
    data,
    x='x',
    y='y',
    z='z',
    text='word',
    color='cluster',  # Clusterzugehörigkeit als Farbe
    symbol='cluster',  # Unterschiedliche Symbole für Cluster
    color_discrete_sequence=px.colors.qualitative.Set1  # Diskrete Farben für Cluster
)

# Plot-Layout und Legende anpassen
fig.update_traces(marker=dict(size=8), textposition='top center')
fig.update_layout(
    title="3D-Cluster-Visualisierung von Word2Vec",
    legend=dict(
        title="Cluster",
        orientation="v",  # Vertikale Ausrichtung der Legende
        x=1.10,  # Position rechts außerhalb des Plots
        y=1,    # Obere Position
        bgcolor="rgba(255,255,255,0.8)",  # Hintergrund der Legende
        bordercolor="black",  # Randfarbe
        borderwidth=1         # Randbreite
    ),
    scene=dict(
        xaxis_title='PCA Dimension 1',
        yaxis_title='PCA Dimension 2',
        zaxis_title='PCA Dimension 3'
    )
)

# Zeige den Plot
fig.show()
