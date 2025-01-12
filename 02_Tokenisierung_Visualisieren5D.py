from sklearn.decomposition import PCA
import plotly.express as px
from gensim.models import Word2Vec
import pandas as pd
import numpy as np

# Lade das Word2Vec-Modell
model = Word2Vec.load("word2vec_model.model")

# Reduziere die Dimension auf 3
pca = PCA(n_components=3)
word_vectors = model.wv[model.wv.index_to_key]
word_vectors_3d = pca.fit_transform(word_vectors)

# Berechne die vierte Dimension (z. B. Vektorlänge)
vector_lengths = np.linalg.norm(word_vectors, axis=1)

# Berechne die fünfte Dimension (z. B. Häufigkeit des Wortes)
# Hinweis: Hier wird ein zufälliger Wert als Platzhalter genutzt, ersetze ihn mit einer realen Metrik.
word_frequencies = [model.wv.get_vecattr(word, "count") for word in model.wv.index_to_key]

dimensionen = 25

# Daten für Plotly vorbereiten
data = pd.DataFrame({
    'x': word_vectors_3d[:dimensionen, 0],
    'y': word_vectors_3d[:dimensionen, 1],
    'z': word_vectors_3d[:dimensionen, 2],
    'word': model.wv.index_to_key[:dimensionen],
    'color': vector_lengths[:dimensionen],  # Füge die vierte Dimension (Farbe) hinzu
    'size': word_frequencies[:dimensionen]  # Füge die fünfte Dimension (Größe) hinzu
})

# Interaktiven 3D-Plot erstellen
fig = px.scatter_3d(
    data,
    x='x',
    y='y',
    z='z',
    text='word',
    color='color',  # Farbe durch die vierte Dimension
    size='size',    # Größe durch die fünfte Dimension
    color_continuous_scale='Viridis',  # Farbskala
)

# Plot-Layout anpassen
fig.update_traces(textposition='top center')
fig.update_layout(
    title="Interaktive 3D-Visualisierung von Word2Vec mit Farbe und Größe",
    scene=dict(
        xaxis_title='PCA Dimension 1',
        yaxis_title='PCA Dimension 2',
        zaxis_title='PCA Dimension 3'
    )
)

# Zeige den Plot
fig.show()
