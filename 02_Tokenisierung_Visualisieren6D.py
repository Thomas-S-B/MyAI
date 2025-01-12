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
word_frequencies = [model.wv.get_vecattr(word, "count") for word in model.wv.index_to_key]

# Sechste Dimension (z. B. Kategorie)
# Weise zufällig Kategorien zu (ersetze das mit einer realen Metrik)
categories = ['A', 'B', 'C', 'D', 'E']  # Beispielkategorien
category_labels = [categories[i % len(categories)] for i in range(len(word_vectors))]

dimensionen = 10

# Daten für Plotly vorbereiten
data = pd.DataFrame({
    'x': word_vectors_3d[:dimensionen, 0],
    'y': word_vectors_3d[:dimensionen, 1],
    'z': word_vectors_3d[:dimensionen, 2],
    'word': model.wv.index_to_key[:dimensionen],
    'color': vector_lengths[:dimensionen],  # Füge die vierte Dimension (Farbe) hinzu
    'size': word_frequencies[:dimensionen],  # Füge die fünfte Dimension (Größe) hinzu
    'category': category_labels[:dimensionen]  # Füge die sechste Dimension (Form) hinzu
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
    symbol='category',  # Form durch die sechste Dimension
    color_continuous_scale='Viridis',  # Farbskala
)

# Plot-Layout anpassen
fig.update_traces(textposition='top center')
fig.update_layout(
    title="Interaktive 3D-Visualisierung von Word2Vec mit 6 Dimensionen",
    scene=dict(
        xaxis_title='PCA Dimension 1',
        yaxis_title='PCA Dimension 2',
        zaxis_title='PCA Dimension 3'
    )
)

# Zeige den Plot
fig.show()

