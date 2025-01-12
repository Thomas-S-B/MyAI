from sklearn.decomposition import PCA
import plotly.express as px
from gensim.models import Word2Vec
import pandas as pd

# Lade das Word2Vec-Modell
model = Word2Vec.load("word2vec_model.model")

# Reduziere die Dimension auf 3
pca = PCA(n_components=3)
word_vectors = model.wv[model.wv.index_to_key]
word_vectors_3d = pca.fit_transform(word_vectors)

# Daten für Plotly vorbereiten
data = pd.DataFrame({
    'x': word_vectors_3d[:250, 0],
    'y': word_vectors_3d[:250, 1],
    'z': word_vectors_3d[:250, 2],
    'word': model.wv.index_to_key[:250]
})

# Interaktiven 3D-Plot erstellen
fig = px.scatter_3d(data, x='x', y='y', z='z', text='word')

# Plot-Layout anpassen
fig.update_traces(marker=dict(size=5), textposition='top center')
fig.update_layout(
    title="Interaktive 3D-Visualisierung von Word2Vec",
    scene=dict(
        xaxis_title='PCA Dimension 1',
        yaxis_title='PCA Dimension 2',
        zaxis_title='PCA Dimension 3'
    )
)

# Zeige den Plot
fig.show()

