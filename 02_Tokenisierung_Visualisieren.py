from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from gensim.models import Word2Vec

model = Word2Vec.load("word2vec_model.model")

# Reduziere die Dimension auf 2
pca = PCA(n_components=2)
word_vectors = model.wv[model.wv.index_to_key]
word_vectors_2d = pca.fit_transform(word_vectors)

# Visualisiere die Vektoren
plt.figure(figsize=(20, 10))
for i, word in enumerate(model.wv.index_to_key[:250]):  # Zeige die ersten 250 Wörter
    plt.scatter(word_vectors_2d[i, 0], word_vectors_2d[i, 1])
    plt.text(word_vectors_2d[i, 0], word_vectors_2d[i, 1], word, fontsize=10)

plt.show()

