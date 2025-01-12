from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.cluster import KMeans
import numpy as np

file_path = "./00_Eingabe/Testtext.txt"

# Schritt 1: Lade die Textdatei
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            yield simple_preprocess(line)  # Tokenisiere die Zeilen

# Schritt 2: Lade und verarbeite den Text
sentences = list(read_text_file(file_path))  # Liste von tokenisierten Sätzen

# Schritt 3: Trainiere das Word2Vec-Modell
model = Word2Vec(
    sentences,  # Tokenisierte Sätze
    vector_size=100,  # Größe der Wortvektoren
    window=5,         # Kontextfenstergröße
    min_count=1,      # Minimale Häufigkeit eines Wortes
    workers=4,        # Anzahl der Threads
    sg=1              # 0 für CBOW, 1 für Skip-Gram
)

# Schritt 4: Speichere das trainierte Modell
model.save("word2vec_model.model")

# Optional: Speichere die Wortvektoren separat
model.wv.save_word2vec_format("word_vectors.txt", binary=False)

######################################################################################
# Schritt 5: Lade Modell und Nutzung des Modells
######################################################################################
loaded_model = Word2Vec.load("word2vec_model.model")

# Ähnliche Wörter finden
print(loaded_model.wv.most_similar("materialien", topn=5))  # Ähnliche Wörter suchen

# Wortvektor abrufen
word_vector = model.wv["materialien"]
print(word_vector)

# Wörter vergleichen (Ähnlichkeit)
similarity = model.wv.similarity("materialien", "den")
print(similarity)

# Wortarithmetik
result = model.wv.most_similar(positive=["den", "materialien"], negative=["bieten"], topn=1)
print(f"'den' - 'bieten' + 'materialien' ≈ {result[0][0]}")

# Clustering
#   Vektoren extrahieren
word_vectors = np.array([model.wv[word] for word in model.wv.index_to_key[:100]])
#   KM-Means++-Clustering (ist der Standard)
kmeans = KMeans(n_clusters=5, random_state=0).fit(word_vectors)
#   Cluster anzeigen
for i, word in enumerate(model.wv.index_to_key[:100]):
    print(f"Wort: {word}, Cluster: {kmeans.labels_[i]}")

# Textklassifikation
def sentence_vector(sentence, model):
    words = sentence.lower().split()
    vectors = [model.wv[word] for word in words if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)
text = "Durch Lösung der Beispiele!"
vector = sentence_vector(text, model)
print(f"Vektor für den Satz '{text}': {vector}")

# Ähnliche Dokumente finden
doc1 = "Durch Lösung der Beispiele!"
doc2 = "Beispiele sind die Lösung."
doc3 = "Ziel benennen."
docs = [doc1, doc2, doc3]
doc_vectors = [sentence_vector(doc, model) for doc in docs]
#  Ähnlichkeit zwischen Dokumenten berechnen
from sklearn.metrics.pairwise import cosine_similarity
similarity_matrix = cosine_similarity(doc_vectors)
print("Ähnlichkeitsmatrix:")
print(similarity_matrix)

# Fehlende Wörter vorhersagen
words = ["Das", "Beispiel", "ist", "___", "."]
candidates = model.wv.most_similar(positive=["beispiel", "beschreiben"], negative=[], topn=5)
print("Wahrscheinlichstes Wort für die Lücke:", candidates)
