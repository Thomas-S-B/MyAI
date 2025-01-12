from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

file_path = "./02_Tokenisierung/Testtext.txt"

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
    sg=0              # 0 für CBOW, 1 für Skip-Gram
)

# Schritt 4: Speichere das trainierte Modell
model.save("word2vec_model.model")

# Optional: Speichere die Wortvektoren separat
model.wv.save_word2vec_format("word_vectors.txt", binary=False)

# Schritt 5: Lade und nutze das Modell
# Ähnliche Wörter finden
loaded_model = Word2Vec.load("word2vec_model.model")
print(loaded_model.wv.most_similar("materialien", topn=5))  # Ähnliche Wörter suchen

# Wortvektor abrufen
word_vector = model.wv["materialien"]
print(word_vector)

# Wörter vergleichen (Ähnlichkeit)
similarity = model.wv.similarity("materialien", "diese")
print(similarity)


