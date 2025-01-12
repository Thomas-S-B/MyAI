import sentencepiece as spm

# Training auf einem Textkorpus (z. B. "train.txt")
spm.SentencePieceTrainer.train(
    input='/home/thomas/myKI/00_Eingabe/Ausführungsplanung.txt',  # Textdatei mit Rohtext
    model_prefix='/home/thomas/myKI/01_Ausgabe_Training/ModelAusführungsplanung',  # Präfix für das gespeicherte Modell
    vocab_size=1900,  # Vokabulargröße
    model_type='bpe'  # oder 'unigram'
)
