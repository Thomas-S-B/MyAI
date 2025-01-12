import sentencepiece as spm

# Training auf einem Textkorpus
spm.SentencePieceTrainer.train(
    input='./00_Eingabe/Testtext.txt',  # Textdatei mit Rohtext
    model_prefix='./01_Ausgabe_Training/ModelAusführungsplanung',  # Präfix für das gespeicherte Modell
    vocab_size=1900,  # Vokabulargröße
    model_type='bpe'  # oder 'unigram'
)
