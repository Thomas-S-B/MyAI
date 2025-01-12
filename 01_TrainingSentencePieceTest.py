import sentencepiece as spm

# Training auf einem Textkorpus 
spm.SentencePieceTrainer.train(
    input='/home/thomas/myKI/00_Eingabe/Testtext.txt',  # Textdatei mit Rohtext
    model_prefix='/home/thomas/myKI/01_Ausgabe_Training/MeinModel',  # Präfix für das gespeicherte Modell
    vocab_size=1900,  # Vokabulargröße
    model_type='bpe'  # oder 'unigram'
)
