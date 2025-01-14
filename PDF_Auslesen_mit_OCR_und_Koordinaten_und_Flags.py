import pdfplumber
import os
import re

# PDF-Datei öffnen
pdf_path = "./abc.pdf"

# Name der Ausgabedatei erstellen
output_path = os.path.splitext(pdf_path)[0] + "_OCR_mit_Koordinaten_und_Flags.txt"

# Funktion zur Prüfung, ob ein Wert entlang einer Linie liegt
def is_on_line(ref_item, other_items, threshold=5):
    """Prüft, ob ein Text auf einer horizontalen Linie mit anderen Werten liegt."""
    ref_y = ref_item["top"]  # Y-Koordinate der oberen Linie des Referenzwertes
    for item in other_items:
        if item == ref_item:
            continue
        if abs(ref_y - item["top"]) <= threshold:
            return True
    return False

# Funktion zur Prüfung, ob ein Text eine Gewichtsangabe ist
def is_weight(text):
    weight_pattern = re.compile(r"\b\d+(\.\d+)?\s?(kg|g|t|tonnen|kilogramm|gramm)\b", re.IGNORECASE)
    return bool(weight_pattern.search(text))

# Funktion zur Ermittlung des Labels, das links von einem numerischen Wert steht
def find_left_label(ref_item, other_items, threshold=5, is_vertical_flag=False):
    """Findet das Textlabel links eines numerischen Wertes, basierend auf der Ausrichtung."""
    ref_x = ref_item["x0"]  # X-Koordinate des Referenzwertes
    ref_y = ref_item["top"]  # Y-Koordinate des Referenzwertes

    left_label = ""
    for item in other_items:
        if item == ref_item:
            continue
        # Nur Objekte auf derselben Linie und links vom Referenzwert betrachten
        if abs(ref_y - item["top"]) <= threshold:
            # Falls der Text vertikal ist, prüfen wir nur vertikale Labels
            if is_vertical_flag:
                if abs(item["x1"] - item["x0"]) <= 10 and item["x1"] < ref_x:
                    left_label = item.get("text", "")
            else:
                # Falls der Text horizontal ist, prüfen wir nur horizontale Labels
                if item["x1"] < ref_x:
                    left_label = item.get("text", "")
    return left_label

# Funktion zur Bestimmung der Ausrichtung (horizontal oder vertikal)
def is_vertical(ref_item, threshold=10):
    """Prüft, ob der Text vertikal ausgerichtet ist."""
    return abs(ref_item["x1"] - ref_item["x0"]) <= threshold  # Kleine X-Differenz deutet auf vertikal hin

# Text und Koordinaten aus PDF extrahieren
with pdfplumber.open(pdf_path) as pdf, open(output_path, "w", encoding="utf-8") as output_file:
    for page_number, page in enumerate(pdf.pages, start=1):
        # Verwende extract_text() für eine breitere Textextraktion, auch für kleine Texte
        page_text = page.extract_text()  # Extrahiert den gesamten Text
        text_objects = page.extract_words()  # Extrahiert Text mit Koordinaten

        # Output für die Seite
        output_file.write(f"Seite {page_number}:\n")
        
        # Durchlaufen aller Textobjekte
        for obj in text_objects:
            text = obj.get("text", "")
            x0 = obj.get("x0", 0)
            y0 = obj.get("top", 0)
            x1 = obj.get("x1", 0)
            y1 = obj.get("bottom", 0)

            # Prüfen, ob der Text numerisch ist oder eine Zahl mit einer Einheit enthält
            is_numeric = re.match(r"^(\d+(\.\d+)?)(\s*(kg|g|t|tonnen|kilogramm|gramm))?$", text)

            # Prüfen, ob der Text auf einer horizontalen Linie liegt
            on_line = is_on_line(obj, text_objects)

            # Prüfen, ob der Text eine Gewichtsangabe ist
            is_weight_flag = is_weight(text)

            # Bestimmen, ob der Text horizontal oder vertikal ausgerichtet ist
            vertical_flag = is_vertical(obj)

            # Finden des Labels links vom numerischen Wert, basierend auf der Ausrichtung
            left_label = find_left_label(obj, text_objects, is_vertical_flag=vertical_flag) if is_numeric else ""

            # Schreiben der Informationen in die Ausgabedatei
            output_file.write(f"Text: '{text}'\n")
            output_file.write(f"  Position: ({x0:.2f}, {y0:.2f}) bis ({x1:.2f}, {y1:.2f})\n")
            output_file.write(f"  Numerisch: {bool(is_numeric)}\n")
            output_file.write(f"  Auf Linie: {on_line}\n")
            output_file.write(f"  Gewichtsangabe: {is_weight_flag}\n")
            output_file.write(f"  Linkes Label: '{left_label}'\n")
            output_file.write(f"  Vertikal: {vertical_flag}\n")
        output_file.write("\n")

print(f"Die Daten mit Text, Koordinaten und Flags wurden erfolgreich in {output_path} gespeichert.")

