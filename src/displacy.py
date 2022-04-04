import spacy
from spacy import displacy
from pathlib import Path
from src.utils import plots_dir

nlp = spacy.load("en_core_web_lg")

doc = nlp("It has that slight sweaty feel but much lighter.")

svg = displacy.render(doc, style="dep", jupyter=False)
output_path = Path(plots_dir() / "dep_example.svg")
output_path.open("w", encoding="utf-8").write(svg)
