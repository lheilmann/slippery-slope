from pathlib import Path


def root_dir() -> Path:
    return Path(__file__).parent.parent


def data_dir() -> Path:
    return root_dir() / "data"


def transcripts_dir() -> Path:
    return data_dir() / "transcripts"


def participants_dir() -> Path:
    return data_dir() / "participants"


def patterns_dir() -> Path:
    return data_dir() / "patterns" / "combined"


def audio_dir() -> Path:
    return data_dir() / "patterns" / "audio"


def features_dir() -> Path:
    return root_dir() / "features"


def clusters_dir() -> Path:
    return results_dir() / "clusters"


def descriptors_dir() -> Path:
    return results_dir() / "descriptors"


def plots_dir() -> Path:
    return root_dir() / "plots"


def references_dir() -> Path:
    return root_dir() / "references"


def glove_dir() -> Path:
    return references_dir() / "glove"


def liwc_dir() -> Path:
    return references_dir() / "liwc"


def nrc_dir() -> Path:
    return references_dir() / "nrc"


def model_dir() -> Path:
    return root_dir() / "model"


def results_dir() -> Path:
    return root_dir() / "results"
