import os
import numpy as np
import spacy
import pandas as pd
from numpy import ndarray
from src.utils import patterns_dir, transcripts_dir, participants_dir

nlp = spacy.load("en_core_web_lg")

# ===== Language =====

groups = ["controlled", "free", "combined"]
default_group = "combined"


def get_number_of_participants(group=default_group) -> int:
    assert group in groups
    sequences = _read_sequences(group)  # n participants X 16 patterns
    return len(sequences)


def get_full_text(group=default_group) -> str:
    assert group in groups
    transcripts = _read_transcripts(group)
    transcripts_merged = [" ".join(p) for p in transcripts]  # merge trials per participant
    return " ".join(transcripts_merged)  # merge participants


def get_transcripts(group=default_group) -> list:
    assert group in groups
    return [" ".join(p) for p in _read_transcripts(group)]  # join trials per participant


def get_transcript(participant: int) -> list:
    transcripts = _read_transcripts()
    index = participant - 1  # convert to array index
    return transcripts[index]


def get_descriptions_as_dict(group=default_group) -> dict:
    assert group in groups
    transcripts = _read_transcripts(group)  # n participants X 16 patterns
    sequences = _read_sequences(group)  # n participants X 16 patterns
    assert sequences.shape == transcripts.shape

    patterns = {i + 1: [] for i in range(32)}

    for i, sequence in enumerate(sequences):
        for j, pattern_number in enumerate(sequence):
            trial = transcripts[i, j]
            patterns[pattern_number].append(trial)

    assert np.array(list(patterns.values())).shape == (
        32, len(sequences) / 2)  # 32 patterns X 12 occurrences (= n/2 participants)
    return patterns


def get_descriptions(group=default_group) -> ndarray:
    assert group in groups
    descriptions_as_dict = get_descriptions_as_dict(group)
    return np.array(list(descriptions_as_dict.values()))  # 32 patterns X 12 occurrences (= n/2 participants)


def get_merged_descriptions(group=default_group) -> list:
    assert group in groups
    descriptions = get_descriptions(group)  # 32 patterns X 12 occurrences (= n/2 participants)
    merged_descriptions = [" ".join(trials_per_pattern) for trials_per_pattern in descriptions]
    assert np.array(merged_descriptions).shape == (32,)
    return merged_descriptions


def get_norm_descriptions(group=default_group) -> ndarray:
    assert group in groups
    descriptions = get_descriptions(group)

    def normalize(text: str) -> str:
        doc = nlp(str(text))
        lemmas = set([token.lemma_.lower() for token in doc if token.pos_ in ["NOUN", "ADJ"]])
        return " ".join(lemmas)

    norm_vec = np.vectorize(normalize)
    norm_descriptions = norm_vec(descriptions)

    assert norm_descriptions.shape == descriptions.shape  # 32 patterns X 12 occurrences (= n/2 participants)
    return norm_descriptions


def get_merged_norm_descriptions(group=default_group) -> list:
    assert group in groups
    norm_descriptions = get_norm_descriptions(group)  # 32 patterns X n/2 participants
    merged_norm_descriptions = [" ".join(trials_per_pattern) for trials_per_pattern in norm_descriptions]
    assert np.array(merged_norm_descriptions).shape == (32,)
    return merged_norm_descriptions


def _read_transcripts(group=default_group) -> ndarray:
    assert group in groups
    transcripts = np.array([]).reshape(0, 16)  # number of patterns per participant

    for filename in sorted(os.listdir(transcripts_dir())):
        if not filename.endswith(".txt"):
            continue

        with open(transcripts_dir() / filename) as f:
            pattern = f.read().splitlines()
            transcripts = np.vstack((transcripts, pattern))

    if group == "controlled":
        transcripts = transcripts[:12]
        return np.array(transcripts)

    if group == "free":
        transcripts = transcripts[12:]
        return np.array(transcripts)

    # "combined"
    return np.array(transcripts)


def _read_sequences(group=default_group) -> ndarray:
    assert group in groups

    sequences = np.genfromtxt(participants_dir() / "sequences.txt", delimiter=",", dtype=int)

    if group == "controlled":
        sequences = sequences[:12]
        return np.array(sequences)

    if group == "free":
        sequences = sequences[12:]
        return np.array(sequences)

    # "combined"
    return np.array(sequences)


# ===== Signal =====


def load_original_patterns() -> pd.DataFrame:
    return pd.read_csv(patterns_dir() / "original.csv")


def load_original_pattern(id: int) -> np.ndarray:
    assert id in range(1, 33)
    df = pd.read_csv(patterns_dir() / "original.csv")
    signal = df.loc[df["id"] == id]
    y = signal["y"].to_numpy()
    return y


def load_normalized_patterns() -> pd.DataFrame:
    return pd.read_csv(patterns_dir() / "normalized.csv")
