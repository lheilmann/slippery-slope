import numpy as np
import librosa
import librosa.feature
import pandas as pd
import matplotlib.pyplot as plt
from src.data import load_normalized_patterns
from src.utils import features_dir, plots_dir
from tqdm import tqdm

# Plots
font = {
    'size': 16
}
plt.rc('font', **font)

SAMPLING_RATE = 44100
FRAME_LENGTH = 2048
HOP_LENGTH = 2048


# ----- Time-domain features -----

def get_mean_amplitude(y: np.ndarray) -> float:
    return np.mean(y)


def get_zero_count(y: np.ndarray) -> int:
    return len([sample for sample in y if -2 <= sample <= 2])


def get_mean_onset_strength(y: np.ndarray) -> float:
    onset_env = librosa.onset.onset_strength(y=y, sr=SAMPLING_RATE)
    return np.mean(onset_env)


def get_beat_count(y: np.ndarray) -> int:
    rms = librosa.feature.rms(y=y)
    amplitude_env = rms[0]
    peaks = librosa.util.peak_pick(amplitude_env, 3, 3, 3, 5, 0.5, 10)
    return len(peaks)


def get_mean_distance_between_beats(y: np.ndarray):
    rms = librosa.feature.rms(y=y)
    amplitude_env = rms[0]
    peaks = librosa.util.peak_pick(amplitude_env, 3, 3, 3, 5, 0.5, 10)
    temp = np.insert(peaks, 0, 0, axis=0)
    distances = [peaks[i] - temp[i] for i in range(len(peaks))]
    return np.mean(distances)


def get_std_distance_between_beats(y: np.ndarray):
    rms = librosa.feature.rms(y=y)
    amplitude_env = rms[0]
    peaks = librosa.util.peak_pick(amplitude_env, 3, 3, 3, 5, 0.5, 10)
    temp = np.insert(peaks, 0, 0, axis=0)
    distances = [peaks[i] - temp[i] for i in range(len(peaks))]
    return np.std(distances)


# ----- Frequency-domain features -----


def get_mean_rms(y: np.ndarray) -> float:
    rms = librosa.feature.rms(y=y,
                              frame_length=FRAME_LENGTH,
                              hop_length=HOP_LENGTH)
    return np.mean(rms)


def get_mean_spectral_centroid(y: np.ndarray):
    per_frame = librosa.feature.spectral_centroid(y=y,
                                                  sr=SAMPLING_RATE,
                                                  hop_length=HOP_LENGTH)[0]
    return np.mean(per_frame)


# Load combined signal data
df = load_normalized_patterns()

# Initialize feature matrix
X = pd.DataFrame(columns=[])

# Calculate signal features for each pattern
for pattern_id in tqdm(range(1, 33)):
    signal = df.loc[df["id"] == pattern_id]
    y = signal["y"].to_numpy()
    D = librosa.stft(y)
    times = librosa.times_like(D,
                               sr=SAMPLING_RATE,
                               hop_length=HOP_LENGTH)


    # print("---- Pattern {} -----".format(pattern_id))
    # rms = librosa.feature.rms(y=y)
    # plt.figure(figsize=(10, 10))
    # plt.plot(range(len(rms[0])), rms[0])
    # plt.show()
    # break

    # rms = librosa.feature.rms(y=y,
    #                           frame_length=FRAME_LENGTH,
    #                           hop_length=HOP_LENGTH)
    # path = plots_dir() / "rms_3.png"
    # plt.figure(figsize=(35, 5))
    # plt.plot(range(len(rms[0])), rms[0], linewidth=0.5)
    # plt.xlabel('Frame')
    # plt.ylabel('RMS')
    # plt.savefig(path,
    #             bbox_inches='tight',
    #             transparent=False,
    #             pad_inches=0,
    #             dpi=300
    #             )
    # break

    feature_vec = pd.DataFrame.from_dict({
        "id": [pattern_id],
        # -- Temporal features
        "mean_amplitude": ["%.2f" % round(get_mean_amplitude(y), 2)],
        "rms": ["%.2f" % round(get_mean_rms(y), 4)],
        "pulse_count": [get_beat_count(y)],
        "std_pulse_dist": ["%.2f" % round(get_std_distance_between_beats(y), 2)],
        "zero_count": [get_zero_count(y)],
        "mean_onset_strength": ["%.2f" % round(get_mean_onset_strength(y), 2)],
        # -- Spectral features
        "spectral_centroid": ["%.2f" % round(get_mean_spectral_centroid(y), 2)],
    })
    X = pd.concat([X, feature_vec], ignore_index=True)

print(X.head())

# Save feature matrix to disk
path = features_dir() / "signal" / "handmade" / "features.csv"
X.to_csv(path, index=False)
