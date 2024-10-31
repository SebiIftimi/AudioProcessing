import librosa
import numpy as np
from scipy.spatial.distance import euclidean

# Extract MFCC features from an audio segment
def extract_features(y, sr):
    """
    Extracts the mean MFCC (Mel-frequency cepstral coefficients) from an audio segment.
    
    Args:
        y (np.ndarray): Audio time series data.
        sr (int): Sampling rate of the audio data.
    
    Returns:
        np.ndarray: Mean MFCC values for the segment, used as a feature vector.
    """
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Extract 13 MFCC coefficients
    mfccs_mean = np.mean(mfccs.T, axis=0)  # Calculate mean of each MFCC coefficient across frames
    return mfccs_mean

# Compare two audio segments and determine if they are likely from the same speaker
def compare_voices(segment1, segment2, sr):
    """
    Compares two audio segments based on the Euclidean distance between their feature vectors.
    
    Args:
        segment1 (np.ndarray): First audio segment to compare.
        segment2 (np.ndarray): Second audio segment to compare.
        sr (int): Sampling rate of the audio segments.
    
    Returns:
        str: "Same person" if the distance is below the threshold, otherwise "Different persons".
    """
    threshold = 50  # Distance threshold for determining if segments are from the same speaker
    
    # Extract features (mean MFCCs) for each segment
    features1 = extract_features(segment1, sr)
    features2 = extract_features(segment2, sr)

    # Calculate Euclidean distance between feature vectors of the two segments
    distance = euclidean(features1, features2)
    print(f"Distance: {distance}")
    
    # Check if distance is below threshold and return appropriate label
    if distance < threshold:
        return "Same person"
    else:
        return "Different persons"
