# audio_processing.py

import numpy as np
import librosa
import parselmouth
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean

# Function for segmenting audio into fixed-length segments with optional overlap
def audio_segmentation(audio_data, sr, segment_length=2.0, overlap=0.5):
    """
    Segments the audio data into smaller fixed-length chunks.
    
    Args:
        audio_data (np.ndarray): Audio waveform data.
        sr (int): Sampling rate of the audio data.
        segment_length (float): Length of each segment in seconds (default is 2.0 seconds).
        overlap (float): Overlap ratio between consecutive segments (default is 50%).
    
    Returns:
        np.ndarray: 2D array where each row is an audio segment.
    """
    segment_samples = int(segment_length * sr)  # Number of samples per segment
    hop_length = int(segment_samples * (1 - overlap))  # Step size based on overlap
    segments = librosa.util.frame(audio_data, frame_length=segment_samples, hop_length=hop_length).T
    return segments

# Function for extracting audio features from each segment
def audio_features_extraction(segments, sr):
    """
    Extracts a variety of audio features from each audio segment.
    
    Args:
        segments (np.ndarray): Array of audio segments.
        sr (int): Sampling rate of the audio data.
    
    Returns:
        np.ndarray: Feature matrix where each row represents the feature vector of a segment.
    """
    features = []
    for segment in segments:
        # Extract audio features for the segment
        mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
        mel = librosa.feature.melspectrogram(y=segment, sr=sr)
        chroma = librosa.feature.chroma_stft(y=segment, sr=sr)
        rms = librosa.feature.rms(y=segment)
        zcr = librosa.feature.zero_crossing_rate(y=segment)
        
        # Combine statistical properties (mean and std) of each feature into a feature vector
        feature_vector = np.hstack([
            np.mean(mfccs, axis=1), np.std(mfccs, axis=1),
            np.mean(mel, axis=1), np.std(mel, axis=1),
            np.mean(chroma, axis=1), np.std(chroma, axis=1),
            np.mean(rms), np.std(rms),
            np.mean(zcr), np.std(zcr)
        ])
        features.append(feature_vector)
    return np.array(features)

# Function to calculate the mean and standard deviation for all features in a matrix
def aggregate_features(features):
    """
    Aggregates features across segments by computing their mean and standard deviation.
    
    Args:
        features (np.ndarray): Feature matrix with features for each segment.
    
    Returns:
        np.ndarray: Concatenated vector of mean and std for all features.
    """
    mean_features = np.mean(features, axis=0)
    std_features = np.std(features, axis=0)
    return np.hstack((mean_features, std_features))

# Function to calculate Pearson correlation between two feature vectors
def calculate_pearson_corr(feature1, feature2):
    """
    Computes the Pearson correlation coefficient between two feature vectors.
    
    Args:
        feature1 (np.ndarray): First feature vector.
        feature2 (np.ndarray): Second feature vector.
    
    Returns:
        float: Pearson correlation coefficient.
    """
    pearson_corr, _ = pearsonr(feature1, feature2)
    return pearson_corr

# High-level function to perform segmentation, feature extraction, and correlation analysis
def all_functions(audio_data, sr):
    """
    Processes the audio data by segmenting, extracting features, and calculating 
    Pearson correlation between the first segment and each subsequent segment.
    
    Args:
        audio_data (np.ndarray): Audio waveform data.
        sr (int): Sampling rate of the audio data.
    
    Returns:
        list: Pearson correlation values between the first segment and each other segment.
    """
    seg = audio_segmentation(audio_data, sr, segment_length=2.0, overlap=0.5)
    feature = audio_features_extraction(segments=seg, sr=sr)
    correlations = []
    
    # Aggregate features of the first segment
    aggregated_features_1 = aggregate_features([feature[0]])
    for i in range(1, len(seg)):
        aggregated_features_current = aggregate_features([feature[i]])
        pearson_corr = calculate_pearson_corr(aggregated_features_1, aggregated_features_current)
        correlations.append(pearson_corr)
    return correlations

# Function to extract MFCC features specifically from an audio clip
def extract_features(y, sr):
    """
    Extracts MFCC features and computes the mean for each coefficient.
    
    Args:
        y (np.ndarray): Audio waveform data.
        sr (int): Sampling rate of the audio data.
    
    Returns:
        np.ndarray: Mean MFCC values for the audio clip.
    """
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    print(f"Mfcc mean = {mfccs_mean}")
    return mfccs_mean

# Function to compare audio segments and determine if they are from the same speaker
def compare_voices(segment1, segment2, sr, threshold=45):
    """
    Compares two audio segments based on their Euclidean distance in feature space.
    
    Args:
        segment1 (np.ndarray): First audio segment.
        segment2 (np.ndarray): Second audio segment.
        sr (int): Sampling rate of the audio segments.
        threshold (float): Distance threshold for determining speaker similarity (default is 45).
    
    Returns:
        str: "Same person" if distance is below the threshold, otherwise "Different persons".
    """
    # Extract features for each segment
    features1 = extract_features(segment1, sr)
    features2 = extract_features(segment2, sr)

    # Compute Euclidean distance between the feature vectors
    distance = euclidean(features1, features2)
    print(f"Distance: {distance}")
    
    # Determine if segments are from the same person based on threshold
    if distance < threshold:
        return "Same person"
    else:
        return "Different persons"
