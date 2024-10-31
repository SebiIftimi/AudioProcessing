import numpy as np
import librosa
import os
import pandas as pd
from scipy.spatial.distance import euclidean

# Extract MFCC features from an audio segment
def extract_features(y, sr):
    """
    Extracts the mean of 13 MFCC (Mel-frequency cepstral coefficients) from an audio segment.
    
    Args:
        y (np.ndarray): Audio time series data.
        sr (int): Sampling rate of the audio data.
    
    Returns:
        np.ndarray: Mean MFCC values as a feature vector for the segment.
    """
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

# Compare two feature vectors and determine if they are similar based on a threshold
def compare_voices(features1, features2, threshold=45):
    """
    Compares two audio feature vectors using Euclidean distance and checks if they are similar.
    
    Args:
        features1 (np.ndarray): Feature vector of the first segment.
        features2 (np.ndarray): Feature vector of the second segment.
        threshold (float): Distance threshold for determining similarity (default is 45).
    
    Returns:
        bool: True if distance is below the threshold (similar), False otherwise.
    """
    if len(features1) != len(features2):
        print(f"Feature length mismatch: {len(features1)} vs {len(features2)}")
        return False
    distance = euclidean(features1, features2)
    return distance < threshold

# Read all CSV files in a directory and extract feature data
def read_all_csv(directory):
    """
    Reads all CSV files in a specified directory, extracting stored feature vectors.
    
    Args:
        directory (str): Path to the directory containing CSV files.
    
    Returns:
        list of tuples: List containing tuples of person_id, feature vector, and file path.
    """
    data = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)
            for _, row in df.iterrows():
                person_id = os.path.splitext(filename)[0]  # Use filename as person_id
                features = row.iloc[:].values.astype(float)
                if len(features) == 13:  # Ensure correct feature length
                    data.append((person_id, features, file_path))
    return data

# Append a feature vector to a CSV file
def write_csv(features, filename):
    """
    Appends a feature vector to a specified CSV file.
    
    Args:
        features (np.ndarray): Feature vector to write.
        filename (str): Path to the CSV file.
    """
    df = pd.DataFrame([features], columns=[f'MFCC{i}' for i in range(1, 14)])
    df.to_csv(filename, mode='a', header=not os.path.exists(filename), index=False)

# Process an audio segment, identify or register a person, and save features
def process_segment(segment, sr, directory):
    """
    Processes an audio segment, compares it to existing data, and either identifies
    the speaker or registers them as a new person if they do not match existing data.
    
    Args:
        segment (np.ndarray): Audio segment data.
        sr (int): Sampling rate of the audio segment.
        directory (str): Directory containing CSV files for each known person.
    
    Returns:
        str: Message indicating if the segment was identified or registered as a new person.
    """
    features = extract_features(segment, sr)  # Extract features from the segment
    all_data = read_all_csv(directory)  # Read all stored feature data
    
    person_found = False  # Flag to indicate if the person was found
    person_id = None  # Variable to hold identified person_id

    # Compare extracted features to stored features for each known person
    for pid, stored_features, file_path in all_data:
        if compare_voices(features, stored_features):
            write_csv(features, file_path)  # Append features if person is identified
            person_id = pid
            person_found = True
            break

    # If no match was found, register a new person and create a new CSV file
    if not person_found:
        new_person_id = f"Person_{len(set([x[0] for x in all_data])) + 1}"  # Generate a new ID
        new_file_path = os.path.join(directory, f"{new_person_id}.csv")
        write_csv(features, new_file_path)  # Save features in new file
        person_id = new_person_id
    
    # Return a message indicating whether the segment was identified or registered
    if person_found:
        message = f"Segment was identified as belonging to person {person_id}"
    else:
        message = f"Segment was identified as a new person: {person_id}"
    
    return message
