% Set up Python environment to run audio processing functions
pyenv('Version', '/Users/sebiiftimi/myenvpractica1/bin/python3');

% Import custom Python module for audio processing
audio = py.importlib.import_module('audio_processing');

% Read audio file and get audio data and sample rate
[audio_data, fs] = audioread('1.wav');

% Detect speech segments within the audio file
% The function detectSpeech(audio_data, fs) returns an Nx2 matrix where:
% - N is the number of detected speech segments
% - The first column contains the start sample of each segment
% - The second column contains the end sample of each segment
speech_segments = detectSpeech(audio_data, fs);

% Set threshold for merging speech segments
% If consecutive segments have a time gap of 0.6 seconds or less, 
% they will be concatenated to account for short pauses, like breaths.
gap_between_segments = 0.6;

% Initialize 'new_segments' to store concatenated speech segments
new_segments = [];
new_segments = speech_segments(1,:);  % Start with the first detected segment

% Loop through each detected speech segment to merge close segments
% For each segment, if the gap to the previous segment is within the threshold, 
% concatenate them; otherwise, add it as a separate segment.
for i = 2:size(speech_segments, 1)
    prev_segment = speech_segments(i-1, :);  % Previous segment
    curr_segment = speech_segments(i, :);    % Current segment
    if (speech_segments(i) - prev_segment(2)) / fs <= gap_between_segments
        % Merge with previous segment if gap is below threshold
        new_segments(end,2) = speech_segments(i,2);
    else
        % Add current segment as a new entry in 'new_segments'
        new_segments = [new_segments; curr_segment];
    end
end

% Initialize a cell array to store features for each merged audio segment
features = {}; 
sr = fs;  % Define sample rate for feature extraction

% Extract vocal features for each segment in 'new_segments'
% Loop through each segment to extract and aggregate features
for i = 1:size(new_segments, 1)
    % Extract audio samples for the current segment
    segment = audio_data(new_segments(i,1):new_segments(i,2));

    % Convert MATLAB array to Python-compatible numpy array
    py_segment = py.numpy.array(segment);
    
    % Segment audio and extract features using Python functions
    py_segments = audio.audio_segmentation(py_segment, sr);
    segment_features = audio.audio_features_extraction(py_segments, sr);
    features{i} = audio.aggregate_features(segment_features);
end

% Convert features from Python to MATLAB format
for i = 1:size(new_segments, 1)
    features{i} = double(features{i});
end

% Calculate Pearson correlation between the first segment and each other segment
mean_features_1 = py.numpy.array(features{1}');  % Mean features for the first segment

for i = 2:length(features)
    mean_features_current = py.numpy.array(features{i}');  % Mean features for current segment
    
    % Calculate Pearson correlation between segment 1 and the current segment
    pearson_corr = audio.calculate_pearson_corr(mean_features_1, mean_features_current);
    fprintf("Correlation between segment 1 and segment %d: %4f\n", i, pearson_corr);

    % Determine if the current segment is likely from the same speaker as the first segment
    if pearson_corr >= 0.8
        disp("Same person as in the first segment");
    else
        disp("Not the same person.");
    end
end
