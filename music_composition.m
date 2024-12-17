javaaddpath('C:\Program Files\MATLAB\R2024a\java\jar/KaraokeMidiJava.jar'); 
folderPath = "D:\Tanush\BTech Symbi\Sem 5\ESMT\mini project\dataset\2018";

% get a list of all MIDI files in the folder
midiFiles = dir(fullfile(folderPath, '*.midi'));

% initialize an empty matrix to hold all MIDI data
allNotes = [];

for file = midiFiles'
    % Read each MIDI file
    midiMatrix = readmidi_java(fullfile(folderPath, file.name));
    
    pitchColumn = 4;    % MIDI pitch values
    durationColumn = 2; % Note duration in beats

    % Extract pitch and duration
    pitch_values = midiMatrix(:, pitchColumn);
    duration_values = midiMatrix(:, durationColumn);

    % Create matrix of duration and pitch for current file
    notes = [duration_values, pitch_values]; 
    allNotes = [allNotes; notes];  % Concatenate with the previous data
end

disp('Total Notes Extracted:');
disp(size(allNotes, 1));
%%
% separate matrices for pitch and duration
pitch_values = allNotes(:, 2);
duration_values = allNotes(:, 1);

%%
% NORMALIZING THE COLUMNS

duration_min = min(duration_values);
duration_max = max(duration_values);
pitch_min = min(pitch_values);
pitch_max = max(pitch_values);


% Normalize pitch and duration
duration_values = (duration_values - duration_min) / (duration_max - duration_min);
pitch_values = (pitch_values - pitch_min) / (pitch_max - pitch_min);

%%
% sequence generation
numNotes = size(pitch_values, 1);
sequenceLength = 10;  % Length of each sequence
numSequences = numNotes - sequenceLength;  % Number of sequences
inputIndices = reshape((1:numSequences)', [], 1) + (0:sequenceLength-1);  % Indices for inputs
%% CREATING INPUT AND OUTPUT SEQUENCES FOR PITCH

pitch_inputs = pitch_values(inputIndices, :);  % Vectorized extraction of inputs
pitch_outputs = pitch_values(sequenceLength+1:numNotes, :);  % Vectorized extraction of outputs

%% CREATING INPUT AND OUTPUT SEQUENCES FOR DURATION

duration_inputs = duration_values(inputIndices, :);  % Vectorized extraction of inputs
duration_outputs = duration_values(sequenceLength+1:numNotes, :);  % Vectorized extraction of outputs

%%
numFeatures = size(pitch_inputs, 2);
numResponses = size(pitch_outputs, 2);
numSequences = size(pitch_inputs, 1) / sequenceLength;  % Number of sequences
%% RESHAPING THE INPUT AND OUTPUT SEQUENCES FOR PITCH

reshaped_pitch_inputs = reshape(pitch_inputs', numFeatures, sequenceLength, numSequences);
reshaped_pitch_outputs = reshape(pitch_outputs', numResponses, numSequences);

%% RESHAPING THE INPUT AND OUTPUT SEQUENCES FOR DURATION

reshaped_duration_inputs = reshape(duration_inputs', numFeatures, sequenceLength, numSequences);
reshaped_duration_outputs = reshape(duration_outputs', numResponses, numSequences);

%%
% Define the LSTM network architecture
% Define shared layers
sharedLayers = [
    sequenceInputLayer([numFeatures sequenceLength])
    flattenLayer
    lstmLayer(128, 'OutputMode', 'sequence')
    dropoutLayer(0.3)
];

% separate output layer for pitch
pitchOutputLayer = [
    fullyConnectedLayer(1)  % Single output for pitch
    regressionLayer];  % Regression layer for pitch output

% separate output layer for duration
durationOutputLayer = [
    fullyConnectedLayer(1)  % Single output for duration
    regressionLayer];  % Regression layer for duration output

% Combine layers for each model
pitchLayers = [sharedLayers; pitchOutputLayer];
durationLayers = [sharedLayers; durationOutputLayer];

% training options 
options = trainingOptions('adam', ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 0.0005, ...
    'GradientThreshold', 1, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.2, ...
    'LearnRateDropPeriod', 20, ...
    'Verbose', 1, ...
    'Plots', 'none', ...
    'Shuffle', 'every-epoch');
%%

% Train each model
pitchNet = trainNetwork(reshaped_pitch_inputs, reshaped_pitch_outputs, pitchLayers, options);
durationNet = trainNetwork(reshaped_duration_inputs, reshaped_duration_outputs, durationLayers, options);


%%
% Generate new sequences 
pitch_seed = pitch_values(1:sequenceLength, :);  % Use the first sequence as the initial seed
duration_seed = duration_values(1:sequenceLength, :);  % Use the first sequence as the initial seed
pitch_predicted = [];
duration_predicted = [];

final_pitch_predicted = [];
final_duration_predicted = [];
%%
% PITCH PREDICTION
for i = 1:10  % Generate 10 new notes
    % Reshape seed for LSTM input
    inputSequence = reshape(pitch_seed', numFeatures, sequenceLength, 1);  % Adjust input shape for prediction
    nextNote = predict(pitchNet, inputSequence);  % Predict the next features
    pitch_predicted(end + 1, :) = nextNote;  % Store predicted features
    % Update seed for next prediction
    pitch_seed = [pitch_seed(2:end, :); nextNote'];  % Shift left and add new predicted note
end

% Combine the seed with predicted notes for the final output
final_pitch_predicted = [pitch_seed; pitch_predicted];  % Final output sequence

%%
% DURATION PREDICTION
for i = 1:10  % Generate 50 new notes
    % Reshape seed for LSTM input
    inputSequence = reshape(duration_seed', numFeatures, sequenceLength, 1);  % Adjust input shape for prediction
    nextNote = predict(durationNet, inputSequence);  % Predict the next features
    duration_predicted(end + 1, :) = nextNote;  % Store predicted features
    % Update seed for next prediction
    duration_seed = [duration_seed(2:end, :); nextNote'];  % Shift left and add new predicted note
end

% Combine the seed with predicted notes for the final output
final_duration_predicted = [duration_seed; duration_predicted];  % Final output sequence


%% 
% time column with an interval of 0.5 seconds for each note
numRows = size(final_pitch_predicted,1);
time = (0.5:0.5:numRows/2)';

%%
% de-normalizing the predicted values
original_duration_values = final_duration_predicted * (duration_max - duration_min) + duration_min;
original_pitch_values = round(final_pitch_predicted * (pitch_max - pitch_min) + pitch_min);

%%
% generating final output file in midi format
playableFile = zeros(numRows,7);

playableFile(:,1) = time;
playableFile(:,2) = original_duration_values;
playableFile(:,3) = 1;
playableFile(:,4) = original_pitch_values;
playableFile(:,5) = 64;

writemidi_java(playableFile, "playableTest3.midi")