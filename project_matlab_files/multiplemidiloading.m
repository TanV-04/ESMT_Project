% DATASET LOADING

%javaaddpath('C:\Program Files\MATLAB\R2024a\java\jar/KaraokeMidiJava.jar'); %this is for the readmidi_java function
% Specify the folder containing MIDI files
folderPath = "D:\TANVI_COLLEGE_FILES\ESMT_PROJECT_SEM5\maestro-v3.0.0-midi\maestro-v3.0.0\2006";

% Get a list of all MIDI files in the folder
midiFiles = dir(fullfile(folderPath, '*.midi'));

% Preallocate a cell array to store MIDI data
numFiles = length(midiFiles);
midiData = cell(1, numFiles);

% Loop through each file and read it using readmidi_java
for i = 1:numFiles
    % Construct full file path
    filePath = fullfile(folderPath, midiFiles(i).name);
    
    % Read MIDI data using readmidi_java
    midiMatrix = readmidi(filePath);
    
    % Store the MIDI data matrix in the cell array
    midiData{i} = midiMatrix;
    
    % Display the size of each MIDI matrix (optional)
    [rows, cols] = size(midiMatrix);
    fprintf('Loaded file %d: %s, Size: %d x %d\n', i, midiFiles(i).name, rows, cols);
end

% Now midiData contains all the loaded MIDI files' data as matrices

%%

% DATASET PREPROCESSING

pitchColumn = 4;    % MIDI pitch values
timeColumn = 1;     % Time in seconds
durationColumn = 7; % Duration in seconds

% Now you can extract the relevant data:
for i = 1:numFiles
    midiMatrix = midiData{i};
    
    % Extract pitch, time, and duration
    pitchData{i} = midiMatrix(:, pitchColumn);
    timeData{i} = midiMatrix(:, timeColumn);
    durationData{i} = midiMatrix(:, durationColumn);
end

sequenceLength = 100;  % Set the length of each input sequence

% Precompute start and end indices for each MIDI file
sequenceIndices = zeros(numFiles, 2); % To store start and end indices for each file's sequences
currentIndex = 1;

for i = 1:numFiles
    numNotes = length(pitchData{i});
    numSequences = numNotes - sequenceLength;
    
    % Start and end index for this file
    sequenceIndices(i, 1) = currentIndex;
    sequenceIndices(i, 2) = currentIndex + numSequences - 1;
    
    % Update the current index
    currentIndex = currentIndex + numSequences;
end

% Initialize a cell array to hold the results for each parallel iteration
inputSequencesCell = cell(numFiles, 1);
outputNotesCell = cell(numFiles, 1);

% Parallel loop through each MIDI file
parfor i = 1:numFiles
    numNotes = length(pitchData{i});  % Number of notes in the current MIDI file
    numSequences = numNotes - sequenceLength;  % Number of sequences in this file
    
    % Temporary storage for this iteration (to avoid conflicts in parallel execution)
    %tempInput = zeros(numSequences, sequenceLength * 3);  % For pitch, time, duration
    %tempOutput = zeros(numSequences, 3);  % For the next pitch, time, duration
    tempInput = zeros(numSequences, sequenceLength*3);  % For the next pitch, time, duration
    tempOutput = zeros(numSequences, 3); % For the next pitch, time, duration
    
    % Generate sequences from the current MIDI file
    for j = 1:numSequences
        % Reshape and store input sequences and output notes in temporary arrays
        % creates a sequence of length 100 x 3 (features: pitch, time
        % duration) --> each seq contains 100 time steps and 3 features
        tempInput(j, :) = reshape([pitchData{i}(j:j + sequenceLength - 1), ...
                                   timeData{i}(j:j + sequenceLength - 1), ...
                                   durationData{i}(j:j + sequenceLength - 1)]', 1, []);
        tempOutput(j, :) = [pitchData{i}(j + sequenceLength), ...
                            timeData{i}(j + sequenceLength), ...
                            durationData{i}(j + sequenceLength)];
    end
    
    % Store the results in cell arrays (to avoid slicing issues)
    inputSequencesCell{i} = tempInput;
    outputNotesCell{i} = tempOutput;
end

% Concatenate the results after the parallel loop
inputSequences = vertcat(inputSequencesCell{:});
outputNotes = vertcat(outputNotesCell{:});

size(inputSequences);

inputSize = size(inputSequences, 2);  % Number of features (300 for pitch, time, duration)
outputSize = size(outputNotes, 2);  % Number of output notes (pitch, time, duration)

% Define the layers
% definition of a typical architecture for sequence processing
layers = [ ...
    sequenceInputLayer(inputSize, Name="input")
    lstmLayer(128, 'OutputMode', 'sequence', Name="lstm1") % indicates that the architecture is based on LSTM units
    lstmLayer(128, 'OutputMode', 'sequence', Name="lstm2")
    fullyConnectedLayer(outputSize, Name="fc")
    regressionLayer(Name="output")];

% neural network definition
% Define training options
options = trainingOptions('adam', ...
    'MaxEpochs', 80, ...
    'MiniBatchSize', 64, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {inputSequences, outputNotes}, ...
    'ValidationFrequency', 30, ...
    'Plots', 'training-progress');

% Train the network
net = trainNetwork(inputSequences, outputNotes, layers, options);

% Use a seed sequence from the training data (or create your own seed)
% a seed sequence refers to a specific set of input patterns used to
% initialize or guide the responses of a chatbot. it establishes a context
% of a starting point for interactions
% they're initial prompts or queries that lead to meaningful dialogue.
seedSequence = inputSequences(1:10, :);  % Example seed sequence (10 time steps)

% Generate music based on the seed sequence
generatedNotes = predict(net, seedSequence);

% Convert generated notes to a MIDI file format
newMidiData = pianoRoll2midi(generatedNotes);

% Write the generated MIDI data to a file
midiWrite('generatedMusic.mid', newMidiData);

disp('New music generated and saved as generatedMusic.mid');

% analyse the music generated
pitchColumn = 4;
timeColumn = 1;
durationColumn = 7;

pitchData = generatedMidi(:, pitchColumn);
timeData = generatedMidi(:, timeColumn);
durationData = generatedMidi(:, durationColumn);

figure;
plot(timeData, pitchData, 'o'); % pitch contour
title('pitch vs time for generated music');
xlabel('time (s)');
ylabel('pitch');

% performing spectral analysis on generated midi
% create a spectrogram to analyze
% colors represent the amplitude or intensity of a particular frequency
% brighter colors or higher peaks indicate louder frequencies
[s, f, t] = spectrogram(pitchData, 256, 250, 256, 1E3); % 1kHz sample rate for MIDI-like data
figure;
surf(t, f, abs(s));
shading interp;
title('spectrogram of generated music'); % a spectrogram is a visual representation of the strength of a signal over time at diff frequencies. used to analyze audio, vibrations and seismic signals
xlabel('time (s)');
ylabel('frequency (Hz)');