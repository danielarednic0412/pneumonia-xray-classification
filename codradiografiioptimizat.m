%% 1. Configurarea Căilor și Încărcarea Datelor
% Folosim calea absolută pentru a evita erorile de folder
datasetPath = 'C:\Users\Pc\Desktop\dataset chest xrays\chest_xray'; 

% Crearea stocului de date pentru antrenare, validare și testare
imdsTrain = imageDatastore(fullfile(datasetPath, 'train'), ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');

imdsVal = imageDatastore(fullfile(datasetPath, 'val'), ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');

imdsTest = imageDatastore(fullfile(datasetPath, 'test'), ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');

%% 2. Vizualizarea Datelor
disp('Distribuția claselor în setul de antrenare:');
countEachLabel(imdsTrain)

figure;
perm = randperm(length(imdsTrain.Files), 9);
for i = 1:9
    subplot(3,3,i);
    imshow(readimage(imdsTrain, perm(i)));
    title(char(imdsTrain.Labels(perm(i))));
end
sgtitle('Exemple de Radiografii din Dataset');

%% 3. Pregătirea Rețelei (Transfer Learning + Class Weights)
net = resnet18; 
inputSize = net.Layers(1).InputSize; 
lgraph = layerGraph(net);
numClasses = numel(categories(imdsTrain.Labels));

newDropoutLayer = dropoutLayer(0.5, 'Name', 'new_dropout');
newConnectedLayer = fullyConnectedLayer(numClasses, 'Name', 'fc_new', ...
    'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10);

% --- MODIFICARE NOUĂ: Calcularea Ponderilor (Class Weights) ---
labelCounts = countEachLabel(imdsTrain);
totalImages = sum(labelCounts.Count);
classWeights = totalImages ./ (numClasses * labelCounts.Count);

newClassificationLayer = classificationLayer('Name', 'class_output', ...
    'Classes', labelCounts.Label, ...
    'ClassWeights', classWeights);
% --------------------------------------------------------------

lgraph = replaceLayer(lgraph, 'fc1000', newConnectedLayer);
lgraph = replaceLayer(lgraph, 'ClassificationLayer_predictions', newClassificationLayer);

%% 4. Preprocesare (Data Augmentation)
% --- MODIFICARE NOUĂ: Augmentarea Datelor ---
imageAugmenter = imageDataAugmenter( ...
    'RandRotation', [-10, 10], ...      
    'RandXTranslation', [-15, 15], ...  
    'RandYTranslation', [-15, 15]);     

% Aplicăm augmentarea doar pe antrenare
augImdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    'DataAugmentation', imageAugmenter, ...
    'ColorPreprocessing', 'gray2rgb');
% --------------------------------------------

augImdsVal = augmentedImageDatastore(inputSize(1:2), imdsVal, 'ColorPreprocessing', 'gray2rgb');
augImdsTest = augmentedImageDatastore(inputSize(1:2), imdsTest, 'ColorPreprocessing', 'gray2rgb');

%% 5. Opțiuni de Antrenare
options = trainingOptions('adam', ...
    'MiniBatchSize', 32, ...
    'MaxEpochs', 5, ...
    'InitialLearnRate', 1e-4, ...
    'ValidationData', augImdsVal, ...
    'ValidationFrequency', 10, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

%% 6. Antrenarea Rețelei și Salvarea Automată
disp('Începe antrenarea modelului optimizat...');
trainedNet_Optimizat = trainNetwork(augImdsTrain, lgraph, options);

% Salvează modelul automat ca să nu îl pierzi!
save('ModelPneumonie_Optimizat.mat', 'trainedNet_Optimizat');
disp('Antrenarea s-a terminat. Modelul a fost salvat ca ModelPneumonie_Optimizat.mat');

%% 7. Evaluarea Performanței pe Setul de Test
disp('Se evaluează performanța pe setul de testare...');
% Am adăugat MiniBatchSize 16 pentru a preveni eroarea "Out of memory"
[YPred, scores] = classify(trainedNet_Optimizat, augImdsTest, 'MiniBatchSize', 16);
YTest = imdsTest.Labels;
accuracy = mean(YPred == YTest);

fprintf('Acuratețea pe setul de testare: %.2f%%\n', accuracy * 100);

% Afișarea matricei de confuzie
figure;
confusionchart(YTest, YPred);
title('Matricea de Confuzie - Model Optimizat');