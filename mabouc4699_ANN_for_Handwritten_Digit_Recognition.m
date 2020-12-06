% Initialize Labels and Sets variables with data
tr_labels = loadMNISTLabels('training_label');
tr_set = loadMNISTImages('training_set');
ts_labels = loadMNISTLabels('test_label');
ts_set = loadMNISTImages('test_set');

%Calculate size of tr_labels for use in segmenting the training data to a 
%... data set = 80% initial training and 20% reserved data. Targets set to 
%... tr_labels and targets_test set to ts_labels. Targets and targets_test 
%... set to 10 where they equal 0. Inputs and inputs_test set to tr_set 
%... and ts_set respectively.
n = size(tr_labels, 1);
m = floor(n * 0.80);
targets  = tr_labels;
targets(targets == 0) = 10;
targetsd = dummyvar(targets);
targets_test = ts_labels;
targets_test(targets_test == 0) = 10;
targetsd_test = dummyvar(targets_test);
inputs = tr_set;
inputs_test = ts_set;

%Transpose targets, targetsd, targets_test, and targetsd_test
targets = targets';
targetsd = targetsd';
targets_test = targets_test';
targetsd_test = targetsd_test';

% x1 and y1 represent data and labels for 4/5 of the training data; x2 and 
%... y2 represent 1/5 of the original training data set.
x1 = inputs(:,1:m);
y1 = targetsd(:, 1:m);
x2 = inputs(:, m:n);
y2 = targets(m:n);
y2d = targetsd(:, m:n);

%Trains the network using the trainscg function using sweep matrices values
%... for number of hidden values. Model is a cell matrix array that is of 
%... length of sweep matrix by 1 dimensionality. Resulting trained NN saved
%... to models at index(i). Predicted values saved to p and e.
%... Hidden Layer Size to choose from: 40 and 60
%... Training Functions to choose from: trainscg, traincgb, or traincgp
sweep = [60];
models = cell(length(sweep), 1);
for i = 1:length(sweep)
    hiddenLayerSize = sweep(i);
    net = patternnet(hiddenLayerSize);
    net.trainFcn = 'trainscg';
    [models{i}, tr] = train(net, x1, y1);
    e = models{i}(x1);
    p = models{i}(x2);
    [~, p] = max(p);
end

%Another scores matrix is made to hold results from using the testing data
%... on the trained network. Results are saved to q variable, and like the
%... same operations executed earlier
for i = 1:length(sweep)
    q = models{i}(inputs_test);
    [~, q] = max(q);
end

%Begin systematic testing: Plot Regression + testing_Accuracy script
%... testing_Accuracy script calculates the confusion matrix for each
%... segment of code: training input 80%, training input 20% reserved, and
%... testing input. It uses the function (TP + TN)/(TP + TP + FP + FN) to 
%... calculate the accuracy of the network for all three possible inputs

%Plots the regression of both training data 
plotregression(y1, e, 'Initial', y2, p, 'Train', targetsd_test, models{1}(inputs_test), 'Test')

%Setting the elements in the q array = 0 where they equal 10 (
%... Results in a fix to the issue of displaying the labels above 0 imgs).
%... Plots 80 28x28 images with their labels identified by the trained
%... network.
q(q == 10) = 0;
figure('Name','Digit Identification','NumberTitle','off');
for i = 1:80
    subplot(10,8,i)
    subimg = reshape(inputs_test(:,i), 28, 28, []);
    imshow(subimg), title(q(i));
end