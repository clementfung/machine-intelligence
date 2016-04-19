clear all;
close all;
trainData = csvread('data/train_sample_features.csv', 3);
[n, p ]= size(trainData);
X = trainData(:,1:p-2);
Y = trainData(:, p);
p = p - 2;

[trainInd,valInd,testInd] = dividerand(1:n);
rel_cols = 1:6;

trainData = [X(trainInd,rel_cols) Y(trainInd)];
testData = X(testInd, rel_cols);

fismat = genfis1(trainData);
predFis = anfis(trainData, fismat);
