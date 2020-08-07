
clear all; close all; clc

% load catData_w.mat
% load dogData_w.mat

perm = randperm(42000); % pick indicies randomly
trainingIndex = perm(1:33600);
testingIndex = perm(33601:end);

digitTrain = csvread('train.csv',1,0);
TTrain = digitTrain(:,1);
%TTrain = categorical(TTrain);
digitTrain = digitTrain(:,2:785);
%%

trainingData = digitTrain(trainingIndex,:);
LTrain = TTrain(trainingIndex); % label of training data
testingData = digitTrain(testingIndex,:);
LTest = TTrain(testingIndex); % label of testing data
label=[LTrain; LTest];
CD=[trainingData; testingData];

labels = zeros(2,33600);
for i = 1:33600
    j = LTrain(i); % note that this ranges from 0 to 9
    % we add one because we can't index at zero
    labels(j+1,i) = 1;
end

x = trainingData';
x2 = testingData';

net = patternnet(10,'trainscg');
net.layers{1}.transferFcn = 'tansig';

net = train(net,trainingData',labels);
view(net)
y = net(x);
y2= net(x2);
perf = perform(net,labels,y);
classes2 = vec2ind(y); % the training data set
classes3 = vec2ind(y2); % the testing data set lables
% note that this ranges from 1 to 10

subplot(4,1,1), bar(y(1,:),'FaceColor',[.6 .6 .6],'EdgeColor','k')
subplot(4,1,2), bar(y(2,:),'FaceColor',[.6 .6 .6],'EdgeColor','k')
subplot(4,1,3), bar(y2(1,:),'FaceColor',[.6 .6 .6],'EdgeColor','k')
subplot(4,1,4), bar(y2(2,:),'FaceColor',[.6 .6 .6],'EdgeColor','k')

misclassified = LTest' - classes3; % the number misclassified
for i = 1:length(misclassified)
    if misclassified(i) == -1 % correct, because 
        misclassified(i) = 0;
    else
        misclassified(i) = 1;
    end
end

accuracy = 1 - mean(misclassified)
