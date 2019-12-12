clear all
clc
close all


load randomprojection_AR.mat; 
DATA = DATA./ repmat(sqrt(sum(DATA .* DATA)), [size(DATA, 1) 1]); %normalize
c = length(unique(Label));
numClass = zeros(c,1);
for i = 1 : c
    numClass(i, 1) = length(find(Label == i));
end
%% select training and test samples
train_num = 6;
for time = 1 : 10
train_data = []; test_data = []; 
train_label = []; test_label = [];
for i = 1 : c
    index = find(Label == i); 
    randindex = index(randperm(length(index)));
    train_data = [train_data DATA(:,randindex(1 : train_num))];
    train_label = [train_label  Label(randindex(1 : train_num))];
  
    test_data = [test_data DATA(:, randindex(train_num + 1 : end))];
    test_label = [test_label  Label(randindex(train_num + 1 : end))];
end
    
for i = 1 : size(train_data, 2)
    a = train_label(i);
    H_train(a, i) = 1;
end 

%% parameters
alpha = 0.1;
beta = 0.1;
gamma = 0.1;
lambda = 0.01;
[Q, T, M, value_AR] = LRDLSR(train_data, c, H_train, train_num, alpha, beta, gamma, lambda);

%% KNN classfication
T_train = Q * train_data;
T_test = Q * test_data; 
T_train = T_train./ repmat(sqrt(sum(T_train .* T_train)), [size(T_train, 1) 1]);
T_test = T_test./ repmat(sqrt(sum(T_test .* T_test)), [size(T_test, 1) 1]);
mdl = fitcknn(T_train', train_label);
class_test = predict(mdl, T_test');
acc(time) = sum(test_label' == class_test)/length(test_label)*100  
end


mean(acc)
std(acc)
imagesc(T)
colormap(gray(256))


