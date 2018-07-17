%% Task2.m
% Bayesian classifier
clear all; close all;
%% Load the data
load cbt1data;
X_train = [diseased; healthy]; % Padding the data of the entire training data
T_train = [ones(300,1); 2*ones(500,1)]; % Padding the labels 

%% Fit class-conditional Gaussians for each class, from training samples
% Using the Naive (independence) assumption
cl = unique(T_train); % there are two classes, diseased and healthy
for c = 1:length(cl)
    pos = find(T_train==cl(c));
    % Find the means of particular class
    class_mean(c,:) = mean(X_train(pos,:)); % class-wise & attribute-wise mean
    class_var(c,:) = var(X_train(pos,:),1); % class-wise & attribute-wise variance
end  
%% Compute the predictive probabilities (with Naive assumption)
probab_train = [];
for c = 1:length(cl)
    sigmac = diag(class_var(c,:));
    if c ==1
        probab_prior = 3/8;
    else
        probab_prior = 5/8;
    end
    diff_train = [X_train(:,1)-class_mean(c,1) X_train(:,2)-class_mean(c,2)];
    const_train = 1/sqrt((2*pi)^size(X_train,2) * det(sigmac));
    probab_train(:,c) = probab_prior*const_train*exp(-0.5*diag(diff_train*inv(sigmac)*diff_train'));
    % this is using maximum a posteriori estimate, given the uniform size of classes
end
% get proper probability estimates
probab_train = probab_train./repmat(sum(probab_train,2),[1,2]);

%% Fit class-conditional Gaussians for each class, from training samples
% without using Naive assumption
for c = 1:length(cl)
    pos = find(T_train==cl(c));
    % Find the means of particular class
    class_mean(c,:) = mean(X_train(pos,:)); % class-wise & attribute-wise mean
    class_var(:,:,c) = cov(X_train(pos,:),1); % class-wise & attribute-wise co-variance
end

%% Compute the predictive probabilities (without Naive assumption)
probab_train = [];
for c = 1:length(cl)
    sigmac = class_var(:,:,c); % this is the main difference, in with/without using Naive assumption
    if c ==1
        probab_prior = 3/8;
    else
        probab_prior = 5/8;
    end
    diff_train = [X_train(:,1)-class_mean(c,1) X_train(:,2)-class_mean(c,2)];
    const_train = 1/sqrt((2*pi)^size(X_train,2) * det(sigmac));
    probab_train(:,c) = probab_prior*const_train*exp(-0.5*diag(diff_train*inv(sigmac)*diff_train'));
    % this is using maximum a posteriori estimate, given the uniform size of classes
end
% get proper probability estimates
probab_train = probab_train./repmat(sum(probab_train,2),[1,2]);