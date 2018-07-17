%% Task4.m
% Bayesian classifier
clear all; close all;
%% Load the data
load cbt1data;

X_train = [diseased; healthy];
X_test = unseen;
T_train = [ones(300,1); 2*ones(500,1)]; 
%% Fit class-conditional Gaussians for each class, from training samples
% Using the Naive (independence) assumption
cl=unique(T_train);
for c = 1:length(cl)
    pos = find(T_train==cl(c));
    % Find the means
    class_mean(c,:) = mean(X_train(pos,:)); % class-wise & attribute-wise mean
    class_var(c,:) = var(X_train(pos,:),1); % with naive
end  
%% Compute the predictive probabilities (with Naive assumption)
% for training samples and testing samples
[Xv, Yv] = meshgrid(-2:0.1:12,-2:0.1:12);
Probs =[];
probab_test=[];
for c = 1:length(cl)
    sigmac = diag(class_var(c,:)); %with naive
   
    if c ==1
        probab_prior = 3/8;
    else
        probab_prior = 5/8;
    end
    diff_test = [X_test(:,1)-class_mean(c,1) X_test(:,2)-class_mean(c,2)];
    const_test = 1/sqrt((2*pi)^size(X_test,2) * det(sigmac));
    probab_test(:,c) = probab_prior*const_test*exp(-0.5*diag(diff_test*inv(sigmac)*diff_test'));
    
    temp = [Xv(:)-class_mean(c,1) Yv(:)-class_mean(c,2)];
    Probs(:,:,c) = reshape(probab_prior*const_test*exp(-0.5*diag(temp*inv(sigmac)*temp')),size(Xv));
end
probab_test = probab_test./repmat(sum(probab_test,2),[1,2]);
Probs = Probs./repmat(sum(Probs,3),[1,1,2]);
[~,p_test_with] = max(probab_test,[],2); % assign labels as per highest probability

%% Plot the data and predictions (with Naive assumption)
col_train = {'go','bo'};
col_test = {'gs','bs'};
figure(1);
hold off
for i =1:2
    subplot(1,2,i);
    hold off
    for c = 1:length(cl)
        
        %pos_train = find(T_train==cl(c));  
        %plot(X_train(pos_train,1),X_train(pos_train,2),col_train{c},...
         %   'markersize',5,'linewidth',2);
        
        pos_test = find(p_test_with==cl(c));
        disp(size(pos_test));
        plot(X_test(pos_test,1),X_test(pos_test,2),col_test{c},...
           'markersize',2,'linewidth',2);
        hold on
    end
    contour(Xv,Yv,Probs(:,:,i));
    if i == 1        
        ti = sprintf('MAP with Naive Probability for class diseased');
    else 
        ti = sprintf('MAP with Naive Probability for class healthy');
    end
    title(ti);
    legend('Diseased','Healthy');
end
%% Fit class-conditional Gaussians for each class, from training samples
% Using the Naive (independence) assumption
cl=unique(T_train);
for c = 1:length(cl)
    pos = find(T_train==cl(c));
    % Find the means
    class_mean(c,:) = mean(X_train(pos,:)); % class-wise & attribute-wise mean
    class_var(:,:,c) = cov(X_train(pos,:),1);% without naive
end  
%% Compute the predictive probabilities (with Naive assumption)
% for training samples and testing samples
[Xv, Yv] = meshgrid(-2:0.1:12,-2:0.1:12);
Probs =[];
probab_test=[];
for c = 1:length(cl)
    sigmac = class_var(:,:,c); %without naive
   
    if c ==1
        probab_prior = 3/8;
    else
        probab_prior = 5/8;
    end
    
    diff_test = [X_test(:,1)-class_mean(c,1) X_test(:,2)-class_mean(c,2)];
    const_test = 1/sqrt((2*pi)^size(X_test,2) * det(sigmac));
    probab_test(:,c) = probab_prior*const_test*exp(-0.5*diag(diff_test*inv(sigmac)*diff_test'));
    
    temp = [Xv(:)-class_mean(c,1) Yv(:)-class_mean(c,2)];
    Probs(:,:,c) = reshape(probab_prior*const_test*exp(-0.5*diag(temp*inv(sigmac)*temp')),size(Xv));
end
probab_test = probab_test./repmat(sum(probab_test,2),[1,2]);
Probs = Probs./repmat(sum(Probs,3),[1,1,2]);
[~,p_test_with] = max(probab_test,[],2); % assign labels as per highest probability

%% Plot the data and predictions (with Naive assumption)
cl = unique(T_train); % find the number of unique classes from labels
col_train = {'go','bo'};
col_test = {'gs','bs'};
figure(2);
hold off
for i =1:2
    subplot(1,2,i);
    hold off
    for c = 1:length(cl)
        
        %pos_train = find(T_train==cl(c));  
        %plot(X_train(pos_train,1),X_train(pos_train,2),col_train{c},...
         %   'markersize',5,'linewidth',2);
        
        pos_test = find(p_test_with==cl(c));
        disp(size(pos_test));
        plot(X_test(pos_test,1),X_test(pos_test,2),col_test{c},...
           'markersize',2,'linewidth',2);
        hold on
    end
    contour(Xv,Yv,Probs(:,:,i));
    if i == 1        
        ti = sprintf('MAP without Naive Probability for class diseased');
    else 
        ti = sprintf('MAP without Naive Probability for class healthy');
    end
    title(ti);
    legend('Diseased','Healthy');
end