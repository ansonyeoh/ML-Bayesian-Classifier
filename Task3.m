%% Task3.m
% Computer Based Test 1
% Xiang Yang 6/11/2017
clear all; close all;

%% Load the data
load cbt1data.mat
x = [diseased; healthy];
t = [ones(300,1); 2*ones(500,1)];
N = size(x,1);

%% Run over 2-fold cross-validation 
K =2; % 2-fold CV
Nrep = 10; % Repeat N times
% For sum up the error and calculagte the average
sum_error_train_with = 0;
sum_error_train_without = 0;
sum_error_test_with = 0;
sum_error_test_without = 0;

for rep = 1:Nrep % Repeat loop
    sizes = repmat(floor(N/K),1,K);
    sizes(end) = sizes(end) + N - sum(sizes);
    csizes = [0 cumsum(sizes)];
    for fold = 1:K
        order = randperm(N); % Make the assignment random
        foldindex = order(csizes(fold)+1:csizes(fold+1));
        X_test = x(foldindex,:); % Get one fold of test data
        T_test = t(foldindex); % get the corresponding label as X_test
        X_train = x;
        X_train(foldindex,:) =[]; % The whole training data apart from the X_test
        T_train = t;
        T_train(foldindex) =[];% The whole labels except test labels

       %% Fit class-conditional Gaussians for each class, from training samples
        % Using the Naive (independence) assumption
        cl = unique(T_train);
        class_mean = [];
        class_var = [];
        for c = 1:length(cl)
            pos = find(T_train==cl(c));
            % Find the means
            class_mean(c,:) = mean(X_train(pos,:)); % class-wise & attribute-wise mean
            class_var(c,:) = var(X_train(pos,:),1); % class-wise & attribute-wise variance
        end  

       %% Compute the predictive probabilities (with Naive assumption)
        % for training samples and testing samples
        probab_train = [];
        probab_test = [];
        for c = 1:length(cl)
            sigmac = diag(class_var(c,:));

            diff_train = [X_train(:,1)-class_mean(c,1) X_train(:,2)-class_mean(c,2)];
            const_train = 1/sqrt((2*pi)^size(X_train,2) * det(sigmac));
            probab_train(:,c) = const_train*exp(-0.5*diag(diff_train*inv(sigmac)*diff_train'));

            diff_test = [X_test(:,1)-class_mean(c,1) X_test(:,2)-class_mean(c,2)];
            const_test = 1/sqrt((2*pi)^size(X_test,2) * det(sigmac));
            probab_test(:,c) = const_test*exp(-0.5*diag(diff_test*inv(sigmac)*diff_test'));
            % this is using maximum likelihood, given the uniform size of classes
        end
        % get proper probability estimates
        probab_train = probab_train./repmat(sum(probab_train,2),[1,2]);
        probab_test = probab_test./repmat(sum(probab_test,2),[1,2]);

        %% find class label predictions from probabilities (with Naive assumption)
        [~,p_train_with] = max(probab_train,[],2); % assign labels as per highest probability
        error_train_with=sum(T_train~=p_train_with); % error - # of mis-classifications
        [~,p_test_with] = max(probab_test,[],2); % assign labels as per highest probability
        error_test_with=sum(T_test~=p_test_with); % error - # of mis-classifications

       %% Plot the data and predictions (with Naive assumption)
        cl = unique(T_train); % find the number of unique classes from labels
        col_train = {'go','bs'};
        col_test = {'rx','yd'};
        figure(2);
        hold on
        for c = 1:length(cl)
            pos_train = find(T_train==cl(c));
            pos_test = find(p_test_with==cl(c));
            plot(X_train(pos_train,1),X_train(pos_train,2),col_train{c},...
                'markersize',10,'linewidth',2);
            plot(X_test(pos_test,1),X_test(pos_test,2),col_test{c},...
                'markersize',10,'linewidth',2);
            title("2-fold CV of ML with Naive Assumption (repeat "+ Nrep +" times)");
            legend('Train->Diseased', 'Test->Diseased','Train->Healthy','Test->Healthy');
        end
        
       %% Fit class-conditional Gaussians for each class, from training samples
        % without using Naive assumption
        class_mean = [];
        class_var = [];
        for c = 1:length(cl)
            pos = find(T_train==cl(c));
            % Find the means
            class_mean(c,:) = mean(X_train(pos,:)); % class-wise & attribute-wise mean
            class_var(:,:,c) = cov(X_train(pos,:),1); % class-wise & attribute-wise co-variance
        end
        
       %% Compute the predictive probabilities (without Naive assumption)
        % for training samples and testing samples
        probab_train = [];
        probab_test = [];
        for c = 1:length(cl)
            sigmac = class_var(:,:,c); % this is the main difference, in with/without using Naive assumption
            diff_train = [X_train(:,1)-class_mean(c,1) X_train(:,2)-class_mean(c,2)];
            const_train = 1/sqrt((2*pi)^size(X_train,2) * det(sigmac));
            probab_train(:,c) = const_train*exp(-0.5*diag(diff_train*inv(sigmac)*diff_train'));
            diff_test = [X_test(:,1)-class_mean(c,1) X_test(:,2)-class_mean(c,2)];
            const_test = 1/sqrt((2*pi)^size(X_test,2) * det(sigmac));
            probab_test(:,c) = const_test*exp(-0.5*diag(diff_test*inv(sigmac)*diff_test'));
            % this is using maximum likelihood, given the uniform size of classes
        end
        % get proper probability estimates
        probab_train = probab_train./repmat(sum(probab_train,2),[1,2]);
        probab_test = probab_test./repmat(sum(probab_test,2),[1,2]);
       
       %% find class label predictions from probabilities (without Naive assumption)
        [~,p_train_without] = max(probab_train,[],2); % assign labels as per highest probability
        error_train_without=sum(T_train~=p_train_without); % error - # of mis-classifications
        [~,p_test_without] = max(probab_test,[],2); % assign labels as per highest probability
        error_test_without=sum(T_test~=p_test_without); % error - # of mis-classifications
       %% Plot the data and predictions (without Naive assumption)
        col_train = {'go','bs'};
        col_test = {'rx','yd'};
        figure(3);
        hold on
        for c = 1:length(cl)
            pos_train = find(T_train==cl(c));
            pos_test = find(p_test_without==cl(c));
            plot(X_train(pos_train,1),X_train(pos_train,2),col_train{c},...
                'markersize',10,'linewidth',2);
            plot(X_test(pos_test,1),X_test(pos_test,2),col_test{c},...
                'markersize',10,'linewidth',2);
            title("2-fold CV of ML without Naive Assumption (repeat "+ Nrep +" times)");
            legend('Train->Diseased', 'Test->Diseased','Train->Healthy','Test->Healthy');
        end 
    end
    sum_error_train_with = sum_error_train_with + error_train_with;
    sum_error_train_without = sum_error_train_without + error_train_without;
    sum_error_test_with = sum_error_test_with + error_test_with;
    sum_error_test_without = sum_error_test_without + error_test_without;
end

f = figure("Position", [500 500 350 60]);
data = {sum_error_train_with/Nrep sum_error_test_with/Nrep; sum_error_train_without/Nrep sum_error_test_without/Nrep};
rowname = {'ML with Naive'; 'ML without Naive'};
columnname = {'Train Error'; 'Test Error'};
uitable('Position',[0 0 318 56],'Data',data,'Columnname',columnname,'Rowname',rowname);