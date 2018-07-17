% specify mu (mean) and sigma (covariance) for two classes
% to observe relation between variance/covariance and class shape
clear all;close all;clc;
N=100;
%% identical variance and zero covariance
mu1 = [2,2];    mu2 = [8,8];
sigma1 = [2,0;0,2];   sigma2 = [2,0;0,2];
% generate point clouds
c1pc = mvnrnd(mu1,sigma1,N); %% class 1 point cloud
c2pc = mvnrnd(mu2,sigma2,N); %% class 1 point cloud
% plot the point cloud
figure;
plot(c1pc(:,1),c1pc(:,2),'b+'); hold on
plot(c2pc(:,1),c2pc(:,2),'ro'); hold off

%% non-identical variance and zero covariance
mu1 = [2,2];    mu2 = [8,8];
sigma1 = [2,0;0,3];   sigma2 = [2,0;0,3];
% generate point clouds
c1pc = mvnrnd(mu1,sigma1,N); %% class 1 point cloud
c2pc = mvnrnd(mu2,sigma2,N); %% class 1 point cloud
% plot the point cloud
figure;
plot(c1pc(:,1),c1pc(:,2),'b+'); hold on
plot(c2pc(:,1),c2pc(:,2),'ro'); hold off

%% non-identical variance and identical covariance between class
mu1 = [2,2];    mu2 = [8,8];
sigma1 = [2,2;2,3];   sigma2 = [2,2;2,3];
% generate point clouds
c1pc = mvnrnd(mu1,sigma1,N); %% class 1 point cloud
c2pc = mvnrnd(mu2,sigma2,N); %% class 1 point cloud
% plot the point cloud
figure;
plot(c1pc(:,1),c1pc(:,2),'b+'); hold on
plot(c2pc(:,1),c2pc(:,2),'ro'); hold off

%% non-identical variance and non-identical covariance
mu1 = [2,2];    mu2 = [8,8];
sigma1 = [2,2;2,3];   sigma2 = [2,-2;-2,4];
% generate point clouds
c1pc = mvnrnd(mu1,sigma1,N); %% class 1 point cloud
c2pc = mvnrnd(mu2,sigma2,N); %% class 1 point cloud
% plot the point cloud
figure;
plot(c1pc(:,1),c1pc(:,2),'b+'); hold on
plot(c2pc(:,1),c2pc(:,2),'ro'); hold off
%% non-identical variance and non-identical covariance, but common mean
mu1 = [2,2];    mu2 = [2,2];
sigma1 = [2,2;2,3];   sigma2 = [2,-2;-2,4];
% generate point clouds
c1pc = mvnrnd(mu1,sigma1,N); %% class 1 point cloud
c2pc = mvnrnd(mu2,sigma2,N); %% class 1 point cloud
% plot the point cloud
figure;
plot(c1pc(:,1),c1pc(:,2),'b+'); hold on
plot(c2pc(:,1),c2pc(:,2),'ro'); hold off