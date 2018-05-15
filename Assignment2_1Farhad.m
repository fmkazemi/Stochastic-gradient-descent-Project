%%%%%%%%%%%%           Farhad Mohammad Kazemi       %%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%           ############         %%%%%%%%%%%%%%%%%%
close all
clc
clear all;
load('A2T1.mat');
x=A2T1(:,1:2);
y=A2T1(:,3);

m = length(y); % store the number of training examples
n = size(x,2); % number of features
theta_batch_vec = [0 0]';
theta_stoch_vec = [0 0]';
alpha = 0.002;
err = [0 0]';
%theta_batch_vec_v = zeros(10000,2);
theta_batch_vec_v = zeros(5000,2);
theta_stoch_vec_v = zeros(500*5000,2);
for kk = 1:5000
	% batch gradient descent - loop over all training set
	h_theta_batch = (x*theta_batch_vec);
	h_theta_batch_v = h_theta_batch*ones(1,n);
	y_v = y*ones(1,n);
	theta_batch_vec = theta_batch_vec - alpha*1/m*sum((h_theta_batch_v - y_v).*x).';
	theta_batch_vec_v(kk,:) = theta_batch_vec;
	j_theta_batch(kk) = 1/(2*m)*sum((h_theta_batch - y).^2);

	% stochastic gradient descent - loop over one training set at a time
	for (jj = 1:500)
		h_theta_stoch = (x(jj,:)*theta_stoch_vec);
		h_theta_stoch_v = h_theta_stoch*ones(1,n);
		y_v = y(jj,:)*ones(1,n);
		theta_stoch_vec = theta_stoch_vec - alpha*1/m*((h_theta_stoch_v - y_v).*x(jj,:)).';
		j_theta_stoch(kk) = 1/(2*m)*sum((h_theta_stoch - y).^2);
		theta_stoch_vec_v(500*(kk-1)+jj,:) = theta_stoch_vec;
	end
end
figure;
j_theta_stoch10epoch=[];
for (f=1:10:5000)
j_theta_stoch10epoch=[j_theta_stoch10epoch,j_theta_stoch(1,f)];
end
plot(1:10:5000,j_theta_stoch10epoch);
xlabel('epochs');
ylabel('J(theta)');
title(sprintf('Stochastic Gradient Descent'));
   

figure;
plot(j_theta_batch);
xlabel('epochs');
ylabel('J(theta)');
title(sprintf('Batch Gradient Descent'));
