clear all;clc
load('KSEdata.mat');

% data normalization,means to 0 and deviations to 1
[data, ps] = mapstd(data);

tau = 0.25; % time step
M = 64;  % number of spatial grid points 

inSize = size(data,1);
resSize = 5000;
sparsity = 0.006; % sparsity
k=round(sparsity*resSize);
rho = 0.1; % spectral radius
sigma = 1; 
gamma = 0.9; % leaky rate


initialen = 10000;
trainlen = 20000;
len = initialen+trainlen;
testlen = 10000;

% generate weight matrix
Win1 = sigma*(-1 + 2*rand(resSize,inSize));
adj1 = zeros(resSize,inSize);
for m=1:resSize
    for n=1:inSize
        if(rand(1,1)<0.01)
            adj1(m,n)=1;
        end
    end
end
Win = adj1.*Win1;

adj2 = zeros(resSize,resSize);
for i = 1:resSize
    num = randperm(resSize,k);
    for j = 1:k
        adj2(i,num(j)) = 1;
    end
end
Wres1 = rand(resSize,resSize); 
Wres2 = adj2.*Wres1 ;
SR = max(abs(eig(Wres2))) ;
Wres = Wres2 .* ( rho/SR); 

% training period
states1 = zeros(resSize,len);
for i = 2:len
    ut = data(:,i);
    states1(:,i)=(1-gamma)*states1(:,i-1) + gamma*tanh(Wres*states1(:,i-1) + Win*ut);
end
states = states1(:,initialen:len-1);
states(2:2:resSize,:) = states(2:2:resSize,:).^2; % half neurons are nonlinear(even terms)

% Tikhonov regularization to solve Wout
traindata=data(:,initialen+1:len);
beta = 1e-4; % regularization parameter
Wout = ((states*states' + beta*eye(resSize)) \ (states*traindata'))';
trainoutput = Wout*states;    
mse1=mean(sum((trainoutput-traindata).^2));
r = states(:,end);

% testing period
vv = trainoutput(:,end);
output = [ ];
for i = 1:testlen
    ut = vv ; 
    r = (1-gamma)*r + gamma*tanh(Wres*r + Win*ut);
    r2 = r;
    r2(2:2:resSize)=r2(2:2:resSize).^2;
    vv = Wout * r2;
    output = [output vv];
end

original = data(:,len+1:len+testlen);
predict = output;

lambda_max = 0.05; % largest lyapunov exponent
t = (1:1:testlen)*tau*lambda_max;
s = 1:1:M;

% plot
figure
subplot(3,1,1)
imagesc(t,s,original)
title('Actual')
xlabel('$$\Lambda_{max}t$$', 'Interpreter', 'Latex')
xlim([0, 20])
subplot(3,1,2)
imagesc(t,s,predict)
title('Prediction')
xlabel('$$\Lambda_{max}t$$', 'Interpreter', 'Latex')
xlim([0, 20])
caxis(3*[-1,1])
subplot(3,1,3)
imagesc(t,s,original - predict)
title('Error')
xlabel('$$\Lambda_{max}t$$', 'Interpreter', 'Latex')
caxis(3*[-1,1])
xlim([0, 20])
colormap('jet')
h=colorbar();
pos3=set(h,'Position', [0.93 0.13 0.025 0.78]);