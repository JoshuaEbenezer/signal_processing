clear;
clc;
close all;

fnames=dir('train/*.mat');
normal = zeros(50,2,38400);
arrhy = zeros(50,2,38400);

test_normal = zeros(5,2,38400);
test_arrhy = zeros(5,2,38400);

Fs = 128;
% number of sampling instances
N = 38400;
% frequency base
f = (-N/2:N/2-1)*Fs/N;
% time base
t = 0:1/Fs:5*60-1/Fs;

data_size = length(fnames); % no of samples (+ve and -ve)
val_index = 1:5;    % 5 fold cross validation
data_indices = 1:data_size;    % indices of names of files in database
normal_indices = 1:data_size/2;
abnormal_indices = data_size/2+1:data_size;

%% Butterworth lowpass filter
n = 4;
Wn =12/(Fs/2);
% Zero-Pole-Gain design
[z,p,k] = butter(n,Wn,'low');
sos = zp2sos(z,p,k);

precision = zeros(1,5);
recall = zeros(1,5);

for val_index = 1    % cross validation folds
    test_indices_normal = (val_index-1)*5+1:val_index*5;        % test indices
    test_indices_abnormal = data_size/2 + (val_index-1)*5+1:data_size/2 + val_index*5;
    ntest_indices_bool = ismember(normal_indices,test_indices_normal);
    antest_indices_bool = ismember(abnormal_indices,test_indices_abnormal);
    train_indices_normal = normal_indices(~ntest_indices_bool);    % train indices
    train_indices_abnormal = abnormal_indices(~antest_indices_bool);
    
    index = 0;
    target = ones(1,50);
    target(1:25)=0;
    norm_energy =zeros(data_size/2-5,2,1);
    abnorm_energy = zeros(data_size/2-5,2,1);

    norm_autocorr = zeros(data_size/2-5,2,1);
    abnorm_autocorr = zeros(data_size/2-5,2,1);
    
    norm_mean =zeros(data_size/2-5,2,1);
    abnorm_mean = zeros(data_size/2-5,2,1);
    
    norm_range =zeros(data_size/2-5,2,1);
    abnorm_range = zeros(data_size/2-5,2,1);
    
    norm_std =zeros(data_size/2-5,2,1);
    abnorm_std = zeros(data_size/2-5,2,1);
    
    norm_dwt4 =zeros(data_size/2-5,2,2407);
    abnorm_dwt4 = zeros(data_size/2-5,2,2407);
    
    norm_skew = zeros(20,2,1);
    abnorm_skew = zeros(20,2,1);
    
    ntest_skew = zeros(5,2,1);
    atest_skew = zeros(5,2,1);
    
    norm_kur = zeros(20,2,1);
    abnorm_kur = zeros(20,2,1);
    
    ntest_kur = zeros(5,2,1);
    atest_kur = zeros(5,2,1);
    
    %% Training of normal instances
    for ntrain_index = train_indices_normal
        % Read the data and normalize it
        index = index+1;
        file=fullfile('train',fnames(ntrain_index).name);
        s = load(file);
        normal(index,:,:) = L2normalize(s);
        
        % Find the energy of the filtered signals
        % Find auto-correlations of original signals
        % Define the product(s) as features for each signal
        [norm_energy(index,:),norm_autocorr(index,:),norm_mean(index,:),norm_std(index,:),norm_range(index,:),norm_dwt4(index,:,:),norm_skew(index,:),norm_kur(index,:)] = feature_extract(normal(index,:,:),sos);
    end
    index = 0;
    norm_features = cat(1,norm_energy(:,:)',norm_autocorr(:,:)',norm_mean(:,:)',norm_std(:,:)',norm_range(:,:)',norm_dwt4(:,:)',norm_skew(:,:)',norm_kur(:,:)');
    %% Training of abnormal samples
    for antrain_index = train_indices_abnormal
        index = index+1;
        % Read the data and normalize it
        file=fullfile('train',fnames(antrain_index).name);
        s = load(file);
        arrhy(index,:,:) = L2normalize(s);
        
        % Find the energy of the filtered signals
        % Find auto-correlations of original signals
        [abnorm_energy(index,:),abnorm_autocorr(index,:),abnorm_mean(index,:),abnorm_std(index,:),abnorm_range(index,:),abnorm_dwt4(index,:,:),abnorm_skew(index,:),abnorm_kur(index,:)] = feature_extract(arrhy(index,:,:),sos);
    end
    abnorm_features = cat(1,abnorm_energy(:,:)',abnorm_autocorr(:,:)',abnorm_mean(:,:)',abnorm_std(:,:)',abnorm_range(:,:)',abnorm_dwt4(:,:)',abnorm_skew(:,:)',abnorm_kur(:,:)');

    %% -------------------------------- Validation ------------------------- %%
    tp = 0;
    tn = 0;
    fp=0;
    fn=0;
    
    nindex = 0;     % index for normal test cases
    
    ntest_energy_val = zeros(5,2);
    ntest_auto_corr = zeros(5,2);
    ntest_mean = zeros(5,2);
    ntest_range = zeros(5,2);
    ntest_std = zeros(5,2);
    ntest_dwt4 = zeros(5,2,2407);
    
    atest_energy_val = zeros(5,2);
    atest_auto_corr = zeros(5,2);
    atest_mean = zeros(5,2);
    atest_range = zeros(5,2);
    atest_std = zeros(5,2);
    atest_dwt4 = zeros(5,2,2407);
    
    %% Testing for normal test cases
    for ntest_index=test_indices_normal
        file=fullfile('train',fnames(ntest_index).name);
        s = load(file);
        
        nindex = nindex+1;
        
        test_normal(nindex,:,:) = L2normalize(s);
        % Find energy of the signal and normalize
        [ntest_energy_val(nindex,:),ntest_auto_corr(nindex,:),ntest_mean(nindex,:),ntest_std(nindex,:),ntest_range(nindex,:),ntest_dwt4(nindex,:,:),ntest_skew(nindex,:),ntest_kur(nindex,:)]=feature_extract(test_normal(nindex,:,:),sos);
        
    end
    
    ntest_features = cat(1,ntest_energy_val(:,:)',ntest_auto_corr(:,:)',ntest_mean(:,:)',ntest_std(:,:)',ntest_range(:,:)',ntest_dwt4(:,:)',ntest_skew(:,:)',ntest_kur(:,:)');
    
    aindex = 0;     % index for abnormal test cases
    
    %% Testing for abnormal test cases
    for atest_index = test_indices_abnormal
        
        file=fullfile('train',fnames(atest_index).name);
        s = load(file);
        
        aindex = aindex+1;
        
        test_arrhy(aindex,:,:)=L2normalize(s);
                
        [atest_energy_val(aindex,:),atest_auto_corr(aindex,:),atest_mean(aindex,:),atest_std(aindex,:),atest_range(aindex,:),atest_dwt4(aindex,:,:),atest_skew(aindex,:),atest_kur(aindex,:)]=feature_extract(test_arrhy(aindex,:,:),sos);

    end
    
    atest_features = cat(1,atest_energy_val(:,:)',atest_auto_corr(:,:)',atest_mean(:,:)',atest_std(:,:)',atest_range(:,:)',atest_dwt4(:,:)',atest_skew(:,:)',atest_kur(:,:)');

    features = cat(2,norm_features,ntest_features,abnorm_features,atest_features);
    train_features = cat(2,norm_features,abnorm_features);

    numberOfDimensions = 20;
    [coeff,score] = pca(train_features');
    reducedDimension = coeff(:,1:numberOfDimensions);
    reducedFeatures = (features' * reducedDimension)';
end



function [normal] = L2normalize(s)
normal=s.val;
% Find energy of the signal and normalize
energy_val = sum(normal(1,:).^2);
energy2 = sum(normal(2,:).^2);
normal(1,:)=normal(1,:)/sqrt(energy_val);
normal(2,:)=normal(2,:)/sqrt(energy2);
end


function [energy,auto_corr,mean_val,std_dev,range_val,dwt4,skew,kur] = feature_extract(signal,sos)

s1=signal(1,1,:);
s2=signal(1,2,:);

% filter the signals
y1 = sosfilt(sos,s1);
y2 = sosfilt(sos,s2);
energy(1) = sum(y1.^2);
energy(2) = sum(y2.^2);

% Auto-correlation of original signals
% 128 samples corresponds to 10s
auto_corr(1) = sum(s1(1,1,1:end-128).*s1(1,1,129:end));
auto_corr(2) = sum(s2(1,1,1:end-128).*s2(1,1,129:end));

mean_val(1) = mean(s1);
mean_val(2) = mean(s2);

std_dev(1) = std(s1);
std_dev(2) = std(s2);

range_val(1) = range(s1);
range_val(2) = range(s2);

[C,~] = wavedec(s1(:),4,'db4');
dwt4(1,:)=C(2407:4813);

[C,~] = wavedec(s2(:),4,'db4');
dwt4(2,:)=C(2407:4813);

skew(1) = skewness(s1);
skew(2) = skewness(s2);

kur(1) = kurtosis(s1);
kur(2) = kurtosis(s2);



%energy_val = energy(1)*abs(auto_corr(1));
%auto_corr = energy(2)*abs(auto_corr(2));

end


