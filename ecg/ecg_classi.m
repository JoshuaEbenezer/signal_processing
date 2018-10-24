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

for val_index = 1     % cross validation folds
    test_indices_normal = (val_index-1)*5+1:val_index*5;        % test indices
    test_indices_abnormal = data_size/2 + (val_index-1)*5+1:data_size/2 + val_index*5;
    ntest_indices_bool = ismember(normal_indices,test_indices_normal);
    antest_indices_bool = ismember(abnormal_indices,test_indices_abnormal);
    train_indices_normal = normal_indices(~ntest_indices_bool);    % train indices
    train_indices_abnormal = abnormal_indices(~antest_indices_bool);
    
    index = 0;
    norm_energy =zeros(data_size/2-5,2,1);
    abnorm_energy = zeros(data_size/2-5,2,1);

    norm_autocorr = zeros(data_size/2-5,2,1);
    abnorm_autocorr = zeros(data_size/2-5,2,1);
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
        [norm_energy(index,:),norm_autocorr(index,:)] = feature_extract(normal(index,:,:),sos);
    end
    index = 0;
    
    %% Training of abnormal samples
    for antrain_index = train_indices_abnormal
        index = index+1;
        % Read the data and normalize it
        file=fullfile('train',fnames(antrain_index).name);
        s = load(file);
        arrhy(index,:,:) = L2normalize(s);
        
        % Find the energy of the filtered signals
        % Find auto-correlations of original signals
        [abnorm_energy(index,:),abnorm_autocorr(index,:)] = feature_extract(arrhy(index,:,:),sos);
    end
    
    %% Definition of thresholds
    norm_energy_val = (mean(norm_energy(:,1))*(mean(abs(norm_autocorr(:,1)))));
    norm_measure2 = (mean(norm_energy(:,2))*(mean(abs(norm_autocorr(:,2)))));
    abnorm_energy_val = (mean(abnorm_energy(:,1))*mean(abs(abnorm_autocorr(:,1))));
    abnorm_measure2 = (mean(abnorm_energy(:,2))*mean(abs(abnorm_autocorr(:,2))));
    
%     norm_energy_val = (mean(abs(norm_autocorr(1))));
%     norm_measure2 =(mean(abs(norm_autocorr(2))));
%     abnorm_energy_val =mean(abs(abnorm_autocorr(1)));
%     abnorm_measure2 =mean(abs(abnorm_autocorr(2)));
    fin_norm_thresh = (norm_energy_val+norm_measure2)/2;
    fin_arrhy_thresh = (abnorm_energy_val+abnorm_measure2)/2;
    
    %% -------------------------------- Validation ------------------------- %%
    tp = 0;
    tn = 0;
    fp=0;
    fn=0;
    
    nindex = 0;     % index for normal test cases
    
    %% Testing for normal test cases
    for ntest_index=test_indices_normal
        file=fullfile('train',fnames(ntest_index).name);
        s = load(file);
        
        nindex = nindex+1;
        
        test_normal(nindex,:,:) = L2normalize(s);
        % Find energy of the signal and normalize
        
        [energy_val,measure2]=feature_extract(test_normal(nindex,:,:),sos);
        m1 = energy_val(1)*abs(measure2(1));
        m2 = energy_val(2) * abs(measure2(2));
        norm_true1 = abs(m1-fin_norm_thresh)<abs(m1-fin_arrhy_thresh);
        norm_true2 = abs(m2-fin_norm_thresh)<abs(m2-fin_arrhy_thresh);
        if (norm_true1)
            tp=tp+1;
        else
            fn=fn+1;
        end
        if (norm_true2)
            tp=tp+1;
        else
            fn=fn+1;
        end
    end
    aindex = 0;     % index for abnormal test cases
    
    %% Testing for abnormal test cases
    for atest_index = test_indices_abnormal
        
        file=fullfile('train',fnames(atest_index).name);
        s = load(file);
        
        aindex = aindex+1;
        
        test_arrhy(aindex,:,:)=L2normalize(s);
                
        [energy_val,measure2]=feature_extract(test_arrhy(aindex,:,:),sos);
        m1 = energy_val(1)*abs(measure2(1));
        m2 = energy_val(2)*abs(measure2(2));
        
        abnorm_true1 = abs(m1-fin_norm_thresh)>abs(m1-fin_arrhy_thresh);
        abnorm_true2 = abs(m2-fin_norm_thresh)>abs(m2-fin_arrhy_thresh);
        
        if (abnorm_true1)
            tn=tn+1;
        else
            fp=fp+1;
        end
        if (abnorm_true2)
            tn=tn+1;
        else
            fp=fp+1;
        end
    end
    recall(val_index) =tp/(tp+fn)*100;
    precision(val_index) =tn/(tn+fp)*100;
    
    fprintf('%f is the precision for val round %d\n',precision(val_index),val_index);
    fprintf('%f is the recall for val round %d\n',recall(val_index),val_index);
    
    
end


    fprintf('%f is the mean precision\n',mean(precision));
    fprintf('%f is the mean recall\n',mean(recall));
function [normal] = L2normalize(s)
normal=s.val;
% Find energy of the signal and normalize
energy_val = sum(normal(1,:).^2);
energy2 = sum(normal(2,:).^2);
normal(1,:)=normal(1,:)/sqrt(energy_val);
normal(2,:)=normal(2,:)/sqrt(energy2);
end


function [energy,auto_corr] = feature_extract(signal,sos)

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

%energy_val = energy(1)*abs(auto_corr(1));
%measure2 = energy(2)*abs(auto_corr(2));

end


