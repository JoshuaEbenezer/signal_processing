%% Script file to run Pan-Tompskins algorithm for QRS detection

clc;
clear;
close all;

%% Load data from MIT-BIH database

[ann,anntype,subtype,chan,num,comments]=rdann('/database/100/', 'atr');
wfdb2mat('/database/100/');
[t,signal,Fs,siginfo]=rdmat('100m');
Ts = 1/Fs; % sampling time interval
ecg1 = signal(:,1);
%% Digital Bandpass (5-15 Hz)

passband = [5 15]/(Fs/2); % passband 5-15 Hz

% Transfer Function design for BPF
% Zero-Pole-Gain design
[z,p,k] = butter(6,passband,'bandpass');
sos = zp2sos(z,p,k);

bpf_filtered = sosfilt(sos,ecg1);

%% Differentiation

diff_op = (-bpf_filtered(1:end-4)-2*bpf_filtered(2:end-3)+2*bpf_filtered(4:end-1)+bpf_filtered(5:end))/(8*Ts);

%% Squaring

square_op = diff_op.^2;

%% Moving window integration

N = round(150*1e-3 * Fs); % number of samples corresponding to 150ms

movwin = zeros(length(t)-N-3,1); % initialize moving window integration op

for i = 0:length(t)-N-4
    movwin(i+1) = sum(square_op(1+i:N+i))/N;
end

%% Training period - first 2 seconds

training_windows = round(2/0.15);    % number of windows corresponding to training period
SPK1 = max(movwin(1:training_windows,:))/2; % first estimate of signal peak - half of maximum
NPK1 = mean(movwin(1:training_windows,:)); % first estimate of noise peak - mean

%% Find local maxima
% find peaks in the signal that are separated by atleast 200 ms
% (physiologically impossible for peaks to occur within 200 ms of another
% peak)
[peak,peak_locations] = findpeaks(movwin,'MinPeakDistance',round(0.2*Fs));

RR1 = 1/8 * sum((peak_locations(2:9)-peak_locations(1:8))); % first estimate
RR2 = RR1; % first estimate

RR_low = 0.92 * RR2;
RR_high = 1.16*RR2;
RR_missed = 1.66*RR2;

qrs_peak = zeros(1,length(peak));
qrs_location= zeros(1,length(peak));

qrs_count = 1;

for index = 1:length(peak_locations)
    
    thresh1 = NPK1 + 0.25*(SPK1-NPK1); % threshold 1
    thresh2 = 0.5*thresh1; % threshold 2
    
    if (peak(index)>thresh1) % condition for QRS
        qrs_peak(qrs_count)=peak(index);
        qrs_location(qrs_count) = peak_locations(index);
        
        qrs_count=qrs_count+1; % increment QRS count
        
        if (qrs_count>7)
            RR1 = 1/8 * sum(qrs_location(qrs_count-7:qrs_count)); % finding RR interval
        else
            RR1=1/8*(sum(qrs_location(1:qrs_count))+sum(peak_locations(1:8-qrs_count))); % estimate RR if there are not enough intervals yet
        end
        
        SPK1 = 0.125*peak(index)+0.875*SPK1; % update signal threshold
    elseif (peak_locations(index)-qrs_location(qrs_count)>RR_missed)
        if (peak(index)>thresh2)
            qrs_peak(qrs_count)=peak(index);
            qrs_location(qrs_count) = peak_locations(index);
            SPK1=0.25*peak(index)+0.75*SPK1; % update signal threshold
            qrs_count=qrs_count+1;
            
            if (qrs_count>7)
                RR1 = 1/8 * sum(qrs_location(qrs_count-7:qrs_count)); % finding RR interval
            else
                RR1=1/8*(sum(qrs_location(1:qrs_count))+sum(peak_locations(1:8-qrs_count))); % estimate RR if there are not enough intervals yet
            end
        end
    else
        NPK1 = 0.125*peak(index)+0.875*NPK1; % update noise threshold
    end
end

