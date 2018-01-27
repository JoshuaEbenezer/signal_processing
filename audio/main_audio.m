% main_audio.m Script file to run programs for processing of audio file
% ============================ main_audio.m ============================= %
%
% Description                          :
% Subroutines  called                  :
% Called by                            : -
% Author of the code                   : Joshua Peter Ebenezer
% Date of creation                     : January 3rd, 2018
% ------------------------------------------------------------------------%

clc;
clear;
close all;
% Read audio file
[audio,Fs] = audioread('first.mp3');

% Transpose for easier handling
audio = audio.';

% Length of the file (in sampling instances)
L = length(audio);
% time interval of sampling (in seconds)
Ts = 1/Fs;
% sampling time array
t = (0:L-1)*Ts;
% frequency
f = Fs*(0:(L/2))/L;

%% --------------------- Display audio in time domain ------------- %%

figure1 = figure;
axes1 = axes('Parent',figure1,'FontSize',18,'FontName','Times New Roman');
plot(t,audio);

xlabel('time (s)','FontWeight','bold','FontSize',24,...
    'FontName','Times New Roman');
ylabel('Amplitude','FontWeight','bold','FontSize',24,...
    'FontName','Times New Roman');

%% --------------------- Find and display fft of audio ------------- %%
audio_fft = fft(audio);
% double sided spectrum
P2 = abs(audio_fft/L);
% single sided spectrum
audio_FT = P2(1:L/2+1);

% recover exact amplitudes
audio_FT(2:end-1) = 2*audio_FT(2:end-1);


figure2 = figure;
axes2 = axes('Parent',figure2,'FontSize',18,'FontName','Times New Roman');
plot(f/1000,audio_FT)

xlabel('f (kHz)','FontWeight','bold','FontSize',24,...
    'FontName','Times New Roman');
ylabel('|FT(f)|','FontWeight','bold','FontSize',24,...
    'FontName','Times New Roman');

%% ------------------ Extract the words without pauses ----------------- %%

%%--------------------------------- Add AWGN --------------------------- %%
snr1 = 30;
snr2 = 20;
snr3 = 10;

noisy1 = awgn(audio,snr1,'measured');
noisy2 = awgn(audio,snr2,'measured');
noisy3 = awgn(audio,snr3,'measured');

%% -------------------------- Frequency domain separation -------------- %%

% designing elliptical bandpass filter
filter_order = 2;
passband_ripple = 0.1;
stopband_atten = 80;

% telephone speech range is from 300 Hz to 3500 Hz
speech_passband = [300 3.5e3]/(Fs/2);

ftype = 'bandpass';

% Zero-Pole-Gain design to avoid numerical instabilities
[z,p,k] = ellip(filter_order,passband_ripple,stopband_atten,speech_passband,ftype);
% convert to second order sections for implementation
sos = zp2sos(z,p,k);


%%------------- Time domain separation simple thresholding-------------- %%
%
% % threshold of 0.02 determined empirically
% words_fil = audio(abs(audio)>0.005);
% % save to file
% % audiowrite('words_fil.wav',words_fil,Fs);
% % duration (in secs) of words alone
% words_duration = length(words_fil)*Ts;
% % indices where words were detected
% word_indices = abs(audio)>0.005;
%
% % filter signal with 30 dB SNR AWGN
% audio_n30fil = sosfilt(sos, noisy1);
%
% % threshold of 0.005 determined empirically
% wordsn30 = audio_n30fil(abs(audio_n30fil)>0.005);
% % save to file
% % audiowrite('wordsn30_fil.wav',wordsn30,Fs);
% % duration (in secs) of words alone
% wordsn30_duration = length(wordsn30)*Ts;
% % indices where words were detected
% wordn30_indices = abs(wordsn30)>0.005;
%
% % filter signal with 20 dB SNR AWGN
% audio_n20fil = sosfilt(sos, noisy2);
% % threshold of 0.02 determined empirically
% wordsn20 = audio_n20fil(abs(audio_n20fil)>0.02);
% % save to file
% % audiowrite('wordsn20_fil.wav',wordsn20,Fs);
% wordsn20_duration = length(wordsn20)*Ts;
% wordn20_indices = abs(wordsn20)>0.02;
%
% % filter signal with 10 dB SNR AWGN
% audio_n10fil = sosfilt(sos, noisy3);
% % threshold of 0.02 determined empirically
% wordsn10 = audio_n10fil(abs(audio_n10fil)>0.04);
% % save to file
% audiowrite('wordsn10_fil.wav',wordsn10,Fs);
% wordsn10_duration = length(wordsn10)*Ts;
% wordn10_indices = abs(wordsn10)>0.04;

%% --------------------- frame by frame processing --------------------- %%

% 1 frame = 0.1 s
% find number of samples corresponding to 0.1 seconds
samples_per_frame = Fs*0.1;
% number of frames
num_frames = floor(length(audio)/samples_per_frame);
% initialize frames
frames = zeros(num_frames,samples_per_frame);
% initialize zero crossing vector
zr_crs_ind = zeros(1,num_frames);
% initialize zero crossings count
zr_crs = zeros(1,num_frames);
% energy of the signal
energy = zeros(1,num_frames);
% summed magnitude of the signal
mag = zeros(1,num_frames);

feature = zeros(1,num_frames);


% % hamming window
% win_hamm = hamming(numel(audio));
% % apply window
% audio = audio.*win_hamm';
mfcc = zeros(num_frames,13);
thresh = 1;
for i=1:num_frames
    % separate frames
    frames(i,:) = audio((i-1)*samples_per_frame+1:i*samples_per_frame);
    % find zero crossings and energy
    [zr_crs_ind(i),energy(i),mag(i)]=feature_extract(frames(i,:),sos,Fs);
    feature(i) = (zr_crs_ind(i)/energy(i))/samples_per_frame;
    % insert condition for speech/silence here
    
end
for i=1:num_frames
    if (feature(i)<thresh)
            mfcc(i,:) = mfcc_coeff(frames(i,:),Fs);
    end
end

zr_crs_ind = zr_crs_ind/max(zr_crs_ind);
energy = energy/max(energy);
mag = mag/max(mag);

figure;
plot((1:num_frames)*0.1,zr_crs_ind);
xlabel('time (s)');
ylabel('Number of zero crossings');

hold on;
plot((1:num_frames)*0.1,energy,'r');
plot(t,audio,'k');

hold on;
plot((1:num_frames)*0.1,feature/max(feature(:)),'m');
% xlabel('time (s)');
% ylabel('Summed magnitude of frame');




