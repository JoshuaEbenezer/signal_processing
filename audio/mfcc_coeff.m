function [ mfcc ] = mfcc_coeff( x,Fs )
%MFCC Returns Mel Frequency cepstral coefficients
%   INPUT arguments:    x (input signal),frequency of sampling
%
%   OUTPUT arguments:   mfcc (coefficients)
%
%   Author:             Joshua Ebenezer
%   Date of creation:   27th January, 2018
%

% number of samples corresponding to 0.1s
N = length(x);
% frequency range
freq = (0:N/2)*Fs/N;

mel = 1127*log(1+freq/700);
mel_spacing = max(mel(:))/40;
% linear mel freq bins
mel_bins = 0:mel_spacing:max(mel(:))-1;
% corresponding freq bins
freq_bins = round(700*(exp(mel_bins/1127)-1), -1);
% initialize weighted energies
weighted_energies = ones(1,40);

stft = (fft(x)); % find short-time fourier transform
% double sided spectrum
P2 = abs(stft/N);
% single sided spectrum
STFT = P2(1:N/2+1);

% recover exact amplitudes
STFT(2:end-1) = 2*STFT(2:end-1);

energies = 1e7*(STFT).^2; % energies

% for first bin no overlapping
f1 = 1;
f2 = find(freq==round(freq_bins(2),-1));
w = triang(round(f2-f1)+1);
weighted_energies(1) = sum(energies(f1:f2).'.*w);

for id=2:length(freq_bins)-1 
    f1 = find(freq==round((freq_bins(id)+freq_bins(id-1))/2,-1));
    f2 = find(freq==freq_bins(id));
    w = triang(round(2*(f2-f1)+1)); % triangular windows
    weighted_energies(id) = sum(energies(f1:2*f2-f1).'.*w);
end

% for last bin no overlapping
f1 = f2; % take last value from earlier loop
f2 = find(freq==round(freq_bins(end),-1));
w = triang(round(f2-f1)+1);
weighted_energies(end) = sum(energies(f1:f2).'.*w);

log_weighted = log(weighted_energies);
mfcc = dct(log_weighted);
% discard first coefficient
mfcc = mfcc(2:14);

end

