function [ zr_crs,energy,mag ] = feature_extract( audio,sos )
%FEATURE_EXTRACT Returns zero crossing count, energy and summed magnitude
%of audio frame
%   INPUT:              audio samples and sos bpf for audio filtering
%   OUTPUT:             zero crossings, energy, summed magnitudes

N = numel(audio);
% hamming window
win_hamm = hamming(N);
% apply window
audio = audio.*win_hamm';
if (sos~=0) % for silence sos is passed as 0
    % speech filter
    audio = sosfilt(sos,audio);
end
% zero crossings
zr_crs = sum(abs(sign(audio(2:end))-sign(audio(1:end-1))))/2;
% short time energy
energy = sum(audio.^2);
% short time magnitude
mag = sum(abs(audio));




end

