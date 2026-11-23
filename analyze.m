close all
% 1. Load WAV file (convert to mono if stereo)
%files = ["cool7.wav","cool7theorybackdoor.wav", ...
 %   "cool7capture0.wav","cool7capture1.wav", "cool7capture2.wav", "cool7capture3.wav", "cool7capture4.wav"];
files = ["song.wav"];
%files = ["speech.wav","speechcapture0.wav", "speechcapture1.wav"];
fc = 40000;
fs = 48000;
demodulate = [0,0,0,0,0,0,0,0];
for i=1:numel(files)
    [y, Fs] = audioread(files(i));
    if size(y, 2) > 1
        y = mean(y, 2);
    end
    
    if (demodulate(i))
        t = (0:(numel(y)-1))'/fs;
        y = y .* cos(2*pi*fc*t);
    end
    
    
    % 2. Compute FFT and frequency axis
    L = length(y);
    Y = fft(y);
    P2 = abs(Y / L);
    P1 = P2(1:floor(L/2) + 1);
    P1(2:end-1) = 2 * P1(2:end-1);
    
    f = Fs * (0:floor(L/2)) / L;
    
    % 3. Plot single-sided amplitude spectrum
    figure;
    plot(f, P1);
    title(append('Amplitude Spectrum for ',files(i), "demodulate=",string(demodulate(i))));
    xlabel('Frequency (Hz)');
    ylabel('|P1(f)|');
    
end
grid on;
