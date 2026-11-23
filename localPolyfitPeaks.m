% Local polynomial regression peak detection
%
% x         = sample locations
% y         = signal
% k         = number of neighboring points
% degree    = polynomial degree (1=linear, 2=quadratic, 3=cubic)
% threshold = residual threshold (robust standard deviations)

function [maxima, residuals] = ...
    localPolyfitPeaks(x, y, k, degree, threshold)


if nargin < 4 || isempty(degree)
    degree = 2;
end

if nargin < 5 || isempty(threshold)
    threshold = 3;
end

x = x(:);
y = y(:);

N = length(y);
h = floor(k/2);

residuals = zeros(N,1);

for i = h+1:N-h

    % Local window excluding center point
    idx = [i-h:i-1, i+1:i+h];

    % Normalize x for numerical stability
    xlocal = x(idx) - x(i);

    % Polynomial least-squares fit
    p = polyfit(xlocal, y(idx), degree);

    % Predict center point (xlocal = 0)
    prediction = polyval(p, 0);

    % Residual at center
    residuals(i) = y(i) - prediction;

end

% Identify positive and negative outliers
maxima = find(residuals > threshold*max(residuals));

end