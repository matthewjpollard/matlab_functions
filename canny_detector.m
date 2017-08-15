function [I_edges]= canny_detector(I, sigma, g_size, thresholds)
% This function uses Canny method to find the edges of an image
% INPUTS:
%          I: Input image, NxM array grayscale
%      alpha: standard deviation for gauss filter
%     g_size: size of the convolution kernel, int or 1x2 array
% thresholds: normalized low/high threshold values for use in double threshold, 1x2 array
%
% OUTPUT: 
%  I_edges: binary image showing only the edges of the input image
%
% Matt Pollard, August 2017

%% Input handling and setup
% Input image, convert to double
I = double(I);

% Grab the low and high thresholds from inputs
t_low  = min(thresholds);
t_high = max(thresholds);

% Define the necessary filters
gauss_kernel = fspecial('Gaussian', g_size, sigma);
deriv_kernel = fspecial('sobel');
grad_y_kernel = deriv_kernel;
grad_x_kernel = flipud(deriv_kernel).';

%% Gaussian Blur, then use Sobel operator
I_blur = conv2(I, gauss_kernel, 'same');

% Get the X and Y gradients via the sobel kernels
mag_X = conv2(I_blur, grad_x_kernel, 'same');
mag_Y = conv2(I_blur, grad_y_kernel, 'same');
%Calculate magnitude
I_grad_mag = abs(mag_X) + abs(mag_Y);

% Calculate directions/orientations
I_angle = atan2d(mag_Y, mag_X); % result in degrees

% Set the size for any loops and preallocation based on the convolution
xs = size(I_blur,1);
ys = size(I_blur,2);

% Fix the angles to 0, 45, 90, or 135 degree
for ii = 1:xs
    for jj = 1:ys
        if (I_angle(ii,jj) < 0)
            % Force the angle to be positive
            I_angle(ii,jj) = 360 + I_angle(ii,jj);
        end
        if ((I_angle(ii, jj) >= 0 ) && (I_angle(ii, jj) < 22.5) || (I_angle(ii, jj) >= 157.5) && (I_angle(ii, jj) < 202.5) || (I_angle(ii, jj) >= 337.5) && (I_angle(ii, jj) <= 360))
            I_angle(ii, jj) = 0;
        elseif ((I_angle(ii, jj) >= 22.5) && (I_angle(ii, jj) < 67.5) || (I_angle(ii, jj) >= 202.5) && (I_angle(ii, jj) < 247.5))
            I_angle(ii, jj) = 45;
        elseif ((I_angle(ii, jj) >= 67.5 && I_angle(ii, jj) < 112.5) || (I_angle(ii, jj) >= 247.5 && I_angle(ii, jj) < 292.5))
            I_angle(ii, jj) = 90;
        elseif ((I_angle(ii, jj) >= 112.5 && I_angle(ii, jj) < 157.5) || (I_angle(ii, jj) >= 292.5 && I_angle(ii, jj) < 337.5))
            I_angle(ii, jj) = 135;
        end
    end
end

%% Non-Maximum Supression
I_suppressed = zeros (xs, ys);
for ii = 2:(xs-1)
    for jj = 2:(ys-1)
        if (I_angle(ii,jj) == 0)
            I_suppressed(ii,jj) = (I_grad_mag(ii,jj) == max([I_grad_mag(ii,jj), I_grad_mag(ii,jj+1), I_grad_mag(ii,jj-1)]));
        elseif (I_angle(ii,jj) == 45)
            I_suppressed(ii,jj) = (I_grad_mag(ii,jj) == max([I_grad_mag(ii,jj), I_grad_mag(ii+1,jj-1), I_grad_mag(ii-1,jj+1)]));
        elseif (I_angle(ii,jj) == 90)
            I_suppressed(ii,jj) = (I_grad_mag(ii,jj) == max([I_grad_mag(ii,jj), I_grad_mag(ii+1,jj), I_grad_mag(ii-1,jj)]));
        elseif (I_angle(ii,jj) == 135)
            I_suppressed(ii,jj) = (I_grad_mag(ii,jj) == max([I_grad_mag(ii,jj), I_grad_mag(ii+1,jj+1), I_grad_mag(ii-1,jj-1)]));
        end
    end
end
I_suppressed = I_suppressed.*I_grad_mag;

%% Hysteresis Thresholding
t_low = t_low * max(max(I_suppressed));
t_high = t_high * max(max(I_suppressed));

I_edges = zeros(xs, ys);
for ii = 1:xs
    for jj = 1:ys
        if (I_suppressed(ii, jj) < t_low)
            % Edge automatically disqualified
            I_edges(ii, jj) = 0;
        elseif (I_suppressed(ii, jj) > t_high)
            % This is a sure edge
            I_edges(ii, jj) = 1;
        % Check all pixels surrounding the current pixel
        elseif ( I_suppressed(ii+1,jj) > t_high || I_suppressed(ii-1,jj) > t_high || I_suppressed(ii,jj+1) > t_high ...
              || I_suppressed(ii,jj-1) > t_high || I_suppressed(ii-1, jj-1) > t_high || I_suppressed(ii-1, jj+1) > t_high ...
              || I_suppressed(ii+1, jj+1) > t_high || I_suppressed(ii+1, jj-1) > t_high)
            % Qualified because it touches a sure edge and was a candidate
            I_edges(ii,jj) = 1;
        end
    end
end
%% Detected Edge Output
% go back to [0,255]
I_edges = uint8(I_edges.*255);
figure, 
subplot(1,2,1)
imagesc(I);
title('Original Image');

subplot(1,2,2);
imagesc(I_edges)
title('Detected Edges');
colormap('gray');
