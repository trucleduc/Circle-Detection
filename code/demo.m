clear all; close all; clc;

addpath('./core');
fileName = 'test.png';

% parameters
angle = 180;
Tr = 0.6;

% read image
I = imread(fileName);

% detect circles
[circles, ~] = detectCircles(I, angle, Tr);

% display
J = I;
[y, x] = ndgrid(1 : size(I, 1), 1 : size(I, 2));
x = x(:); y = y(:);
for i = 1 : size(circles, 1)
    idx = abs(sqrt((x - circles(i, 1)) .^ 2 + (y - circles(i, 2)) .^ 2) - circles(i, 3)) <= 2;
    J(sub2ind(size(J), y(idx), x(idx), ones(sum(idx), 1))) = 0;
    J(sub2ind(size(J), y(idx), x(idx), 2 * ones(sum(idx), 1))) = 255;
    J(sub2ind(size(J), y(idx), x(idx), 3 * ones(sum(idx), 1))) = 0;
end
imshow(J);
