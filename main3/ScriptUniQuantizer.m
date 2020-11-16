% general script, e.g. image loading, function calls, etc.
clc;
clear all;
close all;

imageLena_small = double(imread('lena_small.tif'));
imageLena = double(imread('lena.tif'));
bits_small      = [1 2 3 5 7];
bits = [3 5];
PSNR_small = [];

for bit = bits_small
    qImageLena_small = UniQuant(imageLena_small, bit);
    recImage_small   = InvUniQuant(qImageLena_small, bit);
    PSNR_small = [PSNR_small calcPSNR(imageLena_small, recImage_small)];
end

PSNR = [];
for bit = bits
    qImageLena = UniQuant(imageLena, bit);
    recImage   = InvUniQuant(qImageLena, bit);
    PSNR = [PSNR calcPSNR(imageLena, recImage)];
end
figure;
plot(3*bits_small, PSNR_small);
hold on; 
plot(3*bits,PSNR);
xlabel('Rate (bits/pixel)');
ylabel('PSNR (dB)');

% define your functions, e.g. calcPSNR, UniQuant, InvUniQuant
function qImage = UniQuant(image, bits)
    qImage = floor(image/256*(2^bits));
end

function image = InvUniQuant(qImage, bits)
    image = floor(256*(qImage+0.5)/2^bits);
end




