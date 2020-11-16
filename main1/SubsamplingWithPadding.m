%% 
clc
clear all
close all
%% image read
I_lena = double(imread('lena.tif'));
%% wrap round
% YOUR CODE HERE
mirror_pixel = 4;
I_lena_wrapround = Wrapround(I_lena,mirror_pixel);
%% Resample(subsample)
% YOUR CODE HERE
Sample_rate = 1/2;
resampled_image= mySample(I_lena_wrapround,Sample_rate);
%% Crop Back
I_lena_cropback = cropback(resampled_image,mirror_pixel/2);
% YOUR CODE HERE
%% Wrap Round
% YOUR CODE HERE
I_lena_subsample_wrapround = Wrapround(I_lena_cropback,mirror_pixel/2);
%% 
Sample_rate = 2;
resampled_image= mySample(I_lena_subsample_wrapround,Sample_rate);
%% Crop back
% YOUR CODE HERE
I_rec_lena = cropback(resampled_image,mirror_pixel);

%% Distortion Analysis
PSNR_lena        = calcPSNR(I_lena, I_rec_lena);
fprintf('PSNR lena subsampling = %.3f dB\n', PSNR_lena)

%% 

% put all the sub-functions called in your script here
function MSE = calcMSE(Image, recImage)
% Input         : Image    (Original Image)
%                 recImage (Reconstructed Image)
% Output        : MSE      (Mean Squared Error)
  [W,H,C] = size(Image);
  Image = double(Image);
  recImage = double(recImage);
  error_channel = sum((Image(:,:,:)-recImage(:,:,:)).^2,'all');
  MSE= sum(error_channel,'all')/(W*H*C);
end


function PSNR = calcPSNR(Image, recImage)
% Input         : Image    (Original Image)
%                 recImage (Reconstructed Image)
%
% Output        : PSNR     (Peak Signal to Noise Ratio)
% YOUR CODE HERE
% call calcMSE to calculate MSE
   MSE = calcMSE(Image,recImage);
   PSNR = 10* log10((2^8-1)^2/MSE);
end

function image_wrapround = Wrapround(original_image,mirPixel)
    image_wrapround = padarray(original_image,[mirPixel mirPixel],'symmetric','both');
    %[W,H,m] = size(I_lena_wrapround )
end

function  resampled_image= mySample(wrapround_image,sample_rate)
     [~,~,C] = size(wrapround_image)
     if sample_rate < 1
        for index = 1:C
            resampled_image(:,:,index) = resample(resample(wrapround_image(:,:,index),1,1/sample_rate,3)',1,1/sample_rate,3)';
        end
     else 
        for index = 1:C
            resampled_image(:,:,index) = resample(resample(wrapround_image(:,:,index),sample_rate,1,3)',sample_rate,1,3)';
        end
     end
end

function cropback_image = cropback(resampled_image,boundary_pixel)
    [~,~,C] = size( resampled_image);
    cropback_image(:,:,:) = resampled_image(1+boundary_pixel:end-boundary_pixel,1+boundary_pixel:end-boundary_pixel,:);
end

