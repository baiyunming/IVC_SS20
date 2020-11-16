%% 
clc
clear all
close all
%% 
% read original RGB image 
I_ori = double(imread('lena.tif'));
%% 
% YOUR CODE HERE for chroma subsampling 
YCbCr =  ictRGB2YCbCr(I_ori);
Cb_rec  =  Subsample_ChromPlane(YCbCr(:,:,2));
Cr_rec  =  Subsample_ChromPlane(YCbCr(:,:,3));
%% 
YCbCr_rec(:,:,1) = YCbCr(:,:,1);
YCbCr_rec(:,:,2) = Cb_rec;
YCbCr_rec(:,:,3) = Cr_rec;
%% 
rgb_rec = ictYCbCr2RGB(YCbCr_rec);
%% 
% Evaluation
% I_rec is the reconstructed image in RGB color space
PSNR = calcPSNR(I_ori, rgb_rec)
fprintf('PSNR is %.2f dB\n', PSNR);

%%
% put all the sub-functions called in your script here
function rgb = ictYCbCr2RGB(yuv)
    rgb(:,:,1) = yuv(:,:,1) + 1.402*yuv(:,:,3);
    rgb(:,:,2) = yuv(:,:,1) - 0.344*yuv(:,:,2) -0.714*yuv(:,:,3);
    rgb(:,:,3) = yuv(:,:,1) + 1.772*yuv(:,:,2);
end

function yuv = ictRGB2YCbCr(rgb)
    yuv (:,:,1) = 0.299*rgb(:,:,1) + 0.587*rgb(:,:,2) + 0.114*rgb(:,:,3); 
    yuv (:,:,2) = -0.169*rgb(:,:,1) - 0.331*rgb(:,:,2) + 0.5*rgb(:,:,3); 
    yuv (:,:,3) = 0.5*rgb(:,:,1) - 0.419*rgb(:,:,2) -0.081*rgb(:,:,3); 
end

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

function CbCr_rec  =  Subsample_ChromPlane(CbCr)
    mirror_pixel = 4;
    CbCr_wrapround = Wrapround(CbCr,mirror_pixel);
    downSample_rate = 1/2;
    downsampled_CbCr= mySample(CbCr_wrapround,downSample_rate);
    CbCr_cropback = cropback(downsampled_CbCr,mirror_pixel/2);
    CbCr_downsample_wrapround = Wrapround(CbCr_cropback,mirror_pixel/2);
    upSample_rate = 2;
    resampled_CbCr= mySample(CbCr_downsample_wrapround,upSample_rate);
    CbCr_rec = round(cropback(resampled_CbCr,mirror_pixel));
end
