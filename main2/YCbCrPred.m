%% 
clc
close all
clear all
%% 
% Read Image
imageLena      = double(imread('lena.tif'));
imageLena_small = double(imread('lena_small.tif'));
%% 
% ICT RGB to YCbCr
imageLena_unsample_YCbCr = ictRGB2YCbCr(imageLena);
imageLena_small_YCbCr = ictRGB2YCbCr(imageLena_small);
%% 
%compress chromonance component get imageLena_YCbC
Y_Lena = imageLena_unsample_YCbCr(:,:,1);
Cb_Lena_subsample  =  resample(resample(imageLena_unsample_YCbCr(:,:,2),1,2,3)',1,2,3)'; 
Cr_rec_subsample  =  resample(resample(imageLena_unsample_YCbCr(:,:,3),1,2,3)',1,2,3)';
%%  create the predictor and obtain the residual image
resLena_Y = getres_Y(Y_Lena);
resLnea_Cb = getres_CbCr(Cb_Lena_subsample);
resLena_Cr = getres_CbCr(Cr_rec_subsample);


%% 
% codebook construction using res_lena_small
% get the PMF of the residual image
resImageLenaSmall = predict(imageLena_small_YCbCr);
rangeError = -255:255;
pmfResLenaSmall  =  stats_marg(resImageLenaSmall, rangeError);
[BinaryTree, HuffCode, BinCode, Codelengths] = buildHuffman( pmfResLenaSmall );
%% 

% Encoding
bytestreamY = enc_huffman_new(round(resLena_Y(:))+255+1, BinCode, Codelengths);
bytestreamCb = enc_huffman_new(round(resLnea_Cb(:))+255+1, BinCode, Codelengths);
bytestreamCr = enc_huffman_new(round(resLena_Cr(:))+255+1, BinCode, Codelengths);
%bytestream = bytestreamY + bytestreamCb + bytestreamCr;
%% 

% Decoding
Reconstructed_resY = double(reshape( dec_huffman_new ( bytestreamY, BinaryTree, max(size(resLena_Y(:))) ), size(resLena_Y)))-255-1 ;
Reconstructed_resCb = double(reshape( dec_huffman_new ( bytestreamCb, BinaryTree, max(size(resLnea_Cb(:))) ), size(resLnea_Cb)))-255-1 ;
Reconstructed_resCr = double(reshape( dec_huffman_new ( bytestreamCr, BinaryTree, max(size(resLena_Cr(:))) ), size(resLena_Cr)))-255-1 ;
%% 

recon_lena_Y = getOriginal_Y(Reconstructed_resY);
recon_lena_Cb =  getOriginal_CbCr(Reconstructed_resCb);
recon_lena_Cr = getOriginal_CbCr(Reconstructed_resCr);

%% 

% upsample Cb Cr
recon_YCbCr(:,:,1) = recon_lena_Y;
recon_YCbCr(:,:,2) = resample(resample(recon_lena_Cb,2,1,3)',2,1,3)';
recon_YCbCr(:,:,3) = resample(resample(recon_lena_Cr,2,1,3)',2,1,3)'; 

rec_image = ictYCbCr2RGB(recon_YCbCr);

%%
% evaluation and show results
figure
subplot(121)
imshow(uint8(imageLena)), title('Original Image')
subplot(122)

PSNR = calcPSNR(imageLena, rec_image);
imshow(uint8(rec_image)), title(sprintf('Reconstructed Image, PSNR = %.2f dB', PSNR))
BPP = (numel(bytestreamY)+ numel(bytestreamCb)+ numel(bytestreamCr)) * 8 / (numel(imageLena)/3);
CompressionRatio = 24/BPP;

fprintf('Bit Rate         = %.2f bit/pixel\n', BPP);
fprintf('CompressionRatio = %.2f\n', CompressionRatio);
fprintf('PSNR             = %.2f dB\n', PSNR);

% Put all sub-functions which are called in your script here.





%% 
function Reconstructed_Data = getOriginal_Y(Reconstructed_Data)
[W,H] = size(Reconstructed_Data);

for i = 2: W
    for j = 2: H
        prediction = (7/8)*Reconstructed_Data(i,j-1) -(1/2)*Reconstructed_Data(i-1,j-1) + (5/8)*Reconstructed_Data(i-1,j);
        Reconstructed_Data(i,j) = Reconstructed_Data(i,j)+ prediction;
    end
end

end


function Reconstructed_Data = getOriginal_CbCr(Reconstructed_Data)
[W,H] = size(Reconstructed_Data);

for i = 2: W
    for j = 2: H
         prediction = (3/8)*Reconstructed_Data(i,j-1) -(1/4)*Reconstructed_Data(i-1,j-1) + (7/8)*Reconstructed_Data(i-1,j);
         Reconstructed_Data(i,j) = Reconstructed_Data(i,j)+prediction;
    end
end

end

%% 
% Put all sub-functions which are called in your script here.
function yuv = ictRGB2YCbCr(rgb)
% Input         : rgb (Original RGB Image)
% Output        : yuv (YCbCr image after transformation)
% YOUR CODE HERE
    yuv (:,:,1) = 0.299*rgb(:,:,1) + 0.587*rgb(:,:,2) + 0.114*rgb(:,:,3); 
    yuv (:,:,2) = -0.169*rgb(:,:,1) - 0.331*rgb(:,:,2) + 0.5*rgb(:,:,3); 
    yuv (:,:,3) = 0.5*rgb(:,:,1) - 0.419*rgb(:,:,2) -0.081*rgb(:,:,3); 
end

function rgb = ictYCbCr2RGB(yuv)
    rgb(:,:,1) = yuv(:,:,1) + 1.402*yuv(:,:,3);
    rgb(:,:,2) = yuv(:,:,1) - 0.344*yuv(:,:,2) -0.714*yuv(:,:,3);
    rgb(:,:,3) = yuv(:,:,1) + 1.772*yuv(:,:,2);
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


function pmf = stats_marg(image, range)
    [counts,~] = hist(image(:),range);
    pmf = counts/sum(counts);
end


function resImage = getres_Y(YCbCr)
[w ,h] = size(YCbCr);
resImage = zeros(w,h);
reconImage = zeros(w,h);
resImage(1,:) = YCbCr(1,:);
resImage(:,1) = YCbCr(:,1);
reconImage(1,:) = YCbCr(1,:);
reconImage(:,1) = YCbCr(:,1);

for i = 2: w
    for j = 2: h
        reconImage(i,j) = (7/8)*reconImage(i,j-1) -(1/2)*reconImage(i-1,j-1) + (5/8)*reconImage(i-1,j);
        resImage(i,j,1) = round(YCbCr(i,j)-reconImage(i,j));
        reconImage(i,j) = reconImage(i,j)+resImage(i,j);
    end
end

end

function resImage = getres_CbCr(YCbCr)
[w ,h] = size(YCbCr);
resImage = zeros(w,h);
reconImage = zeros(w,h);
resImage(1,:) = YCbCr(1,:);
resImage(:,1) = YCbCr(:,1);
reconImage(1,:) = YCbCr(1,:);
reconImage(:,1) = YCbCr(:,1);
 
for i = 2: w
    for j = 2: h
        reconImage(i,j) = (3/8)*reconImage(i,j-1) -(1/4)*reconImage(i-1,j-1) + (7/8)*reconImage(i-1,j);
        resImage(i,j) = round(YCbCr(i,j)-reconImage(i,j));
        reconImage(i,j) = reconImage(i,j)+resImage(i,j);
    end
end

end



function resImage = predict(YCbCr)
[w ,h, c] = size(YCbCr);
resImage = zeros(w,h,c);
reconImage = zeros(w,h,c);
resImage(1,:,:) = YCbCr(1,:,:);
resImage(:,1,:) = YCbCr(:,1,:);
reconImage(1,:,:) = YCbCr(1,:,:);
reconImage(:,1,:) = YCbCr(:,1,:);

for i = 2: w
    for j = 2: h
        reconImage(i,j,1) = (7/8)*reconImage(i,j-1,1) -(1/2)*reconImage(i-1,j-1,1) + (5/8)*reconImage(i-1,j,1);
        resImage(i,j,1) = round(YCbCr(i,j,1)-reconImage(i,j,1));
        reconImage(i,j,1) = reconImage(i,j,1)+resImage(i,j,1);
    end
end
%% 
for i = 2: w
    for j = 2: h
        reconImage(i,j,2) = (3/8)*reconImage(i,j-1,2) -(1/4)*reconImage(i-1,j-1,2) + (7/8)*reconImage(i-1,j,2);
        resImage(i,j,2) = round(YCbCr(i,j,2)-reconImage(i,j,2));
        reconImage(i,j,2) = reconImage(i,j,2)+resImage(i,j,2);
    end
end
%% 
for i = 2: w
    for j = 2: h
        reconImage(i,j,3) = (3/8)*reconImage(i,j-1,3) -(1/4)*reconImage(i-1,j-1,3) + (7/8)*reconImage(i-1,j,3);
        resImage(i,j,3) = round(YCbCr(i,j,3)-reconImage(i,j,3));
        reconImage(i,j,3) = reconImage(i,j,3)+resImage(i,j,3);
    end
end

end


