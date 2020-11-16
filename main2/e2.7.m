% Read Image
imageLena      = double(imread('lena.tif'));
imageLena_small = double(imread('lena_small.tif'));

%ICT RGB to YCbCr
imageLena_unsample_YCbCr = ictRGB2YCbCr(imageLena);
imageLena_small_YCbCr = ictRGB2YCbCr(imageLena_small);

[w,h,c] = size(imageLena_unsample_YCbCr);
Cb_rec  =  Subsample_ChromPlane(imageLena_unsample_YCbCr(:,:,2));
Cr_rec  =  Subsample_ChromPlane(imageLena_unsample_YCbCr(:,:,3));
imageLena_YCbCr =zeros(w,h,c);
imageLena_YCbCr(:,:,1) = imageLena_unsample_YCbCr(:,:,1);
imageLena_YCbCr(:,:,2) = Cb_rec;
imageLena_YCbCr(:,:,3) = Cr_rec;


% create the predictor and obtain the residual image
resImageLena = predict(imageLena_YCbCr);
resImageLenaSmall = predict(imageLena_small_YCbCr);

% codebook construction using res_lena_small
% get the PMF of the residual image
rangeError = -255:255;
pmfResLenaSmall  =  stats_marg(resImageLenaSmall, rangeError);
[BinaryTree, HuffCode, BinCode, Codelengths] = buildHuffman( pmfResLenaSmall );

% Encoding
bytestream = enc_huffman_new(round(resImageLena(:))+255+1, BinCode, Codelengths);

% Decoding
Reconstructed_Data = double(reshape( dec_huffman_new ( bytestream, BinaryTree, max(size(resImageLena(:))) ), size(resImageLena)))-255-1 ;
Reconstructed_Data = getOriginal(Reconstructed_Data);
 
rec_image = ictYCbCr2RGB(Reconstructed_Data);

%% evaluation and show results
figure
subplot(121)
imshow(uint8(imageLena)), title('Original Image')
subplot(122)

PSNR = calcPSNR(imageLena, rec_image);
imshow(uint8(rec_image)), title(sprintf('Reconstructed Image, PSNR = %.2f dB', PSNR))
BPP = numel(bytestream) * 8 / (numel(imageLena)/3);
CompressionRatio = 24/BPP;

fprintf('Bit Rate         = %.2f bit/pixel\n', BPP);
fprintf('CompressionRatio = %.2f\n', CompressionRatio);
fprintf('PSNR             = %.2f dB\n', PSNR);

% Put all sub-functions which are called in your script here.
function Reconstructed_Data = getOriginal(Reconstructed_Data)
[W,H,C] = size(Reconstructed_Data);

for i = 2: W
    for j = 2: H
        prediction = (7/8)*Reconstructed_Data(i,j-1,1) -(1/2)*Reconstructed_Data(i-1,j-1,1) + (5/8)*Reconstructed_Data(i-1,j,1);
        Reconstructed_Data(i,j,1) = Reconstructed_Data(i,j,1)+ prediction;
    end
end

for i = 2: W
    for j = 2: H
        prediction = (3/8)*Reconstructed_Data(i,j-1,2) -(1/4)*Reconstructed_Data(i-1,j-1,2) + (7/8)*Reconstructed_Data(i-1,j,2);
        Reconstructed_Data(i,j,2) = Reconstructed_Data(i,j,2)+prediction;
    end
end

for i = 2: W
    for j = 2: H
         prediction = (3/8)*Reconstructed_Data(i,j-1,3) -(1/4)*Reconstructed_Data(i-1,j-1,3) + (7/8)*Reconstructed_Data(i-1,j,3);
         Reconstructed_Data(i,j,3) = Reconstructed_Data(i,j,3)+prediction;
    end
end

end

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

function image_wrapround = Wrapround(original_image,mirPixel)
    image_wrapround = padarray(original_image,[mirPixel mirPixel],'symmetric','both');
    %[W,H,m] = size(I_lena_wrapround )
end

function  resampled_image= mySample(wrapround_image,sample_rate)
     [~,~,C] = size(wrapround_image);
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

% CbCr channel processing
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

function pmf = stats_marg(image, range)
    [counts,~] = hist(image(:),range);
    pmf = counts/sum(counts);
end

function H = calc_entropy(pmf)
    H = -sum(pmf.*log2(pmf),'omitnan');
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

