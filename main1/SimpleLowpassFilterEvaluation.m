% Read Image
I = double(imread('satpic1.bmp'));

%kernel declaration and frequency responce 
%lowpass filter
lowpass = [1 2 1; 2 4 2; 1 2 1];
lowpass = lowpass/sum(lowpass(:));
%FIR filter
w = fir1(40,0.5);
W2 = w' * w;
%frequency responce 
kernel = W2 ;
frequencyResp = fft2(kernel);
figure('name','Frequency responce of filter');
imagesc(abs(fftshift(frequencyResp)));


% without prefiltering
% YOUR CODE HERE
I_rec_notpre = sub_up_sample(I);
I_rec_notpre = 4*prefilterlowpass2d(I_rec_notpre, kernel);
% Evaluation without prefiltering
% I_rec_notpre is the reconstructed image WITHOUT prefiltering
PSNR_notpre = calcPSNR(I, I_rec_notpre);
fprintf('Reconstructed image, not prefiltered, PSNR = %.2f dB\n', PSNR_notpre)

% with prefiltering
% YOUR CODE HERE
prefilter_I = prefilterlowpass2d(I, kernel);
down_up_sample = sub_up_sample(prefilter_I);
I_rec_pre =  4*prefilterlowpass2d(down_up_sample, kernel);

% Evaluation with prefiltering
% I_rec_pre is the reconstructed image WITH prefiltering
PSNR_pre = calcPSNR(I, I_rec_pre);
fprintf('Reconstructed image, prefiltered, PSNR = %.2f dB\n', PSNR_pre)

%difference between filtered and unfiltered image
figure;
subplot(211),imshow(uint8(I)), title('Original Picture')
subplot(212), imshow(uint8(prefilter_I)), title('prefiltered Picture')

% put all the sub-functions called in your script here
function pic_pre = prefilterlowpass2d(picture, kernel)
% YOUR CODE HERE
     [W,H,C]=size(picture);
     for index = 1:C
         pic_pre(:,:,index) = conv2(picture(:,:,index),kernel,'same');
     end
end

function MSE = calcMSE(Image, recImage)
% YOUR CODE HERE
  [W,H,C] = size(Image);
  Image = double(Image);
  recImage = double(recImage);
  error_channel = sum((Image(:,:,:)-recImage(:,:,:)).^2,'all');
  MSE= sum(error_channel,'all')/(W*H*C);
end

function PSNR = calcPSNR(Image, recImage)
% YOUR CODE HERE
   MSE = calcMSE(Image,recImage);
   PSNR = 10* log10((2^8-1)^2/MSE);
end

function upsample_image = sub_up_sample(Image)
% YOUR CODE HERE
   [W,H,C]=size(Image);
   for index = 1:C
       subsample_image(:,:,index) = downsample(downsample(Image(:,:,index),2)',2)';
       upsample_image(:,:,index) = upsample(upsample(subsample_image(:,:,index),2)',2)';
   end
end