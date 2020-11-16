function PSNR = calcPSNR( image1, image2 )
   MSE = calcMSE(image1,image2);
   PSNR = 10* log10((2^8-1)^2/MSE);
end
