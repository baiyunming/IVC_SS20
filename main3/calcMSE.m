function MSE = calcMSE( image1, image2 )
  [W,H,C] = size(image1);
  error_channel = sum((image1(:,:,:)-image2(:,:,:)).^2,'all');
  MSE= sum(error_channel,'all')/(W*H*C);
end