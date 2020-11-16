function [qImage, clusters] = LloydMax(image, bits, epsilon)
%  Input         : image (Original RGB Image)
%                  bits (bits for quantization)
%                  epsilon (Stop Condition)
%  Output        : qImage (Quantized Image)
%                  clusters (Quantization Table)
%clc;
%clear all;
%close all;
%bits = 2;
%epsilon = 0.001;
%image = double(imread('lena.tif'));

%initial with uniform quantizer
index = 0 :2^bits-1;
rep_value = floor((256/2^bits)*(index+0.5));
%codebook 3 columns: 1 column representativ value, 2 column Sum of all nearest trainingdata, 3 Column Number of nearest trainingdata  
code_book = zeros(2^bits,3);
code_book(:,1) = rep_value;

mse_old = 0;

[w, h, c] = size(image);
%qImage, quantized image with index
qImage = zeros(w,h,c);

while 1
     [distance, index] = pdist2(code_book(:,1),image(:),'euclidean','Smallest',1);
     mse_new = sum(distance.^2)/numel(image(:));
     
     %update the codebook according to euclidean distance 
     for i = 1:numel(image)
        code_book(index(i),2) = code_book(index(i),2)+image(i);
        code_book(index(i),3) = code_book(index(i),3)+1;
        qImage(i) = index(i);
     end
    
     % update the non-zero counter representative value
    nonzero_index = find(code_book(:,3) ~= 0);
    code_book(nonzero_index,1) = round(code_book(nonzero_index,2)./code_book(nonzero_index,3));
    
    %update the zero-count rep_value
    while (nnz(~code_book(:,3))) 
        zero_index = find( code_book(:,3) == 0);
        replace_index = zero_index(1);
        [~,max_index] = max(code_book(:,3));
        code_book(replace_index,1) = code_book(max_index,1)+1;
        %update the counter, not useless, because may exist more than 1 zero counted rep_values
        code_book(replace_index,3) = floor(code_book(max_index,3)/2);
        code_book(max_index,3) = ceil(code_book(max_index,3)/2);
    end
        
    code_book(:,2) = 0;
    code_book(:,3) = 0; 
    
    if (abs(mse_new -mse_old)/mse_new)<epsilon
        break;
    end
    
    mse_old = mse_new;
    mse_new = 0;
end

clusters = code_book;

end