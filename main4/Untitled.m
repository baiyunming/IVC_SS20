lena_small = double(imread('lena_small.tif'));
Lena       = double(imread('lena.tif'));

scales =[0.2, 0.4, 0.8, 1.0, 1.5, 2, 3, 4, 4.5]; 
for scaleIdx = 1 : numel(scales)
    qScale   = scales(scaleIdx);
    first_frame =  double(imread('foreman0020.bmp'));
    first_frame_intraencode =  IntraEncode(first_frame, qScale);
    %% use pmf of first_frame to build and train huffman table
    %lower bound =-1000, upper_bound = 4000
    a = -1000;  
    b = 4000;
    H = hist(first_frame_intraencode(:),a:b);
    H = H/sum(H);
    [ BinaryTree, HuffCode, BinCode, Codelengths] = buildHuffman( H );
    
    %% use trained table to encode current_frame to get the bytestream
    % your code here
    for i =1 :21
    current_frame = double(imread(['foreman',num2str(i+20-1,'%04d'),'.bmp']));
    current_frame_introencode        = IntraEncode(current_frame, qScale);

    bytestream = enc_huffman_new( current_frame_introencode-a+1 , BinCode, Codelengths);
    bitPerPixel(i) = (numel(bytestream)*8) / (numel(Lena)/3);
    %% image reconstruction
    
    k_rec = double(dec_huffman_new ( bytestream, BinaryTree,  size( current_frame_introencode(:),1 ) ))+a-1;
    I_rec = IntraDecode(k_rec, size(Lena),qScale);
    PSNR(i) = calcPSNR(current_frame, I_rec);
    end
     
mean_PSNR = mean(PSNR);
means_rate = mean(bitPerPixel);

distortion(scaleIdx) = mean_PSNR;
Rate(scaleIdx) = means_rate;
 end

  
function dst = IntraEncode(image, qScale)
%  Function Name : IntraEncode.m
%  Input         : image (Original RGB Image)
%                  qScale(quantization scale)
%  Output        : dst   (sequences after zero-run encoding, 1xN)
        YCbCr_image = ictRGB2YCbCr(image);
        dct_coeff = blockproc(YCbCr_image,[8 8],@(block_struct) DCT8x8 (block_struct.data));
        quantize_dct = blockproc(dct_coeff,[8 8],@(block_struct) Quant8x8 (block_struct.data,qScale));
        zig_zag = blockproc(quantize_dct,[8 8],@(block_struct) ZigZag8x8 (block_struct.data));
        EOB = 4000;
        blocks = mat2cell(zig_zag, repmat(64,[1 size(zig_zag,1)/64]), repmat(3,[1 size(zig_zag,2)/3]));
        column_zigzag =  cellfun(@(x) reshape(x,[192,1]), blocks, 'UniformOutput',false);    
        dst_cell = cellfun(@(x) ZeroRunEnc_EoB(x, EOB)', column_zigzag, 'UniformOutput',false);       
        dst = dst_cell(:);
        dst = cell2mat(dst);
end


function dst = IntraDecode(src, img_size , qScale)
    EOB = 4000;
    rec_code = ZeroRunDec_EoB(src(:), EOB)';
    block = mat2cell(rec_code, repmat(192,[1 size(rec_code,1)/192]), [1]);
    %192x1 rehsapr 64x3
    reshaped_rec_code = cellfun(@(x) reshape(x,[64,3]), block, 'UniformOutput',false); 
    
    reshaped_cell = reshape(reshaped_rec_code, img_size(1)/8, img_size(2)/8);
    
    reshaped_rec = cell2mat(reshaped_cell);
    
    %reshaped_rec = reshape(reshaped_rec, 64*img_size(1)/8,3*img_size(2)/8);
    inv_zig_zag = blockproc(reshaped_rec,[64 3],@(block_struct) DeZigZag8x8 (block_struct.data));
    inv_quantization = blockproc(inv_zig_zag,[8 8],@(block_struct) DeQuant8x8 (block_struct.data,qScale));
    inv_dct = blockproc(inv_quantization,[8 8],@(block_struct) IDCT8x8 (block_struct.data));
    dst = ictYCbCr2RGB(inv_dct);
    
end


%% and many more functions
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

function coeff = DCT8x8(block)
%  Input         : block    (Original Image block, 8x8x3)
%
%  Output        : coeff    (DCT coefficients after transformation, 8x8x3)
    coeff = zeros(size(block));
    for i = 1 : size(block,3)
        coeff(:,:,i) = dct2(block(:,:,i));
    end
    
end
function block = IDCT8x8(coeff)
%  Function Name : IDCT8x8.m
%  Input         : coeff (DCT Coefficients) 8*8*3
%  Output        : block (original image block) 8*8*
    block = zeros(size(coeff));
    for i = 1 : size(coeff,3)
        block(:,:,i) = idct2(coeff(:,:,i));
    end
end

function quant = Quant8x8(dct_block, qScale)
%  Input         : dct_block (Original Coefficients, 8x8x3)
%                  qScale (Quantization Parameter, scalar)
%
%  Output        : quant (Quantized Coefficients, 8x8x3)
    luminance_table = [16 11 10 16 24 40 51 61; 12 12 14 19 26 58 60 55; 14 13 16 24 40 57 69 56; 14 17 22 29 51 87 80 62; 18 55 37 56 68 109 103 77; 24 35 55 64 81 104 113 92; 49 64 78 87 103 121 120 101; 72 92 95 98 112 100 103 99];
    chrominance_table = [17 18 24 47 99 99 99 99; 18 21 26 66 99 99 99 99; 24 13 56 99 99 99 99 99; 47 66 99 99 99 99 99 99; 99 99 99 99 99 99 99 99;99 99 99 99 99 99 99 99; 99 99 99 99 99 99 99 99; 99 99 99 99 99 99 99 99];
    scaled_luminace_table = qScale*luminance_table;
    scaled_chrominance_table = qScale*chrominance_table;
    
    quant = zeros(size(dct_block));
    quant(:,:,1) = round(dct_block(:,:,1)./scaled_luminace_table);
    quant(:,:,2) = round(dct_block(:,:,2)./scaled_chrominance_table);
    quant(:,:,3) = round(dct_block(:,:,3)./scaled_chrominance_table);
end

function dct_block = DeQuant8x8(quant_block, qScale)
%  Function Name : DeQuant8x8.m
%  Input         : quant_block  (Quantized Block, 8x8x3)
%                  qScale       (Quantization Parameter, scalar)
%
%  Output        : dct_block    (Dequantized DCT coefficients, 8x8x3)
    luminance_table = [16 11 10 16 24 40 51 61; 12 12 14 19 26 58 60 55; 14 13 16 24 40 57 69 56; 14 17 22 29 51 87 80 62; 18 55 37 56 68 109 103 77; 24 35 55 64 81 104 113 92; 49 64 78 87 103 121 120 101; 72 92 95 98 112 100 103 99];
    chrominance_table = [17 18 24 47 99 99 99 99; 18 21 26 66 99 99 99 99; 24 13 56 99 99 99 99 99; 47 66 99 99 99 99 99 99; 99 99 99 99 99 99 99 99;99 99 99 99 99 99 99 99; 99 99 99 99 99 99 99 99; 99 99 99 99 99 99 99 99];
    scaled_luminace_table = qScale*luminance_table;
    scaled_chrominance_table = qScale*chrominance_table;
    
    dct_block(:,:,1) = scaled_luminace_table.*quant_block(:,:,1);
    dct_block(:,:,2) = scaled_chrominance_table.*quant_block(:,:,2);
    dct_block(:,:,3) = scaled_chrominance_table.*quant_block(:,:,3);
    
end

function zz = ZigZag8x8(quant)
%  Input         : quant (Quantized Coefficients, 8x8x3)
%
%  Output        : zz (zig-zag scaned Coefficients, 64x3)
    zigzag_index_table = [1 2 6 7 15 16 28 29; 3 5 8 14 17 27 30 43; 4 9 13 18 26 31 42 44; 10 12 19 25 32 41 45 54; 11 20 24 33 40 46 53 55; 21 23 34 39 47 52 56 61; 22 35 38 48 51 57 60 62; 36 37 49 50 58 59 63 64];
    quant1 = quant(:,:,1);
    zz1(zigzag_index_table(:)) = quant1(:);
    
    quant2 = quant(:,:,2);
    zz2(zigzag_index_table(:)) = quant2(:);
    
    quant3 = quant(:,:,3);
    zz3(zigzag_index_table(:)) = quant3(:);
    
    zz = [zz1',zz2', zz3']
   
end

function coeffs = DeZigZag8x8(zz)
%  Function Name : DeZigZag8x8.m
%  Input         : zz    (Coefficients in zig-zag order)
%
%  Output        : coeffs(DCT coefficients in original order)
    zigzag_index_table = [1 2 6 7 15 16 28 29; 3 5 8 14 17 27 30 43; 4 9 13 18 26 31 42 44; 10 12 19 25 32 41 45 54; 11 20 24 33 40 46 53 55; 21 23 34 39 47 52 56 61; 22 35 38 48 51 57 60 62; 36 37 49 50 58 59 63 64];
    zz1  = zz(:,1);
    deZigZag8x8_1= zz1( zigzag_index_table(:) );
    deZigZag8x8_1 = reshape(deZigZag8x8_1, 8, 8);
    
    zz2  = zz(:,2);
    deZigZag8x8_2= zz2( zigzag_index_table(:) );
    deZigZag8x8_2 = reshape(deZigZag8x8_2, 8, 8);
    
    zz3  = zz(:,3);
    deZigZag8x8_3= zz3( zigzag_index_table(:) );
    deZigZag8x8_3 = reshape(deZigZag8x8_3, 8, 8);
    
    
    coeffs(:,:,1) = deZigZag8x8_1;
    coeffs(:,:,2) = deZigZag8x8_2;
    coeffs(:,:,3) = deZigZag8x8_3;
    
    
    
    
end

function zze = ZeroRunEnc_EoB(zz, EOB)
n =length(zz);
zero_count=0;
zze = [];
i =1;

while (i<=n)    
   if zz(i) == 0 && zero_count~=0
       zze = [zze(1:numel(zze)-1),zero_count];
       zero_count = zero_count+1;
   elseif zz(i) == 0 && zero_count==0
        zze = [zze 0 0];
        zero_count = zero_count+1;
    
   elseif zz(i)~=0
       zze = [zze zz(i)];
       zero_count = 0;
   end
   
    if mod(i,64)==0 && zz(i)==0
    zze = [zze(1:numel(zze)-2),EOB];
    zero_count = 0;
    end
    i = i+1;
end
end



function dst = ZeroRunDec_EoB(src, EoB)
%  Function Name : ZeroRunDec1.m zero run level decoder
%  Input         : src (zero run encoded sequence 1xM with EoB signs)
%                  EoB (end of block sign)
%
%  Output        : dst (reconstructed zig-zag scanned sequence 1xN)
n = length(src);
current_index = 1;
i = 1;

dst = [];
n = length(src);
current_index = 1;
i = 1;
while (i<=n)
    if (src(i)~=0)&&(src(i)~=EoB)
        dst(current_index) = src(i);
        current_index =numel(dst)+1;
        i = i+1;
    elseif src(i)==0
        dst(current_index:current_index+src(i+1)) = 0;
        current_index = numel(dst)+1;
        i = i+2;
    elseif src(i)==EoB
        num_pad_zero = 64 - mod(numel(dst),64);
        dst(current_index:current_index+num_pad_zero-1) = 0;
        current_index = numel(dst)+1;
        i = i+1;
    
    end
end
end


