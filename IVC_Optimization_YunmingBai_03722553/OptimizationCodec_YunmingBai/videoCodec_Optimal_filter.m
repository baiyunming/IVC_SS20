tic;

%The computation cost for finding the least-square filter is quite large because of the high dimensionality 
%of the image and the optimization problem, it takes hours to find the optimal kernel, so in this script
%the precomputed optimal kernel which are stored in 'optimal_filter.mat' is loaded and directly uesed as In-loop filter 
%each column of optimal_filter is a filter, and the order following the
%frame and quantization-scale order, e.g. the first 21 kernel correspond to
%frame foreman0020 to foreman0040 with quantization scale 0.07; 

%% load precomputed optimal Kernel for each frame and bit-rate
load('optimal_filter.mat');
kernel_ind = 1 ;

%% when computing the optimal filter, instead of running the last section (load the 'optimal_filter.mat') 
%the optimal_Kernel is initilialised as empty matrix
%optimal_Kernel = []; 


%% set scaled quantization, control rate 
scales = [0.07, 0.2, 0.3, 0.4, 0.8, 1.0, 1.5, 2, 3, 4]; 


for scaleIdx = 1 : numel(scales)
  
qScale   = scales(scaleIdx);   
first_frame =  double(imread('foreman0020.bmp'));

%encode and decode first frame
lena_small = double(imread('lena_small.tif'));
k_small  = IntraEncode( ictRGB2YCbCr(lena_small), qScale);
%DCT coefficients, transmitted data, run-length coding, N*1 vector
k_first_frame = IntraEncode(ictRGB2YCbCr(first_frame), qScale);

% use pmf of lena_small to build and train huffman table for first frame
a_firstframe = min(k_first_frame);
b_firstframe = max(k_first_frame);
H = hist(k_small(:),a_firstframe:b_firstframe);
H = H/sum(H);
[ BinaryTree_firstframe, ~, BinCode_firstframe, Codelengths_firstframe] = buildHuffman( H );
    
bytestream = enc_huffman_new( k_first_frame-a_firstframe+1 , BinCode_firstframe, Codelengths_firstframe);
bitPerPixel(1) = (numel(bytestream)*8) / (numel(first_frame)/3);

% get run-length code word
k_rec = double(dec_huffman_new ( bytestream, BinaryTree_firstframe,  size( k_first_frame(:),1 ) ))+a_firstframe-1;
% decode run-lenght codwword and get DCT coefficients, same size as image with 3 RGB channel
I_rec_YCbCr = IntraDecode(k_rec, size(first_frame),qScale);

%function getFilter construct the Least-Square Optimization problem between encoded frame and filter reconstructed frame
%function getFilter return the optimal filter and the filtered reconstructed frame
%[kernel, filter_rec_Im] = getFilter(ictRGB2YCbCr(first_frame), I_rec_YCbCr);
%optimal_Kernel = [optimal_Kernel, kernel]; 

%directly apply the precomputed optimal filter (column vector) stored in optimal_Kernel.mat
kernel_column = optimal_Kernel(:,kernel_ind);
%convert column into kernel matrix with dimension 5*5
kernel_matrix = reshape(kernel_column, 5,5)';
%filter the luminance channel recosntructed frame 
filter_rec_Y = imfilter(I_rec_YCbCr(:,:,1),kernel_matrix,'replicate');
%concantenate the filtered luminanc end chrominance channel of the original reconstruted image
filter_rec_Im = cat(3, filter_rec_Y, I_rec_YCbCr(:,:,2:3));

I_rec_RGB = ictYCbCr2RGB(filter_rec_Im);
PSNR(1) = calcPSNR(first_frame, I_rec_RGB);

kernel_ind = kernel_ind+1;
decoded_frame{1}= filter_rec_Im;

for i = 2:21

%get previous reconstructed image as reference image  
ref_im = decoded_frame{i-1};
%current encoded image YCbCr 
im = double(imread(['foreman',num2str(i+20-1,'%04d'),'.bmp']));
im1 = ictRGB2YCbCr(im);
%motion estimation, get error image (low energy variance) and motion vector
[error_im, motion_vector] = motionEstimation(im1, ref_im);
% error image DCT trandorm Intraencode
residual = IntraEncode(error_im, qScale);
current_min_residual = min(residual);


if i == 2 | current_min_residual<a_residual
    % if second frame or need to update the lower bound of Hufmann table 
    % train residual Huffman table through first and second frame
    a_residual = min(residual);
    b_residual = max(residual);
    H_residual = hist(residual(:),a_residual:b_residual);
    H_residual = H_residual/sum(H_residual);
    [ BinaryTree_residual, ~, BinCode_residual, Codelengths_residual] = buildHuffman( H_residual );
 
    % train motion vector Huffman table through first and second frame
    a_mv = min(motion_vector(:));
    b_mv = max(motion_vector(:));
    H_mv = hist(motion_vector(:),a_mv:b_mv);
    H_mv = H_mv/sum(H_mv);
    [ BinaryTree_mv, ~, BinCode_mv, Codelengths_mv] = buildHuffman( H_mv );
end

%encode residual and motion vector respectively through trained Hufmann table
bytestream_residual = enc_huffman_new(residual-a_residual+1 , BinCode_residual, Codelengths_residual);
bytestream_mv = enc_huffman_new( motion_vector(:)-a_mv+1 , BinCode_mv, Codelengths_mv);
bitPerPixel(i) = ((numel(bytestream_residual)+numel(bytestream_mv))*8) / (numel(im)/3);


%decode transmitted bytestream for residual and motion vector
decoded_residual = double(dec_huffman_new ( bytestream_residual, BinaryTree_residual,  size( residual(:),1 ) ))+a_residual-1;
%intradecode residual image 
rec_residual = IntraDecode(decoded_residual, size(error_im),qScale);
%decode motion vector
rec_mv = reshape(double(dec_huffman_new ( bytestream_mv, BinaryTree_mv,  size(motion_vector(:),1) ))+a_mv-1, size(motion_vector,1),size(motion_vector,2));

%motion compensation
%from reference image, motion vector and residualimage get recosntructed current image 
rec_img_YCbCr = motionCompensation(ref_im,rec_residual,rec_mv);


%% directly apply the precomputed optimal filter (column vector) stored in optimal_Kernel.mat
kernel_column = optimal_Kernel(:,kernel_ind);
kernel_matrix = reshape(kernel_column, 5,5)';
filter_rec_Y = imfilter(rec_img_YCbCr(:,:,1),kernel_matrix,'replicate');
filter_rec_Im = cat(3, filter_rec_Y, rec_img_YCbCr(:,:,2:3));

%% find the optimal postfilter of the current frame and quantization scale, instead of running the last section
%[kernel_column, filter_rec_Im]= getFilter(im1, rec_img_YCbCr);
%optimal_Kernel = [optimal_Kernel, kernel_column]; 

%% convert YCbCr into RGB and compute PSNR 
rec_img_RGB = ictYCbCr2RGB(filter_rec_Im);

PSNR(i) = calcPSNR(im, rec_img_RGB);
decoded_frame{i}= filter_rec_Im;
kernel_ind = kernel_ind+1;

end

%PSNR: 1*21 vector for one quantization scale
%bitPerPixel: 1*21 vector bit-rate for one quantization scale  
mean_PSNR = mean(PSNR);
means_rate = mean(bitPerPixel);

distortion_LS_filter(scaleIdx) = mean_PSNR;
rate_video_LS_filter(scaleIdx) = means_rate;

end

save('distortion_LS_filter.mat','distortion_LS_filter');
save('rate_video_LS_filter.mat','rate_video_LS_filter');

toc;

%% all needed functions
function [ BinaryTree, HuffCode, BinCode, Codelengths] = buildHuffman( p )

global y

p=p(:)/sum(p)+eps;              % normalize histogram
p1=p;                           % working copy

c=cell(length(p1),1);			% generate cell structure 

for i=1:length(p1)				% initialize structure
   c{i}=i;						
end

while size(c)-2					% build Huffman tree
	[p1,i]=sort(p1);			% Sort probabilities
	c=c(i);						% Reorder tree.
	c{2}={c{1},c{2}};           % merge branch 1 to 2
    c(1)=[];	                % omit 1
	p1(2)=p1(1)+p1(2);          % merge Probabilities 1 and 2 
    p1(1)=[];	                % remove 1
end

%cell(length(p),1);              % generate cell structure
getcodes(c,[]);                  % recurse to find codes
code=char(y);

[numCodes maxlength] = size(code); % get maximum codeword length

% generate byte coded huffman table
% code

length_b=0;
HuffCode=zeros(1,numCodes);
for symbol=1:numCodes
    for bit=1:maxlength
        length_b=bit;
        if(code(symbol,bit)==char(49)) HuffCode(symbol) = HuffCode(symbol)+2^(bit-1)*(double(code(symbol,bit))-48);
        elseif(code(symbol,bit)==char(48))
        else 
            length_b=bit-1;
            break;
        end;
    end;
    Codelengths(symbol)=length_b;
end;

BinaryTree = c;
BinCode = code;

clear global y;

return
end


%----------------------------------------------------------------
function getcodes(a,dum)       
global y                            % in every level: use the same y
if isa(a,'cell')                    % if there are more branches...go on
         getcodes(a{1},[dum 0]);    % 
         getcodes(a{2},[dum 1]);
else   
   y{a}=char(48+dum);   
end
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
   PSNR = 10* log10((255)^2/MSE);
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
function [output] = dec_huffman_new (bytestream, BinaryTree, nr_symbols)

output = zeros(1,nr_symbols);
ctemp = BinaryTree;

dec = zeros(size(bytestream,1),8);
for i = 8:-1:1
    dec(:,i) = rem(bytestream,2);
    bytestream = floor(bytestream/2);
end

dec = dec(:,end:-1:1)';
a = dec(:);

i = 1;
p = 1;
while(i <= nr_symbols)&&p<=max(size(a))
    while(isa(ctemp,'cell'))
        next = a(p)+1;
        p = p+1;
        ctemp = ctemp{next};
    end;
    output(i) = ctemp;
    ctemp = BinaryTree;
    i=i+1;
end;

end
function yuv = ictRGB2YCbCr(rgb)
    yuv (:,:,1) = 0.299*rgb(:,:,1) + 0.587*rgb(:,:,2) + 0.114*rgb(:,:,3); 
    yuv (:,:,2) = -0.169*rgb(:,:,1) - 0.331*rgb(:,:,2) + 0.5*rgb(:,:,3); 
    yuv (:,:,3) = 0.5*rgb(:,:,1) - 0.419*rgb(:,:,2) -0.081*rgb(:,:,3); 
end

    function rgb = ictYCbCr2RGB(yuv)
    rgb(:,:,1) = yuv(:,:,1) + 1.402*yuv(:,:,3);
    rgb(:,:,2) = yuv(:,:,1) - 0.344*yuv(:,:,2) -0.714*yuv(:,:,3);
    rgb(:,:,3) = yuv(:,:,1) + 1.772*yuv(:,:,2);
    end


    function block = IDCT8x8(coeff)
%  Function Name : IDCT8x8.m
%  Input         : coeff (DCT Coefficients) 8*8*3
%  Output        : block (original image block) 8*8*3
    block = zeros(size(coeff));
    for i = 1 : size(coeff,3)
        block(:,:,i) = idct2(coeff(:,:,i));
    end
    end

    
function dst = IntraDecode(src, img_size , qScale)
    % dst output YCbCr image
    EOB = 1000;
    rec_code = ZeroRunDec_EoB(src(:), EOB)';
    block = mat2cell(rec_code, repmat(192,[1 size(rec_code,1)/192]), [1]);
    %192x1 rehsapr 64x3
    reshaped_rec_code = cellfun(@(x) reshape(x,[64,3]), block, 'UniformOutput',false); 
    
    reshaped_cell = reshape(reshaped_rec_code, img_size(1)/8, img_size(2)/8);
    
    reshaped_rec = cell2mat(reshaped_cell);
    
    %reshaped_rec = reshape(reshaped_rec, 64*img_size(1)/8,3*img_size(2)/8);
    inv_zig_zag = blockproc(reshaped_rec,[64 3],@(block_struct) DeZigZag8x8 (block_struct.data));
    inv_quantization = blockproc(inv_zig_zag,[8 8],@(block_struct) DeQuant8x8 (block_struct.data,qScale));
    dst = blockproc(inv_quantization,[8 8],@(block_struct) IDCT8x8 (block_struct.data));
    
end


function dst = IntraEncode(YCbCr_image, qScale)
%  Function Name : IntraEncode.m
%  Input         : image (YCbCr Image)
%                  qScale(quantization scale)
%  Output        : dst   (sequences after zero-run encoding, 1xN)
        dct_coeff = blockproc(YCbCr_image,[8 8],@(block_struct) DCT8x8 (block_struct.data));
        quantize_dct = blockproc(dct_coeff,[8 8],@(block_struct) Quant8x8 (block_struct.data,qScale));
        zig_zag = blockproc(quantize_dct,[8 8],@(block_struct) ZigZag8x8 (block_struct.data));
        EOB = 1000;
        blocks = mat2cell(zig_zag, repmat(64,[1 size(zig_zag,1)/64]), repmat(3,[1 size(zig_zag,2)/3]));
        column_zigzag =  cellfun(@(x) reshape(x,[192,1]), blocks, 'UniformOutput',false);    
        dst_cell = cellfun(@(x) ZeroRunEnc_EoB(x, EOB)', column_zigzag, 'UniformOutput',false);       
        dst = dst_cell(:);
        dst = cell2mat(dst);
end

function rec_img_YCbCr = motionCompensation(ref_im,residual,mv)
%  Input         : ref_image(Reference Image, size: height x width x 3, YCbCr)
%                  residual (residual Image, size: height x width x 3, YCbCr)
%                  mv:motion vector
%
%  Output        : motion_vectors_indices (Motion Vector Indices, size: (height/8) x (width/8) x 1 )
%                  error_im: current_img - (reference inage+ motion vector)
     estimated_image = SSD_rec(ref_im, mv);
     rec_img_YCbCr = residual + estimated_image;
end


function [err_im, motion_vectors] = motionEstimation(current_im, ref_im)
%  Input         : ref_image(Reference Image, size: height x width x 3, YCbCr)
%                  current_im (Current Image, size: height x width x 3, YCbCr)
%
%  Output        : motion_vectors_indices (Motion Vector Indices, size: (height/8) x (width/8) x 1 )
%                  error_im: current_img - (reference inage+ motion vector)
     motion_vectors = ssd(ref_im(:,:,1), current_im(:,:,1));
     estimated_image = SSD_rec(ref_im, motion_vectors);
     err_im = current_im - estimated_image;
end


%get scalar quantizaer indexes
%YCbCr each channel different predefined quantization table
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
function motion_vectors_indices = ssd(ref_image, image)
%  Input         : ref_image(Reference Image, size: height x width)
%                  image (Current Image, size: height x width)
%
%  Output        : motion_vectors_indices (Motion Vector Indices, size: (height/8) x (width/8) x 1 )
padded_reference = padarray(ref_image,[4 4],0,'both' );
motion_vectors_indices = zeros(size(image,1)/8,size(image,2)/8);


for i = 1: size(image,1)/8
    for j = 1:size(image,2)/8
        current_block = image((8*i-7):i*8, (8*j-7):j*8);
        min_ssd = 255^2*64;
        for k = -4:4
            for m = -4:4
                row_range = (8*i-3+k):(8*i+4+k);
                column_range = (8*j-3+m):(8*j+4+m);
                reference_block = padded_reference(row_range, column_range);
                distortion = sum((reference_block-current_block).^2,'all');
                if distortion<min_ssd
                    min_ssd = distortion;
                    row = k;
                    column =m; 
                end
            end
        end
        motion_vectors_indices(i,j) = (row+4)*9 + (column+5);
    end
end



end
function rec_image = SSD_rec(ref_image, motion_vectors)
%  Input         : ref_image(Reference Image, YCbCr image)
%                  motion_vectors
%
%  Output        : rec_image (Reconstructed current image, YCbCr image)
padded_reference = padarray(ref_image,[4 4],0,'both' );
rec_image = zeros(size(ref_image));
for i = 1: size(motion_vectors,1)
    for j = 1: size(motion_vectors,2)
       
            motion_vector_index = motion_vectors(i,j);
            row_ind = ceil(motion_vector_index/9)-5;
            column_ind = motion_vector_index-(row_ind+4)*9-5; 
        
            row_range = (4+i*8-7+row_ind): (4+i*8+row_ind);
            column_range = (4+j*8-7+column_ind): (4+j*8+column_ind);
            rec_image((8*i-7):8*i,(8*j-7):8*j,:) = padded_reference(row_range,column_range,:);
    end
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
    
    zz = [zz1',zz2', zz3'];
    
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



function [bytestream] = enc_huffman_new( data, BinCode, Codelengths)

a = BinCode(data(:),:)';
b = a(:);
mat = zeros(ceil(length(b)/8)*8,1);
p  = 1;
for i = 1:length(b)
    if b(i)~=' '
        mat(p,1) = b(i)-48;
        p = p+1;
    end
end
p = p-1;
mat = mat(1:ceil(p/8)*8);
d = reshape(mat,8,ceil(p/8))';
multi = [1 2 4 8 16 32 64 128];
bytestream = sum(d.*repmat(multi,size(d,1),1),2);

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


%dequantization 
%multiply transmitted quantization indexes with quantization table
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


function [kernel, filter_rec_Im]= getFilter(origin_YCbCr, rec_YCbCr)
%input: original YCbCr image
%       reconstructed YCbCr image
%output: kernel (least square) Ax = b 
%        filter_rec_im: filtered reconstructed YCbCr image

padded_matrix = padarray(rec_YCbCr(:,:,1),[2 2],'replicate','both');

ind =1;
for i = 1+2:size(rec_YCbCr,1)+2
    for j = 1+2:size(rec_YCbCr,2)+2
        extract_matrix = padded_matrix(i-2:i+2, j-2:j+2);
        extract_transpose = extract_matrix';
        A(ind,:) = extract_transpose(:);
        ind = ind+1;
    end
end

b = origin_YCbCr(:,:,1)';
b = b(:);

%initila kernel
x0 = [0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0]';


A_dach = A'*A;
b_dach = A'* b;

[kernel,~] = ConjugateGrandient(A_dach,b_dach,x0,eps);



filter_rec_Y = A * kernel;
filter_rec_Y = reshape(filter_rec_Y,[352 288])';

filter_rec_Im = cat(3,filter_rec_Y, rec_YCbCr(:,:,2:3));

end


function [x,steps] = ConjugateGrandient(A,b,x0,eps)
r0 = b - A*x0;
p0 = r0;
if nargin == 3
    eps = 1.0e-6;
end
steps = 0;
while 1
    if abs(p0) < eps
        break;
    end
    steps = steps + 1;
    a0 = r0'*r0/(p0'*A*p0);
    x1 = x0 + a0*p0;

    r1 = r0 -a0*A*p0;

    b0 = r1'*r1/(r0'*r0);

    p1 = r1 + b0*p0;

    x0 = x1;
    r0 = r1;
    p0 = p1;

end
x = x0;
end




