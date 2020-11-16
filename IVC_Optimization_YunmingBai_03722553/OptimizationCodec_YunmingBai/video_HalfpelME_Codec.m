tic; 

%set scaled quantization, control rate 
scales = [0.05, 0.07, 0.2, 0.4, 0.8, 1.0, 1.5, 2, 3, 4, 4.5];
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
I_rec_RGB = ictYCbCr2RGB(I_rec_YCbCr);
PSNR(1) = calcPSNR(first_frame, I_rec_RGB);


decoded_frame{1}= I_rec_YCbCr;

for i = 2:21

%get previous reconstructed image as reference image  
ref_im = decoded_frame{i-1};
%current encoded image YCbCr 
im = double(imread(['foreman',num2str(i+20-1,'%04d'),'.bmp']));
im1 = ictRGB2YCbCr(im);
%motion estimation, get error image (low energy variance) and motion vector
[error_im, motion_vector] = motionEstimation_halfpel(im1, ref_im);
% error image DCT trandorm Intraencode
residual = IntraEncode(error_im, qScale);
current_min_residual = min(residual);
current_min_motionvector = min(motion_vector(:));
current_max_motionvector = max(motion_vector(:));

if i == 2 | current_min_residual<a_residual | current_min_motionvector<a_mv | current_max_motionvector>b_mv
    % if second frame or need to update the lower bound of Hufmann table 
    % train residual Huffman table through first and second frame
    a_residual = min(residual);
    b_residual = max(residual);
    H_residual = hist(residual(:),a_residual:b_residual);
    H_residual = H_residual/sum(H_residual);
    [ BinaryTree_residual, ~, BinCode_residual, Codelengths_residual] = buildHuffman( H_residual );
 
    % train motion vector Huffman table through first and second frame
    a_mv = min(motion_vector(:));
    
     b_mv= max(motion_vector(:));
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
rec_img_YCbCr = motionCompensation_halfpel(ref_im,rec_residual,rec_mv);
rec_img_RGB = ictYCbCr2RGB(rec_img_YCbCr);
PSNR(i) = calcPSNR(im, rec_img_RGB);
decoded_frame{i}= rec_img_YCbCr;

end

%PSNR: 1*21 vector for one quantization scale
%bitPerPixel: 1*21 vector bit-rate for one quantization scale  
mean_PSNR = mean(PSNR);
means_rate = mean(bitPerPixel);

distortion_video_halfpel(scaleIdx) = mean_PSNR;
Rate_video_halfpel(scaleIdx) = means_rate;

end

save('distortion_video_halfpel.mat','distortion_video_halfpel');
save('rate_video_halfpel.mat','Rate_video_halfpel');

toc;

%% all needed functions

function [ BinaryTree, HuffCode, BinCode, Codelengths] = buildHuffman( p );

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
    EOB = 4000;
    rec_code = ZeroRunDec_EoB(src(:), EOB)';
    
    %8*8*3= 192*1 each block
    %seperate the long vector into 192*1 vectors
    block = mat2cell(rec_code, repmat(192,[1 size(rec_code,1)/192]), [1]);
    %192x1 reshape into 64x3
    reshaped_rec_code = cellfun(@(x) reshape(x,[64,3]), block, 'UniformOutput',false); 
    
    %reshape cell_vector of 64*3 matrix into cellmatrix
    reshaped_cell = reshape(reshaped_rec_code, img_size(1)/8, img_size(2)/8);
    
    reshaped_rec = cell2mat(reshaped_cell);
    
    %de-zig-zag: 64*3 each column one channel into 8*8*3
    inv_zig_zag = blockproc(reshaped_rec,[64 3],@(block_struct) DeZigZag8x8 (block_struct.data));
    
    %dequantization 8*8*3 block, from quantization index into value 
    inv_quantization = blockproc(inv_zig_zag,[8 8],@(block_struct) DeQuant8x8 (block_struct.data,qScale));
    
    %inverse DCT
    dst = blockproc(inv_quantization,[8 8],@(block_struct) IDCT8x8 (block_struct.data));
    
end


function dst = IntraEncode(YCbCr_image, qScale)
%  Function Name : IntraEncode.m
%  Input         : image (YCbCr Image)
%                  qScale(quantization scale)
%  Output        : dst   (sequences after zero-run encoding, 1xN) used for encoding 
        %block process, convert to dct coefficients
        dct_coeff = blockproc(YCbCr_image,[8 8],@(block_struct) DCT8x8 (block_struct.data));
        %block process, quantization
        quantize_dct = blockproc(dct_coeff,[8 8],@(block_struct) Quant8x8 (block_struct.data,qScale));
        %block process, zigzag scanning, each block dimension 64*3 
        zig_zag = blockproc(quantize_dct,[8 8],@(block_struct) ZigZag8x8 (block_struct.data));
        
        EOB = 4000;
        %split zig_zag matrix into 64*3 cell matrix, each cell one block
        %size 8*8 with 3 channels
        blocks = mat2cell(zig_zag, repmat(64,[1 size(zig_zag,1)/64]), repmat(3,[1 size(zig_zag,2)/3]));
        
        %resize each block into vector 64*3=192*1 
        column_zigzag =  cellfun(@(x) reshape(x,[192,1]), blocks, 'UniformOutput',false); 
        
        % encode each cell vector 192*1, encode vector zreo_run_length_encoding    
        dst_cell = cellfun(@(x) ZeroRunEnc_EoB(x, EOB)', column_zigzag, 'UniformOutput',false);       
        
        %code word into vell_vector, beacasue different cell different code_word lenght
        %convert cell into vector
        dst = dst_cell(:);
        dst = cell2mat(dst);
end


function rec_img_YCbCr = motionCompensation_halfpel(ref_im,residual,mv)
%  Input         : ref_image(Reference Image, size: height x width x 3, YCbCr)
%                  residual (residual Image, size: height x width x 3, YCbCr)
%                  mv:motion vector
%
%  Output        : motion_vectors_indices (Motion Vector Indices, size: (height/8) x (width/8) x 1 )
%                  error_im: current_img - (reference inage+ motion vector)
     estimated_image = SSD_rec_halfpel(ref_im, mv);
     rec_img_YCbCr = residual + estimated_image;
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

function motion_vectors_indices = ssd_halfpel(ref_im, current_image)
%input only Y channel for motion estimation

padded_reference = padarray(ref_im,[4 4],0,'both' );
motion_vectors_indices = zeros(size(current_image,1)/8,size(current_image,2)/8);

% i: index row_block 
    %j: index column_block  
for i = 1:size(current_image,1)/8
    for j = 1:size(current_image,2)/8
        %from index of clolumn and row block get image 
        current_block = current_image((8*i-7):i*8, (8*j-7):j*8);
        min_ssd_interger = 255^2*64;

        %search area 
        for k = -4:4
            for m = -4:4
                %rowstart index: 8*i-7+k+4 because of padded 4 pixels on left side 
                %same for column start_index 
                row_range = (8*i-3+k):(8*i+4+k);
                column_range = (8*j-3+m):(8*j+4+m);
                reference_block = padded_reference(row_range, column_range);
                distortion = sum((reference_block-current_block).^2,'all');
                
                if distortion<min_ssd_interger
                    min_ssd_interger = distortion;
                    row_integer = k;
                    column_integer =m; 
                end
            end
        end
        %half-pel around k,m possible 
        
        
%half-pel need to new method to combine the row and column and row
%vector with 0.5 possible XX.1.2....9 (9 positions)
%motion_vectors_indices(i,j) = (row_integer+4)*9 + (column_integer+5);
 
motion_vector_integer = (row_integer+4)*9 + (column_integer+5);

row_range_halfpel = (8*i-3+row_integer-1):(8*i+4+row_integer+1);
column_range_halfpel = (8*j-3+column_integer-1):(8*j+4+column_integer+1);
reference_block_halfpel = padded_reference(row_range_halfpel, column_range_halfpel);
                
       
%interpolate take 10*10 reference block, interpolate to 19*19 block
%original optimal interger pel at (3,3), search neighbouring halfpel blocks
x_range= 1:10;
y_range =1:10;
[x,y] = meshgrid(x_range,y_range);

x_interp__range = 1:0.5:10;
y_interp_range = 1:0.5:10;
[x_interp, y_interp] = meshgrid(x_interp__range,y_interp_range);

interpolated_ref_block = interp2(x,y,reference_block_halfpel,x_interp,y_interp,'linear');

for halfpel_y = -1:1
    for halfpel_x = -1:1
        halpfel_reference_block = interpolated_ref_block(3+halfpel_y:2:3+halfpel_y+14,3+halfpel_x:2:3+halfpel_x+14);
        halfpel_error(halfpel_y+2,halfpel_x+2) =  sum((halpfel_reference_block-current_block).^2,'all');
    end
end

[~,ind] = min(halfpel_error(:));

motion_vectors_indices(i,j) = motion_vector_integer+ind*100;
    
    end
    
end
end

function rec_image = SSD_rec_halfpel(ref_image, motion_vectors)

rec_image = zeros(size(ref_image));
padded_reference = padarray(ref_image,[4 4],0,'both' );

%motion vector row and column blocks, each block one motion vector(combination of row and column motion)
for i = 1: size(motion_vectors,1)
    for j = 1: size(motion_vectors,2)
                        
            %from scalar motion vector get row and column motion seperately  
            motion_vector_index_integer = mod(motion_vectors(i,j),100);
            row_ind = ceil(motion_vector_index_integer/9)-5;
            column_ind = motion_vector_index_integer-(row_ind+4)*9-5; 
            
            %rwo and column range of padded_image  
            row_range = (4+i*8-7+row_ind-1): (4+i*8+row_ind+1);
            column_range = (4+j*8-7+column_ind-1): (4+j*8+column_ind+1);
            reference_block_halfpel_Y = padded_reference(row_range, column_range,1);
            rec_image((8*i-7):8*i,(8*j-7):8*j,2) = padded_reference(4+i*8-7+row_ind:4+i*8+row_ind, 4+j*8-7+column_ind:4+j*8+column_ind, 2);
            rec_image((8*i-7):8*i,(8*j-7):8*j,3) = padded_reference(4+i*8-7+row_ind:4+i*8+row_ind, 4+j*8-7+column_ind:4+j*8+column_ind, 3);
            
            
            %interpolate
            %interpolate take 10*10 reference block, interpolate to 19*19 block
            %original optimal interger pel at (3,3), search neighbouring halfpel blocks
            x_range= 1:10;
            y_range =1:10;
            [x,y] = meshgrid(x_range,y_range);

            x_interp__range = 1:0.5:10;
            y_interp_range = 1:0.5:10;
            [x_interp, y_interp] = meshgrid(x_interp__range,y_interp_range);

            interpolated_ref_block = interp2(x,y,reference_block_halfpel_Y,x_interp,y_interp,'linear');
           
            %from full-interpolated referecne image get reference block
            halfpel_position_ind = floor(motion_vectors(i,j)/100);
            [halfpel_row,halfpel_col] = ind2sub([3,3],halfpel_position_ind);
            row_halfpel_start = 3+(-2+halfpel_row);
            column_halfpel_start = 3+(-2+halfpel_col);
            
            row_range_halfpel = row_halfpel_start:2:(row_halfpel_start+14);
            column_range_halfpel = column_halfpel_start:2:(column_halfpel_start+14);
            
            rec_image((8*i-7):8*i,(8*j-7):8*j,1) = interpolated_ref_block(row_range_halfpel,column_range_halfpel);

            
            
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

function [err_im, motion_vectors] = motionEstimation_halfpel(current_im, ref_im)
%  Input         : ref_image(Reference Image, size: height x width x 3, YCbCr)
%                  current_im (Current Image, size: height x width x 3, YCbCr)
%
%  Output        : motion_vectors_indices (Motion Vector Indices, size: (height/8) x (width/8) x 1 )
%                  error_im: current_img - (reference inage+ motion vector)
% use luminace channel to motion estimation     
     motion_vectors = ssd_halfpel(ref_im(:,:,1), current_im(:,:,1));
     estimated_image = SSD_rec_halfpel(ref_im, motion_vectors);
     err_im = current_im - estimated_image;
end

