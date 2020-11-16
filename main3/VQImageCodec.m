%% Main
bits         = 8;
epsilon      = 0.1;
block_size   = 2;
%% lena small for VQ training
image_small  = double(imread('lena.tif'));
clusters= VectorQuantizer(image_small, bits, epsilon, block_size);
qImage_small = ApplyVectorQuantizer(image_small, clusters, block_size);
%% Huffman table training lena small
max = size(clusters,1);
H = hist(qImage_small(:),1:max);
H = H/sum(H);
[ BinaryTree, HuffCode, BinCode, Codelengths] = buildHuffman( H );
%% 
image  = double(imread('lena.tif'));
qImage = ApplyVectorQuantizer(image, clusters, block_size);
%% Huffman encoding
bytestream = enc_huffman_new( qImage(:) , BinCode, Codelengths);
%%
p  = (numel(bytestream) * 8) / (numel(image)/3);
%% Huffman decoding
qReconst_image = double(reshape( dec_huffman_new ( bytestream, BinaryTree, size( qImage(:),1 ) ), size(qImage)));
%%
reconst_image  = InvVectorQuantizer(qReconst_image, clusters, block_size);
PSNR = calcPSNR(image, reconst_image);





