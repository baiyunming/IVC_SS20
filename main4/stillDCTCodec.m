lena_small = double(imread('lena_small.tif'));
image_lena     = double(imread('lena.tif'));

scales = [0.07, 0.2, 0.4, 0.8, 1.0, 1.5, 2, 3, 4, 4.5]; 

for scaleIdx = 1 : numel(scales)
    
qScale = scales(scaleIdx); 
k_small  = IntraEncode(lena_small, qScale);
k        = IntraEncode(image_lena, qScale);

% use pmf of k_small to build and train huffman table
a = min(k);
b = max(k);
H = hist(k_small(:),a:b);
H = H/sum(H);
[ BinaryTree, HuffCode, BinCode, Codelengths] = buildHuffman( H );


bytestream = enc_huffman_new( k-a+1 , BinCode, Codelengths);
bitPerPixel = (numel(bytestream)*8) / (numel(image_lena)/3);


k_rec = double(dec_huffman_new ( bytestream, BinaryTree,  size( k(:),1 ) ))+a-1;
I_rec = IntraDecode(k_rec, size(image_lena),qScale);
PSNR = calcPSNR(image_lena, I_rec);

bbp(scaleIdx) = bitPerPixel;
distortion(scaleIdx) = PSNR;
 
end

