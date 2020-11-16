function image = InvVectorQuantizer(qImage, clusters, block_size)
%  Function Name : VectorQuantizer.m
%  Input         : qImage     (Quantized Image)
%                  clusters   (Quantization clusters)
%                  block_size (Block Size)
%  Output        : image      (Dequantized Images)

C = cell(size(qImage));
for i = 1:size(qImage,1)*size(qImage,2)*size(qImage,3)
         C{i} = reshape(clusters(qImage(i),:),[block_size block_size]);
end

image = cell2mat(C);

end