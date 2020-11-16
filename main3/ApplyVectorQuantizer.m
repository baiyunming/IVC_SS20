function qImage = ApplyVectorQuantizer(image, clusters, bsize)
%  Function Name : ApplyVectorQuantizer.m
%  Input         : image    (Original Image)
%                  clusters (Quantization Representatives)
%                  bsize    (Block Size)
%  Output        : qImage   (Quantized Image)
[w,h, c] = size(image);
cell_Image =  mat2cell(image,repmat(bsize,1,w/bsize),repmat(bsize,1,h/bsize),ones(1,c));

%cellfun(function,input_cell,'UniformOutput',false cellfunction return cellmatrix)
reshaped_cell = cellfun(@(x) reshape(x,[1 4]), cell_Image, 'UniformOutput',false);
block = cell2mat(reshaped_cell(:));

[Ind,~] = knnsearch(clusters,block,'distance','euclidean');
qImage = reshape(Ind,[w/bsize h/bsize c]);

end