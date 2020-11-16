function [clusters] = VectorQuantizer(image, bits, epsilon, bsize)

index = 0 :2^bits-1;
rep_value = floor((256/2^bits)*(index+0.5));
code_book = zeros(2^bits,bsize^2*2+1);

for ind = 1:size(code_book,1)
    code_book(ind,1:4) =repmat(rep_value(ind),1,bsize^2);
end

clusters = code_book(:,1:4);

mse_old = 0;
mse_new = 0;

[w,h,c] = size(image);

%apply the updated cluster-table, knnsearch
cell_Image =  mat2cell(image,repmat(bsize,1,w/bsize),repmat(bsize,1,h/bsize),ones(1,c));
reshaped_cell = cellfun(@(x) reshape(x,[1 4]), cell_Image, 'UniformOutput',false);
block = cell2mat(reshaped_cell(:));

while 1

[index,distance] = knnsearch(clusters,block,'distance','euclidean');
mse_new = sum(distance.^2)/numel(index);


%update-summation and counter for each representative vector
 for i = 1:numel(index)
      code_book(index(i),5:8) = code_book(index(i),5:8)+block(i,:);
      code_book(index(i),9) = code_book(index(i),9)+1;
 end

 
 % update the nonzero-rows
 nonzero_index = find(code_book(:,9) ~= 0);
 code_book(nonzero_index,1:4) = round(code_book(nonzero_index,5:8)./code_book(nonzero_index,9));
 
 
 %cell_splitting
     while (nnz(~code_book(:,9))) 
        zero_index = find( code_book(:,9) == 0);
        replace_index = zero_index(1);
        [~,max_index] = max(code_book(:,9));
        code_book(replace_index,1:3) = code_book(max_index,1:3);
        code_book(replace_index,4) = code_book(max_index,4)+1;
        code_book(replace_index,9) = floor(code_book(max_index,9)/2);
        code_book(max_index,9) = ceil(code_book(max_index,9)/2);
     end
   
     
code_book(:,5:8) = 0; 
code_book(:,9) = 0; 
clusters = code_book(:,1:4);
    

if (abs(mse_new -mse_old)/mse_new)<epsilon
      break;
end
    
mse_old = mse_new;
mse_new = 0;
 
end
end

    



