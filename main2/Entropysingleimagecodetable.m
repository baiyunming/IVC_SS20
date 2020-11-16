%% 
imageLena = double(imread('lena.tif'));
imageSail = double(imread('sail.tif'));
imagesmandril = double(imread('smandril.tif'));
%% 
%lena_pmf = histogram(imageLena(:),256);
[counts,~] = hist(imageLena(:),0:255);
lenaPMF = counts/sum(counts);
%% 
input_image = permute(imageLena,[2 1 3]);

jointPMF = zeros(256,256);

for k=0:numel(input_image)/2-1
    jointPMF( input_image(2*k+1)+1,input_image(2*k+2)+1 )=jointPMF(input_image(2*k+1)+1,input_image(2*k+2)+1)+1;
end



%% 
lenaH = -sum(lenaPMF.*log2(lenaPMF),'omitnan');

%pmfLena       = 
%HLena         = 

%fprintf('--------------Using individual code table--------------\n');
%fprintf('lena.tif      H = %.2f bit/pixel\n', HLena);

