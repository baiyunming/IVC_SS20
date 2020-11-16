% Read Image
imageLena = double(imread('lena.tif'));
% create the predictor and obtain the residual image

 [w h c] = size(imageLena);
 PredImage = zeros(w,h,c);
 for k = 1:c 
     for i = 1:w
            PredImage(i,1,k)=imageLena(i,1,k);
        for j = 1:h-1          
            PredImage(i,j+1,k)=imageLena(i,j,k);
        end
     end
 end

resImage  = imageLena - PredImage;
% get the PMF of the residual image
range = -255:255;
pmfRes    = stats_marg(resImage, range);
% calculate the entropy of the residual image
H_res     = calc_entropy(pmfRes);

fprintf('H_err_OnePixel   = %.2f bit/pixel\n',H_res);

% Put all sub-functions which are called in your script here.
function pmf = stats_marg(image, range)
    [counts,~] = hist(image(:),range);
    pmf = counts/sum(counts);
end

function H = calc_entropy(pmf)
    H = -sum(pmf.*log2(pmf),'omitnan');
end