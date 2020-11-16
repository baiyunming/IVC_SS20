% Read Image
imageLena = double(imread('lena.tif'));
% Calculate Joint PMF
jpmfLena  = stats_joint(imageLena);
% Calculate Joint Entropy
Hjoint    = calc_entropy(jpmfLena);
fprintf('H_joint = %.2f bit/pixel pair\n', Hjoint);

% Put all sub-functions which are called in your script here.
function pmf = stats_joint(image)
%  Input         : image (Original Image)
%       
%  Output        : pmf   (Probability Mass Function)
    input_image = permute(image,[2 1 3]);
    jointPMF = zeros(256,256);
    for k=0:numel(input_image)/2-1
        jointPMF(input_image(2*k+1)+1,input_image(2*k+2)+1)=jointPMF(input_image(2*k+1)+1,input_image(2*k+2)+1)+1;
    end
    pmf =  jointPMF(:);
    pmf =   pmf/sum(pmf);
end

function H = calc_entropy(pmf)
%  Input         : pmf   (Probability Mass Function)
%
%  Output        : H     (Entropy in bits)
    H = -sum(pmf.*log2(pmf),'omitnan');
end