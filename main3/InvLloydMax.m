function image = InvLloydMax(qImage, clusters)
%  Input         : qImage   (Quantized Image)
%                  clusters (Quantization Table)
%  Output        : image    (Recovered Image)
[w h c] = size(qImage);
image = zeros(w,h,c);
for ind = 1:numel(qImage(:))
    image(ind) = clusters(qImage(ind));   
end

end