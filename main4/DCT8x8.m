function coeff = DCT8x8(block)
%  Input         : block    (Original Image block, 8x8x3)
%
%  Output        : coeff    (DCT coefficients after transformation, 8x8x3)
    coeff = zeros(size(block));
    for i = 1 : size(block,3)
        coeff(:,:,i) = dct2(block(:,:,i));
    end
    
end