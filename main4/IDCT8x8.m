function block = IDCT8x8(coeff)
%  Function Name : IDCT8x8.m
%  Input         : coeff (DCT Coefficients) 8*8*3
%  Output        : block (original image block) 8*8*3
    block = zeros(size(coeff));
    for i = 1 : size(coeff,3)
        block(:,:,i) = idct2(coeff(:,:,i));
    end
end