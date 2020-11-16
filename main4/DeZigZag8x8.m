function coeffs = DeZigZag8x8(zz)
%  Function Name : DeZigZag8x8.m
%  Input         : zz    (Coefficients in zig-zag order)
%
%  Output        : coeffs(DCT coefficients in original order)
    zigzag_index_table = [1 2 6 7 15 16 28 29; 3 5 8 14 17 27 30 43; 4 9 13 18 26 31 42 44; 10 12 19 25 32 41 45 54; 11 20 24 33 40 46 53 55; 21 23 34 39 47 52 56 61; 22 35 38 48 51 57 60 62; 36 37 49 50 58 59 63 64];
    zz1  = zz(:,1);
    deZigZag8x8_1= zz1( zigzag_index_table(:) );
    deZigZag8x8_1 = reshape(deZigZag8x8_1, 8, 8);
    
    zz2  = zz(:,2);
    deZigZag8x8_2= zz2( zigzag_index_table(:) );
    deZigZag8x8_2 = reshape(deZigZag8x8_2, 8, 8);
    
    zz3  = zz(:,3);
    deZigZag8x8_3= zz3( zigzag_index_table(:) );
    deZigZag8x8_3 = reshape(deZigZag8x8_3, 8, 8);
    
    
    coeffs(:,:,1) = deZigZag8x8_1;
    coeffs(:,:,2) = deZigZag8x8_2;
    coeffs(:,:,3) = deZigZag8x8_3;
    
    
    
    
end