
function dst = IntraDecode(src, img_size , qScale)
    EOB = 1000;
    rec_code = ZeroRunDec_EoB(src(:), EOB)';
    block = mat2cell(rec_code, repmat(192,[1 size(rec_code,1)/192]), [1]);
    %192x1 rehsapr 64x3
    reshaped_rec_code = cellfun(@(x) reshape(x,[64,3]), block, 'UniformOutput',false); 
    
    reshaped_cell = reshape(reshaped_rec_code, img_size(1)/8, img_size(2)/8);
    
    reshaped_rec = cell2mat(reshaped_cell);
    
    %reshaped_rec = reshape(reshaped_rec, 64*img_size(1)/8,3*img_size(2)/8);
    inv_zig_zag = blockproc(reshaped_rec,[64 3],@(block_struct) DeZigZag8x8 (block_struct.data));
    inv_quantization = blockproc(inv_zig_zag,[8 8],@(block_struct) DeQuant8x8 (block_struct.data,qScale));
    inv_dct = blockproc(inv_quantization,[8 8],@(block_struct) IDCT8x8 (block_struct.data));
    dst = ictYCbCr2RGB(inv_dct);
    
end

