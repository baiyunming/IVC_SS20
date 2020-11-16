function dst = IntraEncode(image, qScale)
%  Function Name : IntraEncode.m
%  Input         : image (Original RGB Image)
%                  qScale(quantization scale)
%  Output        : dst   (sequences after zero-run encoding, 1xN)
        YCbCr_image = ictRGB2YCbCr(image);
        dct_coeff = blockproc(YCbCr_image,[8 8],@(block_struct) DCT8x8 (block_struct.data));
        quantize_dct = blockproc(dct_coeff,[8 8],@(block_struct) Quant8x8 (block_struct.data,qScale));
        zig_zag = blockproc(quantize_dct,[8 8],@(block_struct) ZigZag8x8 (block_struct.data));
        EOB = 1000;
        blocks = mat2cell(zig_zag, repmat(64,[1 size(zig_zag,1)/64]), repmat(3,[1 size(zig_zag,2)/3]));
        column_zigzag =  cellfun(@(x) reshape(x,[192,1]), blocks, 'UniformOutput',false);    
        dst_cell = cellfun(@(x) ZeroRunEnc_EoB(x, EOB)', column_zigzag, 'UniformOutput',false);       
        dst = dst_cell(:);
        dst = cell2mat(dst);
end
