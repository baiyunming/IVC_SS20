function dst = ZeroRunDec_EoB(src, EoB)
%  Function Name : ZeroRunDec1.m zero run level decoder
%  Input         : src (zero run encoded sequence 1xM with EoB signs)
%                  EoB (end of block sign)
%
%  Output        : dst (reconstructed zig-zag scanned sequence 1xN)
n = length(src);
current_index = 1;
i = 1;

dst = [];
n = length(src);
current_index = 1;
i = 1;
while (i<=n)
    if (src(i)~=0)&&(src(i)~=EoB)
        dst(current_index) = src(i);
        current_index =numel(dst)+1;
        i = i+1;
    elseif src(i)==0
        dst(current_index:current_index+src(i+1)) = 0;
        current_index = numel(dst)+1;
        i = i+2;
    elseif src(i)==EoB
        num_pad_zero = 64 - mod(numel(dst),64);
        dst(current_index:current_index+num_pad_zero-1) = 0;
        current_index = numel(dst)+1;
        i = i+1;
    
    end
end