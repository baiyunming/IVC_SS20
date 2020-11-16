function zze = ZeroRunEnc_EoB(zz, EOB)
n =length(zz);
zero_count=0;
zze = [];
i =1;

while (i<=n)    
   if zz(i) == 0 && zero_count~=0
       zze = [zze(1:numel(zze)-1),zero_count];
       zero_count = zero_count+1;
   elseif zz(i) == 0 && zero_count==0
        zze = [zze 0 0];
        zero_count = zero_count+1;
    
   elseif zz(i)~=0
       zze = [zze zz(i)];
       zero_count = 0;
   end
   
    if mod(i,64)==0 && zz(i)==0
    zze = [zze(1:numel(zze)-2),EOB];
    zero_count = 0;
    end
    i = i+1;
end
end

