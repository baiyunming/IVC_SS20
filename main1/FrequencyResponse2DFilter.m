
kernel = [1 2 1; 2 4 2; 1 2 1];
kernel = kernel/sum(kernel(:));
frequencyResp = fft2(kernel);
figure('name','Frequency responce of filter');
ddd = fftshift(frequencyResp);
imagesc(abs(fftshift(frequencyResp)));
%% 
w= fir1(40,0.5);
w2 = w'*w;
frequencyRespW2 = fft2(w2);
figure('name','Frequency responce of filter');
imagesc(abs(fftshift(frequencyRespW2)));

%% 
m = [1,2,3]
size(m)



