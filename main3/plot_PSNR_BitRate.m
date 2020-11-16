RGB2Grayscale_bit_rate = 8;
RGB2Grayscale_PSNR = 15.46;

RGB_Subsampling_bit_rate = 6;
RGB_Subsampling_PSNR = 33.6768;

Chrominace_Subsampling_bit_rate = 12;
Chrominace_Subsampling_PSNR = 38.3251;

Predictive_Still_image_codec_bit_rate = 6.72;
Predictive_Still_image_codec_PSNR = 37.51;

VectorQuantization_bit_rate = 5.6744;
VectorQuantization_PSNR = 34.6496;

bit_rate = [RGB2Grayscale_bit_rate,RGB_Subsampling_bit_rate,Chrominace_Subsampling_bit_rate,Predictive_Still_image_codec_bit_rate,VectorQuantization_bit_rate];
psnr = [RGB2Grayscale_PSNR, RGB_Subsampling_PSNR, Chrominace_Subsampling_PSNR, Predictive_Still_image_codec_PSNR, VectorQuantization_PSNR];

str = ["RGB2Grayscale", "RGBSub", "ChrominaceSub", "Predictive", "VectorQuantization" ];
figure;
scatter(bit_rate,psnr,'filled');
for i  = 1:numel(bit_rate)
    text(bit_rate(i),psnr(i),str(i),'FontSize',8);
end

xlabel('Bit per pixel');
ylabel('PSNR[dB]');
xlim([0 15]);
ylim([10 50]);
