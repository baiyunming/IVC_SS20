%load data
load('distortion_stillImage.mat')
load('distortion_video.mat')
load('rate_stillImageCodec.mat')
load('rate_video.mat')
load('distortion_video_halfpel');
load('rate_video_halfpel');
load('distortion_DPCM');
load('Rate_DPCM');
load('distortion_LS_filter');
load('rate_video_LS_filter');

plot(Rate_stillImage,distortion_stillImage,'b');
hold on;
plot(Rate_video,distortion_video,'r');
hold on;
plot(Rate_DPCM,distortion_DPCM,'k');
hold on;
plot(Rate_video_halfpel,distortion_video_halfpel,'g');
hold on;
plot(rate_video_LS_filter,distortion_LS_filter,'m');



xlim([0.2 4]);
title('RD performance of optimization, ForemanSequence');
legend('Still-Imgae-Codec','Video-Codec','DPCM','HalfPel-ME','OptimalFilter');
xlabel('Rate(bpp)');
ylabel('PSNR(dB)');
