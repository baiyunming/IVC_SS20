Final Optimization IVC Lab 
Yunming Bai 03722553

In the code folder there are 6 matlab.m files, including StillImageCodec (Chapter 4), MotionEstimation-based-VideoCodec (Chapter 5)
and 3 optimization codecs: HalfPel-MotionEstimationCodec, DPCMCodec and Optimal_filterCodec and Compare_Optimization.m is be used to plot the final RD curve. 
In addition to the .m files, a matlab data 'optimal_filter.mat' can be found, which contains the precomputed optimal In-Loop filters.   

The procedure to get the final RD curve is as follows:

(1) run Still_Image_Codec.m, you will get 2 MATLAB Data: 'rate_stillImageCodec.mat' and 'distortion_stillImage.mat', which contain the rate and distortion of StillImageCodec (Chapter 4). 
runtime ca. 410s

(2) run videoCodec.m, you will get 2 MATLAB Data: 'rate_video.mat' and 'distortion_video.mat', which contain the rate and distortion of MotionEstimation-based-VideoCodec (Chapter 5). 
runtime: ca. 280s

(3)run video_HalfpelME_Codec.m, you will get 2 MATLAB Data: 'rate_video_halfpel.mat' and 'distortion_video_halfpel.mat', which contain the rate and distortion of HalfPel-MotionEstimationCodec. 
runtime: ca. 770s

(4) run videoDPCM_Codec, you will get 2 MATLAB Data: 'Rate_DPCM.mat' and 'distortion_DPCM.mat',which contain the rate and distortion of DPCMCodec. 
runtime: ca. 900s

(5) videoCodec_Optimal_filter:  you will get 2 MATLAB Data:  'rate_video_LS_filter.mat' and 'distortion_LS_filter.mat', which contain the rate and distortion of In-loop-filter-Codec. 
note: The traning phase of the optimal In-loop filter takes several hours. Therefore, the pretained filters are stored in 'optimal_filter.mat' and directly applied in script 'videoCodec_Optimal_filter.m'.
Several modifications to the script can be made to get the version of traning the optimal filters, which habe been commented in the script. 
runtime: ca. 320s

(6) run 'Compare_Optimization.m', you will get the RD-curve 