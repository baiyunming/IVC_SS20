% general script, e.g. image loading, function calls, etc.
clc;
clear all;
close all;

epsilon          = 0.001;

imageLena_small  = double(imread('lena_small.tif'));
[qImageLena_small, clusters_small] = LloydMax(imageLena_small, 3, epsilon);
recImage_small   = InvLloydMax(qImageLena_small, clusters_small);
PSNR_small       = calcPSNR(imageLena_small, recImage_small);

imageLena  = double(imread('lena.tif'));
[qImageLena, clusters] = LloydMax(imageLena, 3, epsilon);
recImageLena   = InvLloydMax(qImageLena, clusters);
PSNR       = calcPSNR(imageLena, recImageLena);

