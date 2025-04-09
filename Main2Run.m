
clc;
clear;
close all

[FileName,PathName] = uigetfile('*.jpg;*.png;*.bmp','Pick an Image');
% image reading
P = imresize(imread([PathName,FileName]),[300 300]);
  
figure;
imshow(P);title(' Input Image');


%% Adaptive and multiscale morphological gradient reconstruction
se_start=3;
max_itr=50;
min_impro=0.0001;
%% step1 gaussian filtering
sigma=1.0;gausFilter=fspecial('gaussian',[5 5],sigma);g=imfilter(P,gausFilter,'replicate');
figure;imshow(g);title('Filtered Image')

%% step2 compute gradient image
gg=colorspace('Lab<-RGB',g); 
figure;imshow(gg);title('Sim color space')
a1=sgrad_edge(normalized(gg(:,:,1))).^2;b1=sgrad_edge(abs(normalized(gg(:,:,2)))).^2;c1=sgrad_edge(normalized(gg(:,:,3))).^2;
ngrad_f1=sqrt(a1+b1+c1); 
%% step3 MMGR
f_g=zeros(size(P,1),size(P,2));diff=zeros(max_itr,1);
for i=1:max_itr
    gx=MorphologicalReconstruction(ngrad_f1,strel('disk',i+se_start-1)); 
    f_g2=max(f_g,double(gx));
    f_g1=f_g;f_g=f_g2;
    diff(i)=mean2(abs(f_g1 - f_g2));
	if(i > 1)
		if diff(i) < min_impro, break; end
    end  
end
figure;imshow(f_g,[]);title('RGB Variants of image')

%% step4  watershed
L_seg=watershed(f_g);

% Stage 1
L1=L_seg;
figure;imshow(L1,[]);title('Stage 1: Low Representation')
% Stage 2-dilation
L2=imdilate(L1,strel('square',2));

figure;imshow(L2,[]);title('Stage 2: High Representation')


[~,~,Label_n,centerLab]=ClusterCenterPixels(P,L2);


cluster_n=4;

data_n = size(centerLab, 1); %the row of input matrix
% Change the following to set default options
default_options = [2;	% exponent for the partition matrix U
    50;	% max. number of iteration
    1e-5;	% min. amount of improvement
    1];	% info display during iteration
    options = default_options;
expo = options(1);		% Exponent for U
max_iter = options(2);		% Max. iteration
min_impro = options(3);		% Min. improvement
display = options(4);		% Display info or not
iter_n=0; % actual number of iteration
U = initfcm(cluster_n, data_n);			% Initial  partition
Num=ones(cluster_n,1)*Label_n';
for i = 1:max_iter
    mf = Num.*U.^expo;       % MF matrix after exponential modification
    center = mf*centerLab./((ones(size(centerLab, 2), 1)*sum(mf'))'); % new center
    out = zeros(size(center, 1), size(centerLab, 1));
    if size(center, 2) > 1
        for k = 1:size(center, 1)
            out(k, :) = sqrt(sum(((centerLab-ones(size(centerLab, 1), 1)*center(k, :)).^2)'));
        end
    else	% 1-D data
        for k = 1:size(center, 1)
            out(k, :) = abs(center(k)-centerLab)';
        end
    end
    dist=out+eps;
    tmp = dist.^(-2/(expo-1));
    U = tmp./(ones(cluster_n, 1)*sum(tmp)+eps);
    Uc{i}=U;
    if i> 1
        if abs(max(max(Uc{i} - Uc{i-1}))) < min_impro, break; end
    end
end
iter_n = i;
center_Lab=center;
[~,IDX2]=max(U);
%%
Lr2=zeros(size(L2,1),size(L2,2));
for i=1:max(L2(:))
    Lr2=Lr2+(L2==i)*IDX2(i);
end

Lseg=ClusterCenterPixels(P,Lr2);
figure,imshow(Lseg);title('Final Segmented image of colors');


% slicing of each label
Im1=P.*(uint8((Lr2==1)));
Im2=P.*(uint8((Lr2==2)));
Im3=P.*(uint8((Lr2==3)));
Im4=P.*(uint8((Lr2==4)));

figure,
subplot(221);imshow(Im1);title('Map 1');
subplot(222);imshow(Im2);title('Map 2');
subplot(223);imshow(Im3);title('Map 3');
subplot(224);imshow(Im4);title('Map 4');

% Testing data formation
XTest1=imresize(Im1,[28 28]);
XTest2=imresize(Im2,[28 28]);
XTest3=imresize(Im3,[28 28]);
XTest4=imresize(Im4,[28 28]);


load ViT_TrainData

%% Encoder block 1              

% define network architecture
Enc_layers1 = [
    imageInputLayer([28 28 3], 'Name', 'input')
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool1')
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool2')
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn3')
    selfAttentionLayer(64,'self_attention');
    fullyConnectedLayer(8, 'Name', 'fc')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')];
% lgraph = layerGraph(layers);
% set training options
options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.01, ...
    'MaxEpochs', 40, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ...
    'Plots', 'training-progress');
% training the network
Encode_ViT1 = trainNetwork(XTrain1,categorical(YTrain1), Enc_layers1, options);

Class1=round(double(predict(Encode_ViT1,XTest1)));
%% Encoder block 2

% define network architecture
Enc_layers2 = [
    imageInputLayer([28 28 3], 'Name', 'input')
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool1')
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool2')
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn3')
    selfAttentionLayer(64,'self_attention');
    fullyConnectedLayer(8, 'Name', 'fc')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')];
% lgraph = layerGraph(layers);
% set training options
options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.01, ...
    'MaxEpochs', 35, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'VerboseFrequency', 1, ...
    'Plots', 'training-progress');
% training the network
Encode_ViT2 = trainNetwork(XTrain2,categorical(YTrain2), Enc_layers2, options);

Class2=round(double(predict(Encode_ViT2,XTest2)));

%% Encoder block 3

% define network architecture
Enc_layers3 = [
    imageInputLayer([28 28 3], 'Name', 'input')
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool1')
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool2')
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn3')
    selfAttentionLayer(64,'self_attention');
    fullyConnectedLayer(8, 'Name', 'fc')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')];
% lgraph = layerGraph(layers);
% set training options
options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.01, ...
    'MaxEpochs', 35, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'VerboseFrequency', 1, ...
    'Plots', 'training-progress');
% training the network
Encode_ViT3 = trainNetwork(XTrain3,categorical(YTrain3), Enc_layers3, options);

Class3=round(double(predict(Encode_ViT3,XTest3)));

%% Encoder block 4

% define network architecture
Enc_layers4 = [
    imageInputLayer([28 28 3], 'Name', 'input')
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool1')
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool2')
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn3')
    selfAttentionLayer(64,'self_attention');
    fullyConnectedLayer(8, 'Name', 'fc')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')];
% lgraph = layerGraph(layers);
% set training options
options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.01, ...
    'MaxEpochs', 35, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'VerboseFrequency', 1, ...
    'Plots', 'training-progress');
% training the network
Encode_ViT4 = trainNetwork(XTrain4,categorical(YTrain4), Enc_layers4, options);

Class4=round(double(predict(Encode_ViT4,XTest4)));
[~,mind1]=max(Class1);
[~,mind2]=max(Class2);
[~,mind3]=max(Class3);
[~,mind4]=max(Class4);
AllClassLabels=[mind1 mind2 mind3 mind4];
if nnz(ismember(AllClassLabels,8))>=1
    OutCategory='Heavy Rainfall';
elseif nnz(ismember(AllClassLabels,7))>=1
    OutCategory='Landslide Disaster';
elseif nnz(ismember(AllClassLabels,6))>=1
    OutCategory='Fire Accident';
elseif nnz(ismember(AllClassLabels,5))>=1
    OutCategory='Natural Resource Theft';
elseif nnz(ismember(AllClassLabels,4))>=1
    OutCategory='Residential Occupation';
elseif nnz(ismember(AllClassLabels,2))>=1
    OutCategory='Trees Cutting';
else
    OutCategory='Normal';
end

figure;
imshow(P);

PredictedClass=double(predict(Encode_ViT1,XTrain1));
[c_matrixp,Resultperf,PredictedClass]= confusionTable.getMatrix(YTrain1,PredictedClass,1);
% 
Accuracy=Resultperf.Accuracy;
Error=Resultperf.Error;
Specificity=Resultperf.Specificity;
Recall=Resultperf.Recall;
Precision=Resultperf.Precision;
FalsePositiveRate=Resultperf.FalsePositiveRate;
F1_score=Resultperf.F1_score;
Corr=Resultperf.MatthewsCorrelationCoefficient;
Kappa=Resultperf.Kappa;
TN=Resultperf.TN;
TP=Resultperf.TP;
FN=Resultperf.FN;
FP=Resultperf.FP;

% all metrics
Proposed_ViT_net=[Accuracy,Precision,Recall,F1_score,Corr,Specificity];

TrainRe=de2bi(2.^(double(YTrain1)-1))';
TestRe=de2bi(2.^(double(PredictedClass)-1))';



figure;
bar(Proposed_ViT_net(1:end-1),0.5);
grid on
xticklabels({'Accuracy','Precision','Recall','F-Measure','Correaltion','Specificity'});
xtickangle(45)
ylim([0 1]);
title('Performance metrics');

