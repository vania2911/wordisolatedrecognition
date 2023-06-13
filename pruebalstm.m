%%%Audio Dataset%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%555
ADS = audioDatastore("C:\Users\vanii\OneDrive\Documentos\mexican_dataset_aumented\mexican_dataset","IncludeSubfolders",true,'FileExtension','.wav','LabelSource','foldernames');
% 
% % %%%%%%perform augmentation%%%%%%%%%%%%%%%%%%
% Uncomment if you wish to perform augmentation
%   jj=numel(ADS.Files);
% 
% for i=jj
%     name=ADS.Files{i}
%     newStr = extractBetween(name,'validation/','/');
%     newStr=char(newStr);
%     oldFolder = cd(strcat('/MATLAB Drive/mexican_dataset/validation/',newStr));
% 
% end
% aug=audioDataAugmenter('NumAugmentations',3);
% 
% while hasdata(ADS)
%     [audioIn,info] = read(ADS);
%     
%     data = augment(aug,audioIn,info.SampleRate);
%     
%     [filepath,fn] = fileparts(info.FileName);
%     for i = 1:size(data,1)
%         augmentedAudio = data.Audio{i};
%         
%         % If augmentation caused an audio signal to have values outside of -1 and 1, 
%         % normalize the audio signal to avoid clipping when writing.
%         if max(abs(augmentedAudio),[],'all')>1
%             augmentedAudio = augmentedAudio/max(abs(augmentedAudio),[],'all');
%         end
%         
%         audiowrite(sprintf('%s_aug%d.wav',fn,i),augmentedAudio,info.SampleRate)
%     end
% end
% 
% 

% % %%%%%%%%%histogram%%%%%%%%%%%%%%%%%%%%%%%%%%

 lensig=zeros(numel(ADS.Files),1);
 nr=1;
 while hasdata(ADS)
     word=read(ADS);
     lensig(nr)=numel(word);
     nr=nr+1;
 end

reset(ADS)
histogram(lensig)
grid on
xlabel('Signal Length (Samples)','Interpreter','latex','FontSize',26)
ylabel('Frequency','Interpreter','latex','FontSize',26)


% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 countEachLabel(ADS)
 sf = waveletScattering('SignalLength',66000,'InvarianceScale',0.5,...
 'SamplingFrequency',48000);

% % % % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
 rng default;
 ADS=shuffle(ADS);
 %%%%%%%%%%%%%%%%%%%%%%%%%%%Split
 %%%%%%%%%%%%%%%%%%%%%%%%%%%Dataset%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % ADSnew=subset(ADS,isCommand|isUnknown);
% 
  countEachLabel(ADS)
% 
   [adsTrain,adsVal,adsTest]=splitEachLabel(ADS,0.90,0.05,0.05);
countEachLabel(adsTrain);
 countEachLabel(adsTest);
 countEachLabel(adsVal);

reduceDataset = true;
if reduceDataset
    numUniqueLabels = numel(unique(adsTrain.Labels));
    adsTrain = splitEachLabel(adsTrain,round(numel(adsTrain.Files) / numUniqueLabels /1));
    adsTest = splitEachLabel(adsTest,round(numel(adsTest.Files) / numUniqueLabels / 1));
    adsVal = splitEachLabel(adsVal,round(numel(adsVal.Files) / numUniqueLabels / 1));
end

 countEachLabel(adsTrain)
countEachLabel(adsTest)
 countEachLabel(adsVal)
% % % % % % % % % % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Xtrain=[];
s_catdstrain=transform(adsTrain,@(x)helperReadSPData(x));
while hasdata(s_catdstrain)
    smat1=read(s_catdstrain);
    Xtrain=[Xtrain smat1];
end
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Xtest=[];
s_catdstest=transform(adsTest,@(x)helperReadSPData(x));
while hasdata(s_catdstest)
    smat2=read(s_catdstest);
    Xtest=[Xtest smat2];
end
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Xval=[];
s_catdsval=transform(adsVal,@(x)helperReadSPData(x));
while hasdata(s_catdsval)
    smat3=read(s_catdsval);
    Xval=[Xval smat3];
end 

%%%%%%%%%Extract wavelet coefficients%%%%%%%%%%%%%
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Strain = sf.featureMatrix(Xtrain);
Stest = sf.featureMatrix(Xtest);
Sval = sf.featureMatrix(Xval);

%%%%%%%Coefficient matrices for%%%%%%%%%%%%%%%%%%
 %%%%%%training,testing and validation%%%%%%%%%%%
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

TrainFeatures=Strain(2:end,:,:);
TrainFeatures=squeeze(num2cell(TrainFeatures,[1 2]));
TestFeatures = Stest(2:end,:,:);
TestFeatures = squeeze(num2cell(TestFeatures, [1 2]));
ValFeatures = Sval(2:end,:,:);
ValFeatures = squeeze(num2cell(ValFeatures, [1 2]));

