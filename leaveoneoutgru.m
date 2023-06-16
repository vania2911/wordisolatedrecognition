X=[TrainFeatures;TestFeatures;ValFeatures];
Y=[Ytrain;Ytest;Yval];
N=length(X);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[InputSize,~]=size(TrainFeatures{1});
 Ytrain=adsTrain.Labels;
 Yval=adsVal.Labels;
 Ytest=adsTest.Labels;

 numhiddenunits1=500;
 numhiddenunits2=300;
numhiddenunits3=200;
%numhiddenunits4=200;


 
numclasses=numel(unique(Ytrain));

layersGRU = [ ...
    sequenceInputLayer(InputSize)
    gruLayer(numhiddenunits1,'OutputMode','sequence')
    dropoutLayer(0.5)
    gruLayer(numhiddenunits2,'OutputMode','sequence')
    dropoutLayer(0.5)
   gruLayer(numhiddenunits3,'OutputMode','last')
    dropoutLayer(0.5)
    %gruLayer(numhiddenunits4,'OutputMode','last')
    %dropoutLayer(0.3)
fullyConnectedLayer(numclasses)
softmaxLayer
classificationLayer];

maxEpochs =300;
miniBatchSize = 50;
%validationFrequency = floor(numel(numclasses)/miniBatchSize);
options = trainingOptions('adam', ...
    'InitialLearnRate',0.0035,...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'Verbose', false, ...
    'Plots','training-progress');

cp=classperf(C);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sse=0;
pred={};
net={};
tr={};
testAccuracy={};
yyy={};
p={};
M=10;
tic
for i=1:10
    i
    [train, test]=crossvalind('LeaveMOut',N, ceil(N/M));
    net{i} = trainNetwork(X(train),Y(train),layersGRU,options);
    p{i}=classify(net{i},X(test));
    yyy{i}=Y(test);
    testAccuracy{i} = sum(p{i}==yyy{i})/numel(p{i})*100;
    pp=cellstr(p{i});
    classperf(cp,pp,test);
end
cp.ErrorRate
p=[p{:}];
 pppp=reshape(p,[335*i,1]);
  yyy=[yyy{:}];
 yyyy=reshape(yyy,[335*i,1]);
 cm= confusionchart(yyyy,pppp);



toc
