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
%numhiddenunits4=300;

 
numclasses=numel(unique(Ytrain));

layersLSTM = [ ...
    sequenceInputLayer(InputSize)
    lstmLayer(numhiddenunits1,'OutputMode','sequence')
    dropoutLayer(0.5)
    lstmLayer(numhiddenunits2,'OutputMode','sequence')
    dropoutLayer(0.5)
    lstmLayer(numhiddenunits3,'OutputMode','sequence')
    dropoutLayer(0.5)
    fullyConnectedLayer(numclasses)
    softmaxLayer
    classificationLayer];

maxEpochs =300;
miniBatchSize = 50;
options = trainingOptions('adam', ...
    'InitialLearnRate',0.0035,...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','every-epoch',...
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
M=10;
p={};
rng('default');
for i=1:10
    [train, test]=crossvalind('LeaveMOut',N, ceil(N/M));
    net{i} = trainNetwork(X(train),Y(train),layers,options);
    p{i}=classify(net{i},X(test));
    yyy{i}=Y(test);
    testAccuracy{i} = sum(p{i}==yyy{i})/numel(p{i})*100;
    pp=cellstr(p{i});
    classperf(cp,pp,test);


end
 p=[p{:}];
 pppp=reshape(p,[3350,1]);
  yyy=[yyy{:}];
 yyyy=reshape(yyy,[3350,1]);
 cm= confusionchart(yyyy,pppp);
cp.ErrorRate

CVacc=sse/100;

