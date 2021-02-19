clc
clear all

%% load Data

%MatrixRegressionLearner is made out of the data from both databases.
load('matrixRegressionLearner.mat')
load('data.mat')
load('BSRN_dataUpdated.mat')


%% Matrix definition (Both Databases)
rng(13)
[trainInd,~,testInd] = dividerand(length(matrixRegressionLearner),0.85,0,0.15);

%Convert latitudes to absolute value:
matrixRegressionLearner(:,1)=abs(matrixRegressionLearner(:,1));

matrixRegressionLearner_train=matrixRegressionLearner(trainInd,:);
matrixRegressionLearner_test=matrixRegressionLearner(testInd,:);

inputNNVector_train=matrixRegressionLearner(trainInd,1);
outputNNVector_train=matrixRegressionLearner(trainInd,2);
inputNNVector_test=matrixRegressionLearner(testInd,1);
outputNNVector_test=matrixRegressionLearner(testInd,2);

%% Training Polynomial models (Both Databases):

%Cambiar los parámetros con mi base de datos.
[data.regression.bothDataBases.sf3,data.regression.bothDataBases.gof3] = fit(matrixRegressionLearner_train(:,1), matrixRegressionLearner_train(:,2),'poly3');
[data.regression.bothDataBases.sf2,data.regression.bothDataBases.gof2] = fit(matrixRegressionLearner_train(:,1), matrixRegressionLearner_train(:,2),'poly2');
[data.regression.bothDataBases.sf1,data.regression.bothDataBases.gof1] = fit(matrixRegressionLearner_train(:,1), matrixRegressionLearner_train(:,2),'poly1');

%% TrainNN - Skip if already trained and loaded (Both Databases)
rng(13)

performance=[];

for i=1:20
    
x = inputNNVector_train';
t = outputNNVector_train';
[trainInd,~,testInd] = dividerand(length(x),0.85,0,0.15);
x_train=x(trainInd);
y_train=t(trainInd);

x_test=x(testInd);
y_test=t(testInd);



trainFcn = 'traingdx'; 
hiddenLayerSize = i;
net = fitnet(hiddenLayerSize,trainFcn);

net.divideParam.trainRatio = 87.25/100;
net.divideParam.valRatio = 12.75/100;
net.divideParam.testRatio = 0/100;
net.trainParam.max_fail=50;


[net,~] = train(net,x_train,y_train);


y_pred=net(x_test);
RMSE=sqrt(mean((y_test - y_pred).^2));

performance=[performance;[i, RMSE]];

end

%% Train Optimal NN (Both Databases)

%Select number of units of hidden layer equal to perfromance(:,1) related
%to min(perfromance(:,2)). Use it in the trainNN script

net=trainNN(inputNNVector_train,outputNNVector_train);


%% Test Performance (Both Databases)


yCubic=data.regression.bothDataBases.sf3(matrixRegressionLearner_test(:,1));
yQuadratic=data.regression.bothDataBases.sf2(matrixRegressionLearner_test(:,1));
yLinear=data.regression.bothDataBases.sf1(matrixRegressionLearner_test(:,1));

yNN=net(inputNNVector_test')';
yTree=trainedModel.predictFcn(matrixRegressionLearner_test(:,1));
yOptimizedTree=trainedModel1.predictFcn(matrixRegressionLearner_test(:,1));

error.bothDatabases.Bias.cubic=mean(matrixRegressionLearner_test(:,2) - yCubic);
error.bothDatabases.Bias.quadratic=mean(matrixRegressionLearner_test(:,2) - yQuadratic);
error.bothDatabases.Bias.linear=mean(matrixRegressionLearner_test(:,2) - yLinear);
error.bothDatabases.Bias.NN=mean(matrixRegressionLearner_test(:,2) - yNN);
error.bothDatabases.Bias.tree=mean(matrixRegressionLearner_test(:,2) - yTree);
error.bothDatabases.Bias.optimizedTree=mean(matrixRegressionLearner_test(:,2) - yOptimizedTree);

error.bothDatabases.MAE.cubic=mean(abs((matrixRegressionLearner_test(:,2) - yCubic)));
error.bothDatabases.MAE.quadratic=mean(abs((matrixRegressionLearner_test(:,2) - yQuadratic)));
error.bothDatabases.MAE.linear=mean(abs((matrixRegressionLearner_test(:,2) - yLinear)));
error.bothDatabases.MAE.NN=mean(abs((outputNNVector_test - yNN)));
error.bothDatabases.MAE.tree=mean(abs((matrixRegressionLearner_test(:,2) - yTree)));
error.bothDatabases.MAE.optimizedTree=mean(abs((matrixRegressionLearner_test(:,2) - yOptimizedTree)));

error.bothDatabases.RMSE.cubic=sqrt(mean((matrixRegressionLearner_test(:,2) - yCubic).^2));
error.bothDatabases.RMSE.quadratic=sqrt(mean((matrixRegressionLearner_test(:,2) - yQuadratic).^2));
error.bothDatabases.RMSE.linear=sqrt(mean((matrixRegressionLearner_test(:,2) - yLinear).^2));
error.bothDatabases.RMSE.NN=sqrt(mean((outputNNVector_test - yNN).^2));
error.bothDatabases.RMSE.tree=sqrt(mean((matrixRegressionLearner_test(:,2) - yTree).^2));
error.bothDatabases.RMSE.optimizedTree=sqrt(mean((matrixRegressionLearner_test(:,2) - yOptimizedTree).^2));

%% Plot Functions (Both Databases):

lat=0:0.1:90;
opt_tilt_NN=net(lat);
opt_tilt_yCubic=data.regression.bothDataBases.sf3(lat');
opt_tilt_yQuadratic=data.regression.bothDataBases.sf2(lat');
opt_tilt_yLinear=data.regression.bothDataBases.sf1(lat');
opt_tilt_yTree=trainedModel1.predictFcn(lat');

figure
hold on
grid on
scatter(matrixRegressionLearner(:,1),matrixRegressionLearner(:,2),'r.')
plot(lat,opt_tilt_NN,'g','LineWidth',1)
plot(lat,opt_tilt_yCubic, 'b','LineWidth',1)
plot(lat,opt_tilt_yQuadratic,'m','LineWidth',1)
plot(lat,opt_tilt_yLinear,'k--','LineWidth',1)
plot(lat,opt_tilt_yTree,'LineWidth',1)
xlabel('Latitude in absolute value (º)')
ylabel('Optimum tilt angle (º)')
legend({'BSRN & Energy+','NN','Cubic','Quadratic','Lineal','Tree'},'Location','northwest')

%% Alternative NN in Energy+:

infoEnergyPlus=[data.energyPlus.incOpt2551Sites.latitudV'];
%     data.energyPlus.incOpt2551Sites.longitudV'];
%    data.energyPlus.incOpt2551Sites.altitudV'];

matrixEnergyPlus=[infoEnergyPlus,data.energyPlus.optimum.annual.tilt];

rng(13)
[trainInd,~,testInd] = dividerand(length(matrixEnergyPlus),0.85,0,0.15);

%Convert latitudes to absolute value:
matrixEnergyPlus(:,1)=abs(matrixEnergyPlus(:,1));

matrixEnergyPlus_train=matrixEnergyPlus(trainInd,:);
matrixEnergyPlus_test=matrixEnergyPlus(testInd,:);


inputNNVector_train=matrixEnergyPlus(trainInd,1:end-1);
outputNNVector_train=matrixEnergyPlus(trainInd,end);
inputNNVector_test=matrixEnergyPlus(testInd,1:end-1);
outputNNVector_test=matrixEnergyPlus(testInd,end);

%% Training Polynomial models (Energy+):

%Cambiar los parámetros con mi base de datos.
[data.regression.energyPlus.sf3,data.regression.energyPlus.gof3] = fit(matrixEnergyPlus_train(:,1), matrixEnergyPlus_train(:,2),'poly3');
[data.regression.energyPlus.sf2,data.regression.energyPlus.gof2] = fit(matrixEnergyPlus_train(:,1), matrixEnergyPlus_train(:,2),'poly2');
[data.regression.energyPlus.sf1,data.regression.energyPlus.gof1] = fit(matrixEnergyPlus_train(:,1), matrixEnergyPlus_train(:,2),'poly1');

%% TrainNN - Skip if already trained and loaded (Energy+)
rng(13)

performance=[];

for i=1:20
    
x = inputNNVector_train';
t = outputNNVector_train';
[trainInd,~,testInd] = dividerand(length(x),0.85,0,0.15);
x_train=x(trainInd);
y_train=t(trainInd);

x_test=x(testInd);
y_test=t(testInd);



trainFcn = 'traingdx'; 
hiddenLayerSize = i;
net = fitnet(hiddenLayerSize,trainFcn);

net.divideParam.trainRatio = 87.25/100;
net.divideParam.valRatio = 12.75/100;
net.divideParam.testRatio = 0/100;
net.trainParam.max_fail=50;


[net,~] = train(net,x_train,y_train);


y_pred=net(x_test);
RMSE=sqrt(mean((y_test - y_pred).^2));

performance=[performance;[i, RMSE]];

end

%% Train Optimal NN (Energy+)

%Select number of units of hidden layer equal to perfromance(:,1) related
%to min(perfromance(:,2)). Use it in the trainNN script

net=trainNN(inputNNVector_train,outputNNVector_train);


%% Test Performance (Energy+)


yCubic=data.regression.energyPlus.sf3(matrixEnergyPlus_test(:,1));
yQuadratic=data.regression.energyPlus.sf2(matrixEnergyPlus_test(:,1));
yLinear=data.regression.energyPlus.sf1(matrixEnergyPlus_test(:,1));

yNN=net(inputNNVector_test')';
yTree=trainedModel.predictFcn(matrixEnergyPlus_test(:,1));
yOptimizedTree=trainedModel1.predictFcn(matrixEnergyPlus_test(:,1));

error.energyPlus.Bias.cubic=mean(matrixEnergyPlus_test(:,2) - yCubic);
error.energyPlus.Bias.quadratic=mean(matrixEnergyPlus_test(:,2) - yQuadratic);
error.energyPlus.Bias.linear=mean(matrixEnergyPlus_test(:,2) - yLinear);
error.energyPlus.Bias.NN=mean(matrixEnergyPlus_test(:,2) - yNN);
error.energyPlus.Bias.tree=mean(matrixEnergyPlus_test(:,2) - yTree);
error.energyPlus.Bias.optimizedTree=mean(matrixEnergyPlus_test(:,2) - yOptimizedTree);

error.energyPlus.MAE.cubic=mean(abs((matrixEnergyPlus_test(:,2) - yCubic)));
error.energyPlus.MAE.quadratic=mean(abs((matrixEnergyPlus_test(:,2) - yQuadratic)));
error.energyPlus.MAE.linear=mean(abs((matrixEnergyPlus_test(:,2) - yLinear)));
error.energyPlus.MAE.NN=mean(abs((outputNNVector_test - yNN)));
error.energyPlus.MAE.tree=mean(abs((matrixEnergyPlus_test(:,2) - yTree)));
error.energyPlus.MAE.optimizedTree=mean(abs((matrixEnergyPlus_test(:,2) - yOptimizedTree)));

error.energyPlus.RMSE.cubic=sqrt(mean((matrixEnergyPlus_test(:,2) - yCubic).^2));
error.energyPlus.RMSE.quadratic=sqrt(mean((matrixEnergyPlus_test(:,2) - yQuadratic).^2));
error.energyPlus.RMSE.linear=sqrt(mean((matrixEnergyPlus_test(:,2) - yLinear).^2));
error.energyPlus.RMSE.NN=sqrt(mean((outputNNVector_test - yNN).^2));
error.energyPlus.RMSE.tree=sqrt(mean((matrixEnergyPlus_test(:,2) - yTree).^2));
error.energyPlus.RMSE.optimizedTree=sqrt(mean((matrixEnergyPlus_test(:,2) - yOptimizedTree).^2));

%% Plot Functions (Energy+):

lat=0:0.1:90;
% opt_tilt_NN=net(lat);
opt_tilt_yCubic=data.regression.energyPlus.sf3(lat');
opt_tilt_yQuadratic=data.regression.energyPlus.sf2(lat');
opt_tilt_yLinear=data.regression.energyPlus.sf1(lat');
% opt_tilt_yTree=trainedModel1.predictFcn(lat');

figure
hold on

% plot(lat,opt_tilt_NN,'g','LineWidth',1)
plot(lat,opt_tilt_yCubic, 'b','LineWidth',1)
plot(lat,opt_tilt_yQuadratic,'m','LineWidth',1)
plot(lat,opt_tilt_yLinear,'k--','LineWidth',1)
scatter(matrixEnergyPlus(:,1),matrixEnergyPlus(:,2),20,'r.')
% plot(lat,opt_tilt_yTree,'LineWidth',1)
xlabel('Latitude in absolute value (º)')
ylabel('Optimum tilt angle (º)')
legend({'Cubic','Quadratic','Lineal','Energy+ sites'},'Location','northwest')
grid on

%% Alternative NN in BSRN:

matrixBSRN=table2array(BSRN_dataUpdated);

%To eliminate Longitude, uncomment the following line:
matrixBSRN(:,2)=[];

rng(13)
[trainInd,~,testInd] = dividerand(length(matrixBSRN),0.85,0,0.15);

%Convert latitudes to absolute value:
matrixBSRN(:,1)=abs(matrixBSRN(:,1));

matrixBSRN_train=matrixBSRN(trainInd,:);
matrixBSRN_test=matrixBSRN(testInd,:);


inputNNVector_train=matrixBSRN(trainInd,1:end-1);
outputNNVector_train=matrixBSRN(trainInd,end);
inputNNVector_test=matrixBSRN(testInd,1:end-1);
outputNNVector_test=matrixBSRN(testInd,end);

%% Training Polynomial models (BSRN):

%Cambiar los parámetros con mi base de datos.
[data.regression.BSRN.sf3,data.regression.BSRN.gof3] = fit(matrixBSRN_train(:,1), matrixBSRN_train(:,2),'poly3');
[data.regression.BSRN.sf2,data.regression.BSRN.gof2] = fit(matrixBSRN_train(:,1), matrixBSRN_train(:,2),'poly2');
[data.regression.BSRN.sf1,data.regression.BSRN.gof1] = fit(matrixBSRN_train(:,1), matrixBSRN_train(:,2),'poly1');

%% TrainNN - Skip if already trained and loaded (BSRN)
rng(13)

performance=[];

for i=1:20
    
x = inputNNVector_train';
t = outputNNVector_train';
[trainInd,~,testInd] = dividerand(length(x),0.85,0,0.15);
x_train=x(trainInd);
y_train=t(trainInd);

x_test=x(testInd);
y_test=t(testInd);



trainFcn = 'traingdx'; 
hiddenLayerSize = i;
net = fitnet(hiddenLayerSize,trainFcn);

net.divideParam.trainRatio = 87.25/100;
net.divideParam.valRatio = 12.75/100;
net.divideParam.testRatio = 0/100;
net.trainParam.max_fail=50;


[net,~] = train(net,x_train,y_train);


y_pred=net(x_test);
RMSE=sqrt(mean((y_test - y_pred).^2));

performance=[performance;[i, RMSE]];

end

%% Train Optimal NN (BSRN)

%Select number of units of hidden layer equal to perfromance(:,1) related
%to min(perfromance(:,2)). Use it in the trainNN script

net=trainNN(inputNNVector_train,outputNNVector_train);


%% Test Performance (BSRN)


yCubic=data.regression.BSRN.sf3(matrixBSRN_test(:,1));
yQuadratic=data.regression.BSRN.sf2(matrixBSRN_test(:,1));
yLinear=data.regression.BSRN.sf1(matrixBSRN_test(:,1));

yNN=net(inputNNVector_test')';
yTree=trainedModel.predictFcn(matrixBSRN_test(:,1));
yOptimizedTree=trainedModel1.predictFcn(matrixBSRN_test(:,1));

error.BSRN.Bias.cubic=mean(matrixBSRN_test(:,2) - yCubic);
error.BSRN.Bias.quadratic=mean(matrixBSRN_test(:,2) - yQuadratic);
error.BSRN.Bias.linear=mean(matrixBSRN_test(:,2) - yLinear);
error.BSRN.Bias.NN=mean(matrixBSRN_test(:,2) - yNN);
error.BSRN.Bias.tree=mean(matrixBSRN_test(:,2) - yTree);
error.BSRN.Bias.optimizedTree=mean(matrixBSRN_test(:,2) - yOptimizedTree);

error.BSRN.MAE.cubic=mean(abs((matrixBSRN_test(:,2) - yCubic)));
error.BSRN.MAE.quadratic=mean(abs((matrixBSRN_test(:,2) - yQuadratic)));
error.BSRN.MAE.linear=mean(abs((matrixBSRN_test(:,2) - yLinear)));
error.BSRN.MAE.NN=mean(abs((outputNNVector_test - yNN)));
error.BSRN.MAE.tree=mean(abs((matrixBSRN_test(:,2) - yTree)));
error.BSRN.MAE.optimizedTree=mean(abs((matrixBSRN_test(:,2) - yOptimizedTree)));

error.BSRN.RMSE.cubic=sqrt(mean((matrixBSRN_test(:,2) - yCubic).^2));
error.BSRN.RMSE.quadratic=sqrt(mean((matrixBSRN_test(:,2) - yQuadratic).^2));
error.BSRN.RMSE.linear=sqrt(mean((matrixBSRN_test(:,2) - yLinear).^2));
error.BSRN.RMSE.NN=sqrt(mean((outputNNVector_test - yNN).^2));
error.BSRN.RMSE.tree=sqrt(mean((matrixBSRN_test(:,2) - yTree).^2));
error.BSRN.RMSE.optimizedTree=sqrt(mean((matrixBSRN_test(:,2) - yOptimizedTree).^2));

%% Plot Functions (BSRN):

lat=0:0.1:90;
% opt_tilt_NN=net(lat);
opt_tilt_yCubic=data.regression.BSRN.sf3(lat');
opt_tilt_yQuadratic=data.regression.BSRN.sf2(lat');
opt_tilt_yLinear=data.regression.BSRN.sf1(lat');
% opt_tilt_yTree=trainedModel1.predictFcn(lat');

figure
hold on
grid on

% plot(lat,opt_tilt_NN,'g','LineWidth',1)
plot(lat,opt_tilt_yCubic, 'b','LineWidth',1)
plot(lat,opt_tilt_yQuadratic,'m','LineWidth',1)
plot(lat,opt_tilt_yLinear,'k--','LineWidth',1)
% plot(lat,opt_tilt_yTree,'LineWidth',1)
scatter(matrixBSRN(:,1),matrixBSRN(:,2),'r*')
xlabel('Latitude in absolute value (º)')
ylabel('Optimum tilt angle (º)')
legend({'Cubic','Quadratic','Lineal','BSRN sites'},'Location','northwest')


%% Incorporate Longitude in Both Databases:

infoEnergyPlus=[data.energyPlus.incOpt2551Sites.latitudV',...
    data.energyPlus.incOpt2551Sites.longitudV'];
%    data.energyPlus.incOpt2551Sites.altitudV'];

matrixLongitudeBothDB=[infoEnergyPlus,data.energyPlus.optimum.annual.tilt;...
    BSRN_dataUpdated.Latitude,BSRN_dataUpdated.Longitude,BSRN_dataUpdated.Tilt_Opt_anual_corregido_OK];

rng(13)
[trainInd,~,testInd] = dividerand(length(matrixLongitudeBothDB),0.85,0,0.15);

%Convert latitudes to absolute value:
matrixLongitudeBothDB(:,1)=abs(matrixLongitudeBothDB(:,1));

matrixLongitudeBothDB_train=matrixLongitudeBothDB(trainInd,:);
matrixLongitudeBothDB_test=matrixLongitudeBothDB(testInd,:);


inputNNVector_train=matrixLongitudeBothDB(trainInd,1:end-1);
outputNNVector_train=matrixLongitudeBothDB(trainInd,end);
inputNNVector_test=matrixLongitudeBothDB(testInd,1:end-1);
outputNNVector_test=matrixLongitudeBothDB(testInd,end);

%% Training Polynomial models (Energy+):

%Cambiar los parámetros con mi base de datos.
[data.regression.bothDB_longitude.sf3,data.regression.bothDB_longitude.gof3] = fit(...
    [matrixLongitudeBothDB_train(:,1), matrixLongitudeBothDB_train(:,2)],matrixLongitudeBothDB_train(:,3),'poly33');
[data.regression.bothDB_longitude.sf2,data.regression.bothDB_longitude.gof2] = fit(...
    [matrixLongitudeBothDB_train(:,1), matrixLongitudeBothDB_train(:,2)],matrixLongitudeBothDB_train(:,3),'poly22');
[data.regression.bothDB_longitude.sf1,data.regression.bothDB_longitude.gof1] = fit(...
    [matrixLongitudeBothDB_train(:,1), matrixLongitudeBothDB_train(:,2)],matrixLongitudeBothDB_train(:,3),'poly11');

%% TrainNN - Skip if already trained and loaded (Energy+)
rng(13)

performance=[];

for i=1:20
    
x = inputNNVector_train';
t = outputNNVector_train';
[trainInd,~,testInd] = dividerand(length(x),0.85,0,0.15);
x_train=x(trainInd);
y_train=t(trainInd);

x_test=x(testInd);
y_test=t(testInd);



trainFcn = 'traingdx'; 
hiddenLayerSize = i;
net = fitnet(hiddenLayerSize,trainFcn);

net.divideParam.trainRatio = 87.25/100;
net.divideParam.valRatio = 12.75/100;
net.divideParam.testRatio = 0/100;
net.trainParam.max_fail=50;


[net,~] = train(net,x_train,y_train);


y_pred=net(x_test);
RMSE=sqrt(mean((y_test - y_pred).^2));

performance=[performance;[i, RMSE]];

end

%% Train Optimal NN (Energy+)

%Select number of units of hidden layer equal to perfromance(:,1) related
%to min(perfromance(:,2)). Use it in the trainNN script

net=trainNN(inputNNVector_train,outputNNVector_train);


%% Test Performance (Energy+)


yCubic=data.regression.bothDB_longitude.sf3(inputNNVector_test);
yQuadratic=data.regression.bothDB_longitude.sf2(inputNNVector_test);
yLinear=data.regression.bothDB_longitude.sf1(inputNNVector_test);

yNN=net(inputNNVector_test')';
yTree=trainedModel.predictFcn(inputNNVector_test);
yOptimizedTree=trainedModel1.predictFcn(inputNNVector_test);

error.bothDB_longitude.Bias.cubic=mean(outputNNVector_test - yCubic);
error.bothDB_longitude.Bias.quadratic=mean(outputNNVector_test - yQuadratic);
error.bothDB_longitude.Bias.linear=mean(outputNNVector_test - yLinear);
error.bothDB_longitude.Bias.NN=mean(outputNNVector_test - yNN);
error.bothDB_longitude.Bias.tree=mean(outputNNVector_test - yTree);
error.bothDB_longitude.Bias.optimizedTree=mean(outputNNVector_test - yOptimizedTree);

error.bothDB_longitude.MAE.cubic=mean(abs((outputNNVector_test - yCubic)));
error.bothDB_longitude.MAE.quadratic=mean(abs((outputNNVector_test - yQuadratic)));
error.bothDB_longitude.MAE.linear=mean(abs((outputNNVector_test - yLinear)));
error.bothDB_longitude.MAE.NN=mean(abs((outputNNVector_test - yNN)));
error.bothDB_longitude.MAE.tree=mean(abs((outputNNVector_test - yTree)));
error.bothDB_longitude.MAE.optimizedTree=mean(abs((outputNNVector_test - yOptimizedTree)));

error.bothDB_longitude.RMSE.cubic=sqrt(mean((outputNNVector_test - yCubic).^2));
error.bothDB_longitude.RMSE.quadratic=sqrt(mean((outputNNVector_test - yQuadratic).^2));
error.bothDB_longitude.RMSE.linear=sqrt(mean((outputNNVector_test - yLinear).^2));
error.bothDB_longitude.RMSE.NN=sqrt(mean((outputNNVector_test - yNN)));
error.bothDB_longitude.RMSE.tree=sqrt(mean((outputNNVector_test - yTree).^2));
error.bothDB_longitude.RMSE.optimizedTree=sqrt(mean((outputNNVector_test - yOptimizedTree).^2));

%% Plot Functions (Energy+):

lat=0:1:90;
long=-180:4:180;
opt_tilt_NN=[];
for i=1:length(lat)
    for j=1:length(long)
        opt_tilt_NN(j,i)=net([lat(i);long(j)]);
    end
end
% opt_tilt_NN=net([lat;long]);
% opt_tilt_yCubic=data.regression.bothDB_longitude.sf33(lat',long');
opt_tilt_yCubic=[];
for i=1:length(lat)
    for j=1:length(long)
        opt_tilt_yCubic(j,i)=data.regression.bothDB_longitude.sf3([lat(i);long(j)]');
    end
end
% opt_tilt_yQuadratic=data.regression.bothDB_longitude.sf22(lat',long');
opt_tilt_yQuadratic=[];
for i=1:length(lat)
    for j=1:length(long)
        opt_tilt_yQuadratic(j,i)=data.regression.bothDB_longitude.sf2([lat(i);long(j)]');
    end
end
% opt_tilt_yLinear=data.regression.bothDB_longitude.sf11(lat',long');
opt_tilt_yLinear=[];
for i=1:length(lat)
    for j=1:length(long)
        opt_tilt_yLinear(j,i)=data.regression.bothDB_longitude.sf1([lat(i);long(j)]');
    end
end
% opt_tilt_yTree=trainedModel1.predictFcn([lat',long']);
opt_tilt_yTree=[];
for i=1:length(lat)
    for j=1:length(long)
        opt_tilt_yTree(j,i)=trainedModel1.predictFcn([lat(i);long(j)]');
    end
end

figure
hold on
grid on
scatter3(matrixLongitudeBothDB(:,1),matrixLongitudeBothDB(:,2),matrixLongitudeBothDB(:,3),'r.')
surf(lat,long,opt_tilt_NN, 'FaceColor','g')
surf(lat,long,opt_tilt_yCubic,'FaceColor', 'b')
surf(lat,long,opt_tilt_yQuadratic,'FaceColor','m')
surf(lat,long,opt_tilt_yLinear,'FaceColor','k')
surf(lat,long,opt_tilt_yTree)
% plot(lat,opt_tilt_yCubic, 'b','LineWidth',1)
% plot(lat,opt_tilt_yQuadratic,'m','LineWidth',1)
% plot(lat,opt_tilt_yLinear,'k--','LineWidth',1)
% plot(lat,opt_tilt_yTree,'LineWidth',1)
xlabel('Latitude in absolute value (º)')
ylabel('Optimum tilt angle (º)')
legend({'Energy+','NN','Cubic','Quadratic','Lineal','Tree'},'Location','northwest')


%% Plot Functions (BSRN):

lat=0:0.1:90;

opt_tilt_yCubic_allData=data.regression.bothDataBases.sf3(lat');
opt_tilt_yQuadratic_allData=data.regression.bothDataBases.sf2(lat');

opt_tilt_yCubic_energyPlus=data.regression.energyPlus.sf3(lat');
opt_tilt_yQuadratic_energyPlus=data.regression.energyPlus.sf2(lat');

opt_tilt_yCubic_BSRN=data.regression.BSRN.sf3(lat');
opt_tilt_yQuadratic_BSRN=data.regression.BSRN.sf2(lat');


figure
hold on
grid on
plot(lat,opt_tilt_yCubic_allData, 'k','LineWidth',1)
plot(lat,opt_tilt_yCubic_energyPlus, 'b','LineWidth',1)
plot(lat,opt_tilt_yCubic_BSRN, 'b-.','LineWidth',1)

xlabel('Latitude in absolute value (º)')
ylabel('Optimum tilt angle (º)')
legend({'Cubic All-Data','Cubic Energy+','Cubic BSRN'},'Location','northwest')


figure
hold on
grid on
plot(lat,opt_tilt_yQuadratic_allData,'k','LineWidth',1)
plot(lat,opt_tilt_yQuadratic_energyPlus,'m','LineWidth',1)
plot(lat,opt_tilt_yQuadratic_BSRN,'m-.','LineWidth',1)


xlabel('Latitude in absolute value (º)')
ylabel('Optimum tilt angle (º)')
legend({'Quadratic All-Data','Quadratic Energy+','Quadratic BSRN'},'Location','northwest')

