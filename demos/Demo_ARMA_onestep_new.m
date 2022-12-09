%% 使用Fun_ARIMA_Forecast函数实现预测
% 单步预测
close all
clear all
addpath ../funs %将funs文件夹添加进路径，如果已经执行install_funs，则该行代码可以删除

% 1.导入数据
load Data_EquityIdx   %纳斯达克综合指数
len = 140;
data = DataTable.NASDAQ(1:len); %如果要替换数据，将此处data替换即可。
%2.调用函数并实现单步预测
TrainR = 120; %用于训练的数据长度，也可以写成0.7（作为比例值）
max_ar = 3;  %p上限值
max_ma = 3;  %q上限值
[forData,dataTrain,dataTest,aicorbic,res] = Fun_ARIMA_Forecast_Onestep(data,TrainR,max_ar,max_ma,'on','aic');

% 使用ARIMA进行单步预测的函数（使用n阶差分、不使用对数），可以直接调用，差分阶数自动确定。
% 输入：
% data   为全部数据，要求为一维数据。该数据在程序中会依照TrainR的数值被进一步划分为训练集和测试集
% TrainR 为训练集比例，如果TrainR为小于1的小数，则代表比例值，如果Train为大于1的整数，则代表训练集数据点数。
% max_ar 为最大p值
% max_ma 为最大q值
% figflag 为画图标志位，'on'为画图，'off'为不画
% criterion 为定阶准则，'aic'/'bic'/'aic+bic'三种选择，此变量可以不输入或者输入为[]，此时将使用默认'aic+bic'准则
% 输出：
% forData为预测结果
% dataTrain为训练集数据
% dataTest为测试集数据
% aicorbic为aic+bic在所有pq组合下的值，返回值为矩阵，
% 例如当设置max_ar= 1，max_ma=2时
% 则返回矩阵为： a b c
%               d e f
% a代表 p=0,q=0时的aic+bic的值，b代表p=0,q=1时的值，d代表p=1,q=0时的值，以此类推
% 矩阵中可能会存在NaN值，比如a必定为NaN，当p、q的阶数较高时，也会因为软件无法正确估计参数而产生NaN值
% res：拟合残差值