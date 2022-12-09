%% 使用Fun_ARIMA_Forecast函数实现预测
% 单步预测
close all
clear all
addpath ../funs %将funs文件夹添加进路径，如果已经执行install_funs，则该行代码可以删除

% 1.导入数据
load Data_EquityIdx   %纳斯达克综合指数
len = 2000;
data = DataTable.NASDAQ(1:len); %如果要替换数据，将此处data替换即可。
%2.调用函数并实现单步预测
TrainR = 0.8; %用于训练的数据长度，也可以写成0.7（作为比例值）
max_ar = 3;  %p上限值
max_ma = 3;  %q上限值
[forData,dataTrain,dataTest,aicorbic,res] = Fun_ARIMA_Forecast_MulComp(data,TrainR,max_ar,max_ma,'on','aic');