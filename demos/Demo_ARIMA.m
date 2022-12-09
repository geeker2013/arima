%% 使用Fun_ARIMA_Forecast函数实现预测
% 
close all
clear all
addpath ../funs %将funs文件夹添加进路径，如果已经执行install_funs，则该行代码可以删除
%1.导入数据
load Data_EquityIdx   %纳斯达克综合指数
data = DataTable.NASDAQ(1:100); %如果要替换数据，将此处data替换即可。
%2.调用函数
step = 10; %预测步数
p_max = 5; %p上限
q_max = 5; %q上限
figflag = 'on'; %画图标志位，'on'为画图，'off'为不画
[forData1,lower1,upper1,aicorbic,res] = Fun_ARIMA_Forecast(data,step,p_max,q_max,figflag,'aic'); %该函数的参数信息请在函数文件内查看
%  使用ARIMA进行预测的函数（使用n阶差分、不使用对数），可以直接调用，差分阶数自动确定。
% 输入：
% data为待预测数据，一维数据，最小11个数据。但是数据长度处于11~15时依旧可能出现报错的情况。
% step为拟预测步数
% max_ar 为最大p值
% max_ma 为最大q值
% figflag 为画图标志位，'on'为画图，'off'为不画
% criterion 为定阶准则，'aic'/'bic'/'aic+bic'三种选择，此变量可以不输入或者输入为[]，此时将使用默认'aic+bic'准则
% 输出：
% forData为预测结果，其长度等于step
% lower为预测结果的95%置信下限值
% upper为预测结果的95%置信上限值
% aicorbic为aic+bic在所有pq组合下的值，返回值为矩阵，
% 例如当设置max_ar= 1，max_ma=2时
% 则返回矩阵为： a b c
%               d e f
% a代表 p=0,q=0时的aic+bic的值，b代表p=0,q=1时的值，d代表p=1,q=0时的值，以此类推
% 矩阵中可能会存在NaN值，比如a必定为NaN，当p、q的阶数较高时，也会因为软件无法正确估计参数而产生NaN值
% res：拟合残差值