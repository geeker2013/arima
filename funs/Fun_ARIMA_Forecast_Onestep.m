%% 进行使用ARIMA进行“单步”预测的函数
function [forData,dataTrain,dataTest,aicorbic,res] = Fun_ARIMA_Forecast_Onestep(data,TrainR,max_ar,max_ma,figflag,criterion)
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

%  Copyright (c) 2019 Mr.括号 All rights reserved.
%  原文链接 https://zhuanlan.zhihu.com/p/69630638
%  代码地址：http://www.khscience.cn/docs/index.php/2020/04/19/123/
%  本代码为淘宝买家专用，不开源，请勿公开分享~
%%
if ~exist('criterion')
    criterion  = [];  %如果没有输入criterion参数，则指定为空
end
%% 1.导入数据并划分数据集
if TrainR < 1  %输入的TrainR是比例
    TrainLen = round(length(data)*TrainR);  %用于训练模型的数据长度
else %输入的TrainR是数据长度
    TrainLen = TrainR;
end

data = data(:);       %转化为列向量
dataTotal = data;
totalLen = length(data); %数据总长度
dataTrain = data(1:TrainLen);     %训练集数据
dataTest  = data(1+TrainLen:end); %测试集数据
data = dataTrain;
%% 2.平稳性检验与确定差分阶数i
if getStatAdfKpss(data)  %ADF和KPSS同时通过才认为通过稳定性检验
    Y = data;
    dN = 0;      %dN是差分阶数，如果数据能直接通过检验
else
    for i = 1:5  %最多进行5阶差分，如果5阶差分依然无法平稳则报错且终止程序
        if getStatAdfKpss(diff(data,i))
            Y = diff(data,i);  %进行i阶差分
            dN = i;   %差分阶数
            break
        end
    end
    if dN == 0 %数据经过5阶差分后未平稳 
        msgbox('数据无法通过差分平稳', '错误'); %报错
        return
    end
end
disp(['差分阶数为：',num2str(dN)]) %在命令行窗口输出确定的差分阶数
%% 3.确定ARMA模型阶数
% ACF和PACF法，确定阶数。在本流程中这两张图仅作为辅助参考，不作为定阶最终依据
figure('Name','平稳信号自相关图','Visible',figflag,'color','w')
autocorr(Y)
figure('Name','平稳信号偏自相关图','Visible',figflag,'color','w')
parcorr(Y)
% 通过AIC，BIC等准则暴力选定阶数
try
    [AR_Order,MA_Order,aicorbic] = ARMA_Order_Select(data,max_ar,max_ma,dN,criterion);      %dY需要为列向量，输入输出参数含义见文件ARMA_Order_Select.m
catch ME  %捕捉错误信息
    msgtext = ME.message;
    if (strcmp(ME.identifier,'econ:arima:estimate:InvalidVarianceModel'))
         msgtext = [msgtext,'  ','无法进行arima模型估计，这可能是由于用于训练的数据长度较小，而要进行拟合的阶数较高导致的，请尝试减小max_ar和max_ma的值'];
    end
    msgbox(msgtext, '错误')
end
disp(['p=',num2str(AR_Order),',q=',num2str(MA_Order)]); %在命令行窗口输出确定的p和q值
%% 4.残差检验
Mdl = arima(AR_Order, dN, MA_Order);  %第二个变量值为差分阶数
try
    % 注意：下边这行代码输入的数据是data而不是Y，这是因为在建立Mdl时，已经将差分dN构建至其中
    EstMdl = estimate(Mdl,data);
catch ME %捕捉错误信息
    msgtext = ME.message;
    if (strcmp(ME.identifier,'econ:arima:estimate:InvalidVarianceModel')) %定阶出的pq值依旧可能会导致arima参数估计无法完成，此时需降低max_ar和max_ma
         msgtext = [msgtext,'  ','无法进行arima模型估计，这可能是由于用于训练的数据长度较小，而要进行拟合的阶数较高导致的，请尝试减小max_ar和max_ma的值'];
    end
    msgbox(msgtext, '错误')
    return
end
[res,~,logL] = infer(EstMdl,data);   %res即残差

stdr = res/sqrt(EstMdl.Variance);
figure('Name','残差检验','Visible',figflag,'color','w')
subplot(2,3,1)
plot(stdr)
title('Standardized Residuals')
subplot(2,3,2)
histogram(stdr,10)
title('Standardized Residuals')
subplot(2,3,3)
autocorr(stdr)
subplot(2,3,4)
parcorr(stdr)
subplot(2,3,5)
qqplot(stdr)
% Durbin-Watson 统计是计量经济学分析中最常用的自相关度量
diffRes0 = diff(res);  
SSE0 = res'*res;
DW0 = (diffRes0'*diffRes0)/SSE0 % Durbin-Watson statistic，该值接近2，则可以认为序列不存在一阶相关性。
%% 5.预测
for i = TrainLen:totalLen-1
    if ~isempty(strfind(version,'2018'))||~isempty(strfind(version,'2017'))||~isempty(strfind(version,'2016')) %依据当前MATLAB版本执行不同语句
        [forData(i+1),YMSE] = forecast(EstMdl,1,'Y0',dataTotal(1:i));   %matlab2018及以下版本写为Predict_Y = forecast(EstMdl,step,'Y0',Y);   matlab2019写为Predict_Y = forecast(EstMdl,step,Y);
    elseif ~isempty(strfind(version,'2019'))||~isempty(strfind(version,'2020'))||~isempty(strfind(version,'2021'))
        [forData(i+1),YMSE] = forecast(EstMdl,1,dataTotal(1:i));   %matlab2018及以下版本写为Predict_Y = forecast(EstMdl,step,'Y0',Y);   matlab2019写为Predict_Y = forecast(EstMdl,step,Y);
    else
        warndlg('仅支持MATLAB2017/2018/2019/2020/2021')
    end
end

figure('Visible',figflag,'color','w')
h1 = plot(data,'Color',[.7,.7,.7]); %训练集数据
hold on
h2 = plot(TrainLen:totalLen,dataTotal(TrainLen:end),'Color','k'); %测试集数据
% h1 = plot(length(data):length(data)+step,[data(end);lower],'r:','LineWidth',2);   %绘制置信区间下限
% plot(length(data):length(data)+step,[data(end);upper],'r:','LineWidth',2)         %绘制置信区间上限
h3 = plot(TrainLen:totalLen,[data(end);forData(TrainLen+1:end)'],'r');  %绘制预测曲线
legend([h1 h2 h3],'训练集数据','测试集真实值','测试集预测值',...  %图例
	     'Location','NorthWest')
title('Forecast')
hold off
end
function stat = getStatAdfKpss(data)
try 
    stat = adftest(data) && ~kpsstest(data);
catch ME
    msgtext = ME.message;
    if (strcmp(ME.identifier,'econ:adftest:EffectiveSampleSizeLessThanTabulatedValues'))
         msgtext = [msgtext,'  ','单位根检验无法进行，数据长度不足'];
    end
    msgbox(msgtext, '错误')
end
end



