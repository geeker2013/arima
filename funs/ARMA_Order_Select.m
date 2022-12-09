function [AR_Order,MA_Order,aicorbic] = ARMA_Order_Select(data,max_ar,max_ma,di,criterion)
% 通过AIC，BIC等准则暴力选定阶数，带有差分项
% 输入：
% data对象数据
% max_ar为AR模型搜寻的最大阶数
% max_ma为MA模型搜寻的最大阶数
% di差分阶数
% criterion 为定阶准则，'aic'/'bic'/'aic+bic'三种选择，此变量可以不输入或者输入为[]，此时将使用默认'aic+bic'准则
% 输出：
% AR_Orderr为AR模型输出阶数
% MA_Orderr为AR模型输出阶数
% aicorbic为aic+bic在所有pq组合下的值，返回值为矩阵，
% 例如当设置max_ar= 1，max_ma=2时
% 则返回矩阵为： a b c
%               d e f
% a代表 p=0,q=0时的aic+bic的值，b代表p=0,q=1时的值，d代表p=1,q=0时的值，以此类推
% 矩阵中可能会存在NaN值，比如a必定为NaN，当p、q的阶数较高时，也会因为软件无法正确估计参数而产生NaN值


%  Copyright (c) 2019 Mr.括号 All rights reserved.
%  原文链接 https://zhuanlan.zhihu.com/p/69630638
%  代码地址：http://www.khscience.cn/docs/index.php/2020/04/19/123/
%  本代码为淘宝买家专用，不开源，请勿公开分享~
if ~exist('criterion')||isempty(criterion)  %未指定准则
    criterion = 'aic+bic';
end
T = length(data);

for ar = 0:max_ar
    for ma = 0:max_ma
        if ar==0&&ma==0
            infoC_Sum = NaN;
            continue
        end
        try
            Mdl = arima(ar, di, ma);
            [~, ~, LogL] = estimate(Mdl, data, 'Display', 'off');
            [aic,bic] = aicbic(LogL,(ar+ma+2),T); %除了ar与ma外，还有常数和方差，故+2
            switch criterion
                case 'aic'
                    infoC_Sum(ar+1,ma+1) = aic;  %以AIC之为标准进行选取
                case 'bic'
                    infoC_Sum(ar+1,ma+1) = bic;  %以BIC为标准进行选取
                case 'aic+bic'
                    infoC_Sum(ar+1,ma+1) = bic+aic;  %以BIC和AIC之和为标准进行选取
            end
        catch ME %捕捉错误信息
            msgtext = ME.message;
            if (strcmp(ME.identifier,'econ:arima:estimate:InvalidVarianceModel'))
                 infoC_Sum(ar+1,ma+1) = NaN; %无法估计参数，直接置nan
                %msgtext = [msgtext,'  ','无法进行arima模型估计，这可能是由于用于训练的数据长度较小，而要进行拟合的阶数较高导致的，请尝试减小max_ar和max_ma的值']
            else
                %msgbox(msgtext, '错误') %关闭弹窗报错提醒
                infoC_Sum(ar+1,ma+1) = NaN; %无法估计参数，直接置nan
            end
        end
    end
end
aicorbic = infoC_Sum;  
[x, y]=find(infoC_Sum==min(min(infoC_Sum)));
AR_Order = x -1;
MA_Order = y -1;
end