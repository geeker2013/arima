%% ����ʹ��ARIMA���С�������Ԥ��ĺ���
function [forData,dataTrain,dataTest,aicorbic,res] = Fun_ARIMA_Forecast_Onestep(data,TrainR,max_ar,max_ma,figflag,criterion)
% ʹ��ARIMA���е���Ԥ��ĺ�����ʹ��n�ײ�֡���ʹ�ö�����������ֱ�ӵ��ã���ֽ����Զ�ȷ����
% ���룺
% data   Ϊȫ�����ݣ�Ҫ��Ϊһά���ݡ��������ڳ����л�����TrainR����ֵ����һ������Ϊѵ�����Ͳ��Լ�
% TrainR Ϊѵ�������������TrainRΪС��1��С������������ֵ�����TrainΪ����1�������������ѵ�������ݵ�����
% max_ar Ϊ���pֵ
% max_ma Ϊ���qֵ
% figflag Ϊ��ͼ��־λ��'on'Ϊ��ͼ��'off'Ϊ����
% criterion Ϊ����׼��'aic'/'bic'/'aic+bic'����ѡ�񣬴˱������Բ������������Ϊ[]����ʱ��ʹ��Ĭ��'aic+bic'׼��
% �����
% forDataΪԤ����
% dataTrainΪѵ��������
% dataTestΪ���Լ�����
% aicorbicΪaic+bic������pq����µ�ֵ������ֵΪ����
% ���統����max_ar= 1��max_ma=2ʱ
% �򷵻ؾ���Ϊ�� a b c
%               d e f
% a���� p=0,q=0ʱ��aic+bic��ֵ��b����p=0,q=1ʱ��ֵ��d����p=1,q=0ʱ��ֵ���Դ�����
% �����п��ܻ����NaNֵ������a�ض�ΪNaN����p��q�Ľ����ϸ�ʱ��Ҳ����Ϊ����޷���ȷ���Ʋ���������NaNֵ
% res����ϲв�ֵ

%  Copyright (c) 2019 Mr.���� All rights reserved.
%  ԭ������ https://zhuanlan.zhihu.com/p/69630638
%  �����ַ��http://www.khscience.cn/docs/index.php/2020/04/19/123/
%  ������Ϊ�Ա����ר�ã�����Դ�����𹫿�����~
%%
if ~exist('criterion')
    criterion  = [];  %���û������criterion��������ָ��Ϊ��
end
%% 1.�������ݲ��������ݼ�
if TrainR < 1  %�����TrainR�Ǳ���
    TrainLen = round(length(data)*TrainR);  %����ѵ��ģ�͵����ݳ���
else %�����TrainR�����ݳ���
    TrainLen = TrainR;
end

data = data(:);       %ת��Ϊ������
dataTotal = data;
totalLen = length(data); %�����ܳ���
dataTrain = data(1:TrainLen);     %ѵ��������
dataTest  = data(1+TrainLen:end); %���Լ�����
data = dataTrain;
%% 2.ƽ���Լ�����ȷ����ֽ���i
if getStatAdfKpss(data)  %ADF��KPSSͬʱͨ������Ϊͨ���ȶ��Լ���
    Y = data;
    dN = 0;      %dN�ǲ�ֽ��������������ֱ��ͨ������
else
    for i = 1:5  %������5�ײ�֣����5�ײ����Ȼ�޷�ƽ���򱨴�����ֹ����
        if getStatAdfKpss(diff(data,i))
            Y = diff(data,i);  %����i�ײ��
            dN = i;   %��ֽ���
            break
        end
    end
    if dN == 0 %���ݾ���5�ײ�ֺ�δƽ�� 
        msgbox('�����޷�ͨ�����ƽ��', '����'); %����
        return
    end
end
disp(['��ֽ���Ϊ��',num2str(dN)]) %�������д������ȷ���Ĳ�ֽ���
%% 3.ȷ��ARMAģ�ͽ���
% ACF��PACF����ȷ���������ڱ�������������ͼ����Ϊ�����ο�������Ϊ������������
figure('Name','ƽ���ź������ͼ','Visible',figflag,'color','w')
autocorr(Y)
figure('Name','ƽ���ź�ƫ�����ͼ','Visible',figflag,'color','w')
parcorr(Y)
% ͨ��AIC��BIC��׼����ѡ������
try
    [AR_Order,MA_Order,aicorbic] = ARMA_Order_Select(data,max_ar,max_ma,dN,criterion);      %dY��ҪΪ�������������������������ļ�ARMA_Order_Select.m
catch ME  %��׽������Ϣ
    msgtext = ME.message;
    if (strcmp(ME.identifier,'econ:arima:estimate:InvalidVarianceModel'))
         msgtext = [msgtext,'  ','�޷�����arimaģ�͹��ƣ����������������ѵ�������ݳ��Ƚ�С����Ҫ������ϵĽ����ϸߵ��µģ��볢�Լ�Сmax_ar��max_ma��ֵ'];
    end
    msgbox(msgtext, '����')
end
disp(['p=',num2str(AR_Order),',q=',num2str(MA_Order)]); %�������д������ȷ����p��qֵ
%% 4.�в����
Mdl = arima(AR_Order, dN, MA_Order);  %�ڶ�������ֵΪ��ֽ���
try
    % ע�⣺�±����д��������������data������Y��������Ϊ�ڽ���Mdlʱ���Ѿ������dN����������
    EstMdl = estimate(Mdl,data);
catch ME %��׽������Ϣ
    msgtext = ME.message;
    if (strcmp(ME.identifier,'econ:arima:estimate:InvalidVarianceModel')) %���׳���pqֵ���ɿ��ܻᵼ��arima���������޷���ɣ���ʱ�轵��max_ar��max_ma
         msgtext = [msgtext,'  ','�޷�����arimaģ�͹��ƣ����������������ѵ�������ݳ��Ƚ�С����Ҫ������ϵĽ����ϸߵ��µģ��볢�Լ�Сmax_ar��max_ma��ֵ'];
    end
    msgbox(msgtext, '����')
    return
end
[res,~,logL] = infer(EstMdl,data);   %res���в�

stdr = res/sqrt(EstMdl.Variance);
figure('Name','�в����','Visible',figflag,'color','w')
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
% Durbin-Watson ͳ���Ǽ�������ѧ��������õ�����ض���
diffRes0 = diff(res);  
SSE0 = res'*res;
DW0 = (diffRes0'*diffRes0)/SSE0 % Durbin-Watson statistic����ֵ�ӽ�2���������Ϊ���в�����һ������ԡ�
%% 5.Ԥ��
for i = TrainLen:totalLen-1
    if ~isempty(strfind(version,'2018'))||~isempty(strfind(version,'2017'))||~isempty(strfind(version,'2016')) %���ݵ�ǰMATLAB�汾ִ�в�ͬ���
        [forData(i+1),YMSE] = forecast(EstMdl,1,'Y0',dataTotal(1:i));   %matlab2018�����°汾дΪPredict_Y = forecast(EstMdl,step,'Y0',Y);   matlab2019дΪPredict_Y = forecast(EstMdl,step,Y);
    elseif ~isempty(strfind(version,'2019'))||~isempty(strfind(version,'2020'))||~isempty(strfind(version,'2021'))
        [forData(i+1),YMSE] = forecast(EstMdl,1,dataTotal(1:i));   %matlab2018�����°汾дΪPredict_Y = forecast(EstMdl,step,'Y0',Y);   matlab2019дΪPredict_Y = forecast(EstMdl,step,Y);
    else
        warndlg('��֧��MATLAB2017/2018/2019/2020/2021')
    end
end

figure('Visible',figflag,'color','w')
h1 = plot(data,'Color',[.7,.7,.7]); %ѵ��������
hold on
h2 = plot(TrainLen:totalLen,dataTotal(TrainLen:end),'Color','k'); %���Լ�����
% h1 = plot(length(data):length(data)+step,[data(end);lower],'r:','LineWidth',2);   %����������������
% plot(length(data):length(data)+step,[data(end);upper],'r:','LineWidth',2)         %����������������
h3 = plot(TrainLen:totalLen,[data(end);forData(TrainLen+1:end)'],'r');  %����Ԥ������
legend([h1 h2 h3],'ѵ��������','���Լ���ʵֵ','���Լ�Ԥ��ֵ',...  %ͼ��
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
         msgtext = [msgtext,'  ','��λ�������޷����У����ݳ��Ȳ���'];
    end
    msgbox(msgtext, '����')
end
end



