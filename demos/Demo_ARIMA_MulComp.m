%% ʹ��Fun_ARIMA_Forecast����ʵ��Ԥ��
% ����Ԥ��
close all
clear all
addpath ../funs %��funs�ļ�����ӽ�·��������Ѿ�ִ��install_funs������д������ɾ��

% 1.��������
load Data_EquityIdx   %��˹����ۺ�ָ��
len = 2000;
data = DataTable.NASDAQ(1:len); %���Ҫ�滻���ݣ����˴�data�滻���ɡ�
%2.���ú�����ʵ�ֵ���Ԥ��
TrainR = 0.8; %����ѵ�������ݳ��ȣ�Ҳ����д��0.7����Ϊ����ֵ��
max_ar = 3;  %p����ֵ
max_ma = 3;  %q����ֵ
[forData,dataTrain,dataTest,aicorbic,res] = Fun_ARIMA_Forecast_MulComp(data,TrainR,max_ar,max_ma,'on','aic');