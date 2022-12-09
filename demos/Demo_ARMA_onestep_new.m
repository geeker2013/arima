%% ʹ��Fun_ARIMA_Forecast����ʵ��Ԥ��
% ����Ԥ��
close all
clear all
addpath ../funs %��funs�ļ�����ӽ�·��������Ѿ�ִ��install_funs������д������ɾ��

% 1.��������
load Data_EquityIdx   %��˹����ۺ�ָ��
len = 140;
data = DataTable.NASDAQ(1:len); %���Ҫ�滻���ݣ����˴�data�滻���ɡ�
%2.���ú�����ʵ�ֵ���Ԥ��
TrainR = 120; %����ѵ�������ݳ��ȣ�Ҳ����д��0.7����Ϊ����ֵ��
max_ar = 3;  %p����ֵ
max_ma = 3;  %q����ֵ
[forData,dataTrain,dataTest,aicorbic,res] = Fun_ARIMA_Forecast_Onestep(data,TrainR,max_ar,max_ma,'on','aic');

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