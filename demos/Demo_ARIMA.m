%% ʹ��Fun_ARIMA_Forecast����ʵ��Ԥ��
% 
close all
clear all
addpath ../funs %��funs�ļ�����ӽ�·��������Ѿ�ִ��install_funs������д������ɾ��
%1.��������
load Data_EquityIdx   %��˹����ۺ�ָ��
data = DataTable.NASDAQ(1:100); %���Ҫ�滻���ݣ����˴�data�滻���ɡ�
%2.���ú���
step = 10; %Ԥ�ⲽ��
p_max = 5; %p����
q_max = 5; %q����
figflag = 'on'; %��ͼ��־λ��'on'Ϊ��ͼ��'off'Ϊ����
[forData1,lower1,upper1,aicorbic,res] = Fun_ARIMA_Forecast(data,step,p_max,q_max,figflag,'aic'); %�ú����Ĳ�����Ϣ���ں����ļ��ڲ鿴
%  ʹ��ARIMA����Ԥ��ĺ�����ʹ��n�ײ�֡���ʹ�ö�����������ֱ�ӵ��ã���ֽ����Զ�ȷ����
% ���룺
% dataΪ��Ԥ�����ݣ�һά���ݣ���С11�����ݡ��������ݳ��ȴ���11~15ʱ���ɿ��ܳ��ֱ���������
% stepΪ��Ԥ�ⲽ��
% max_ar Ϊ���pֵ
% max_ma Ϊ���qֵ
% figflag Ϊ��ͼ��־λ��'on'Ϊ��ͼ��'off'Ϊ����
% criterion Ϊ����׼��'aic'/'bic'/'aic+bic'����ѡ�񣬴˱������Բ������������Ϊ[]����ʱ��ʹ��Ĭ��'aic+bic'׼��
% �����
% forDataΪԤ�������䳤�ȵ���step
% lowerΪԤ������95%��������ֵ
% upperΪԤ������95%��������ֵ
% aicorbicΪaic+bic������pq����µ�ֵ������ֵΪ����
% ���統����max_ar= 1��max_ma=2ʱ
% �򷵻ؾ���Ϊ�� a b c
%               d e f
% a���� p=0,q=0ʱ��aic+bic��ֵ��b����p=0,q=1ʱ��ֵ��d����p=1,q=0ʱ��ֵ���Դ�����
% �����п��ܻ����NaNֵ������a�ض�ΪNaN����p��q�Ľ����ϸ�ʱ��Ҳ����Ϊ����޷���ȷ���Ʋ���������NaNֵ
% res����ϲв�ֵ