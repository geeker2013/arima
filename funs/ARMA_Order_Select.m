function [AR_Order,MA_Order,aicorbic] = ARMA_Order_Select(data,max_ar,max_ma,di,criterion)
% ͨ��AIC��BIC��׼����ѡ�����������в����
% ���룺
% data��������
% max_arΪARģ����Ѱ��������
% max_maΪMAģ����Ѱ��������
% di��ֽ���
% criterion Ϊ����׼��'aic'/'bic'/'aic+bic'����ѡ�񣬴˱������Բ������������Ϊ[]����ʱ��ʹ��Ĭ��'aic+bic'׼��
% �����
% AR_OrderrΪARģ���������
% MA_OrderrΪARģ���������
% aicorbicΪaic+bic������pq����µ�ֵ������ֵΪ����
% ���統����max_ar= 1��max_ma=2ʱ
% �򷵻ؾ���Ϊ�� a b c
%               d e f
% a���� p=0,q=0ʱ��aic+bic��ֵ��b����p=0,q=1ʱ��ֵ��d����p=1,q=0ʱ��ֵ���Դ�����
% �����п��ܻ����NaNֵ������a�ض�ΪNaN����p��q�Ľ����ϸ�ʱ��Ҳ����Ϊ����޷���ȷ���Ʋ���������NaNֵ


%  Copyright (c) 2019 Mr.���� All rights reserved.
%  ԭ������ https://zhuanlan.zhihu.com/p/69630638
%  �����ַ��http://www.khscience.cn/docs/index.php/2020/04/19/123/
%  ������Ϊ�Ա����ר�ã�����Դ�����𹫿�����~
if ~exist('criterion')||isempty(criterion)  %δָ��׼��
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
            [aic,bic] = aicbic(LogL,(ar+ma+2),T); %����ar��ma�⣬���г����ͷ����+2
            switch criterion
                case 'aic'
                    infoC_Sum(ar+1,ma+1) = aic;  %��AIC֮Ϊ��׼����ѡȡ
                case 'bic'
                    infoC_Sum(ar+1,ma+1) = bic;  %��BICΪ��׼����ѡȡ
                case 'aic+bic'
                    infoC_Sum(ar+1,ma+1) = bic+aic;  %��BIC��AIC֮��Ϊ��׼����ѡȡ
            end
        catch ME %��׽������Ϣ
            msgtext = ME.message;
            if (strcmp(ME.identifier,'econ:arima:estimate:InvalidVarianceModel'))
                 infoC_Sum(ar+1,ma+1) = NaN; %�޷����Ʋ�����ֱ����nan
                %msgtext = [msgtext,'  ','�޷�����arimaģ�͹��ƣ����������������ѵ�������ݳ��Ƚ�С����Ҫ������ϵĽ����ϸߵ��µģ��볢�Լ�Сmax_ar��max_ma��ֵ']
            else
                %msgbox(msgtext, '����') %�رյ�����������
                infoC_Sum(ar+1,ma+1) = NaN; %�޷����Ʋ�����ֱ����nan
            end
        end
    end
end
aicorbic = infoC_Sum;  
[x, y]=find(infoC_Sum==min(min(infoC_Sum)));
AR_Order = x -1;
MA_Order = y -1;
end