% ��funs��·�����Ƴ�

function uninstall_funs
base_dir = fileparts(which('uninstall_funs'));
rmpath(genpath([base_dir,'/funs']));

disp('================================================');
sprintf('\r');
disp('ж�����');
sprintf('\r');
disp('================================================');
