% 将funs从路径中移除

function uninstall_funs
base_dir = fileparts(which('uninstall_funs'));
rmpath(genpath([base_dir,'/funs']));

disp('================================================');
sprintf('\r');
disp('卸载完成');
sprintf('\r');
disp('================================================');
