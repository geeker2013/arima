% 用于将funs添加至路径，便于funs中的函数文件在各处被调用

function install_funs
base_dir = fileparts(which('install_funs'));
addpath(genpath([base_dir,'/funs']));
savepath
disp('================================================');
sprintf('\r');
disp('安装完成');
sprintf('\r');
disp('================================================');
