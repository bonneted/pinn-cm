@ECHO OFF
SET sweep_id=%1
SET count=%2
cd /d C:\\Users\\u0150568\\PhD\\code\\pinn-cm\HPO
call C:\\ProgramData\\Anaconda3\\Scripts\\activate.bat C:\\Users\\u0150568\\.conda\\envs\\torch-gpu
echo Launching %count% run(s) on sweep %sweep_id%
wandb agent damien-bonnet/HPO-PINN-CM/%sweep_id% --count %count%
