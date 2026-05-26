@echo off
setlocal
set LOG=C:\Users\ESA_Lab\AppData\Local\Temp\opencode\stage34_3stage_40ep.log
if exist "%LOG%" del /f /q "%LOG%"
"C:\Users\ESA_Lab\anaconda3\envs\imu_for_workout\python.exe" -u -m train.micro_macro_recognition --config configs/micro_macro_recognition_stage3_40ep.yaml --micro-source tcn --no-timestamp --run-stamp stage34_3stage_40ep > "%LOG%" 2>&1
endlocal
