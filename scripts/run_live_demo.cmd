@echo off
setlocal
echo.
echo Live demo models:
echo   1. baseline: stage34_3stage_40ep
echo   2. dual-head: exp_dual_head
echo   3. dual-head-macro: exp_dual_head_macro
echo   4. custom run-dir
echo.
set /p MODEL_CHOICE=Choose model [1/2/3/4]: 

if "%MODEL_CHOICE%"=="1" goto model1
if "%MODEL_CHOICE%"=="2" goto model2
if "%MODEL_CHOICE%"=="3" goto model3
if "%MODEL_CHOICE%"=="4" goto model4
echo Invalid model choice: %MODEL_CHOICE%
exit /b 1

:model1
set "RUN_DIR=artifacts/micro_macro_recognition/stage34_3stage_40ep/tcn"
set "MODEL_TAG=baseline"
goto choose_stream

:model2
set "RUN_DIR=artifacts/micro_macro_recognition/exp_dual_head/tcn"
set "MODEL_TAG=dual_head"
goto choose_stream

:model3
set "RUN_DIR=artifacts/micro_macro_recognition/exp_dual_head_macro/tcn"
set "MODEL_TAG=dual_head_macro"
goto choose_stream

:model4
set /p RUN_DIR=Enter run-dir path: 
set "MODEL_TAG=custom"
goto choose_stream

:choose_stream
echo.
echo Live demo streams:
echo   1. kevin / db_weighted_crunch / set0
echo   2. kevin / db_rdl / set1
echo   3. custom csv or set-dir
echo.
set /p STREAM_CHOICE=Choose stream [1/2/3]: 

if "%STREAM_CHOICE%"=="1" (
  set "CSV_PATH=datasets/raw_data/kevin/db_weighted_crunch/set0"
  set "STREAM_TAG=kevin_db_weighted_crunch_set0"
  set "LIVE_PORT=8765"
  goto run
)

if "%STREAM_CHOICE%"=="2" (
  set "CSV_PATH=datasets/raw_data/kevin/db_rdl/set1"
  set "STREAM_TAG=kevin_db_rdl_set1"
  set "LIVE_PORT=8766"
  goto run
)

if "%STREAM_CHOICE%"=="3" (
  set /p CSV_PATH=Enter csv or set-dir path: 
  set /p STREAM_TAG=Enter output stream tag (no spaces): 
  set /p LIVE_PORT=Enter live port [default 8765]: 
  if "%LIVE_PORT%"=="" set "LIVE_PORT=8765"
  goto run
)

echo Invalid stream choice: %STREAM_CHOICE%
exit /b 1

:run
echo.
echo Run dir : %RUN_DIR%
echo Input   : %CSV_PATH%
echo Output  : artifacts/best_demo/live_%MODEL_TAG%_%STREAM_TAG%
echo Port    : %LIVE_PORT%
echo.
"C:\Users\ESA_Lab\anaconda3\envs\imu_for_workout\python.exe" -m evaluation.streaming_micro_macro ^
  --run-dir "%RUN_DIR%" ^
  --csv "%CSV_PATH%" ^
  --output-dir "artifacts/best_demo/live_%MODEL_TAG%_%STREAM_TAG%" ^
  --device cuda ^
  --method step ^
  --live ^
  --micro-smoothing-window 15 ^
  --open-browser ^
  --startup-delay-seconds 5 ^
  --replay-speed 1.0 ^
  --live-update-interval 5 ^
  --live-port %LIVE_PORT% ^
  --keep-server-open
endlocal
