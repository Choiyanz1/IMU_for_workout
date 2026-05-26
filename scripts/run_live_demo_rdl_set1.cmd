@echo off
setlocal
"C:\Users\ESA_Lab\anaconda3\envs\imu_for_workout\python.exe" -m evaluation.streaming_micro_macro ^
  --run-dir artifacts/micro_macro_recognition/stage34_3stage_40ep/tcn ^
  --csv datasets/raw_data/kevin/db_rdl/set1 ^
  --output-dir artifacts/best_demo/live_kevin_db_rdl_set1 ^
  --device cuda ^
  --method step ^
  --live ^
  --micro-smoothing-window 15 ^
  --open-browser ^
  --startup-delay-seconds 5 ^
  --replay-speed 1.0 ^
  --live-update-interval 5 ^
  --live-port 8766 ^
  --keep-server-open
endlocal
