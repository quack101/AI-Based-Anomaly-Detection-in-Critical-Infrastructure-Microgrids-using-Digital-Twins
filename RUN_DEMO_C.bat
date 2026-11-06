@echo off
rem ======================================================================
rem  FULL LIVE DEMO (preset C): A1, A2, A3, A4, A5 in one live Kafka run.
rem  OpenDSS producer -> Kafka -> injector (attacks) -> detector
rem  (fusion BDD+AE  OR  voltage peer check  OR  power peer check) -> live view.
rem  Double-click. First time: installs kafka-python if missing and builds the
rem  A3/A4/A5 attack plans (~5-10 min). Close the previous run's live window first.
rem  The run itself: ~15 min (640 steps). Results: streaming\demo_runs\runC_all\
rem  Console output is also saved to run_demo_C.log
rem ======================================================================
cd /d "%~dp0"
set PY=%~dp0..\.venv-1\Scripts\python.exe
set LOG=%~dp0run_demo_C.log
echo === run start %date% %time% > "%LOG%"

echo [0/4] kafka-python ...
"%PY%" -c "import kafka" >nul 2>&1 || "%PY%" -m pip install "kafka-python>=2.0,<3" >> "%LOG%" 2>&1
echo [1/4] Docker / Kafka broker ...
docker info >nul 2>&1
if errorlevel 1 (
  echo   starting Docker Desktop
  if exist "C:\Program Files\Docker\Docker\Docker Desktop.exe" start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
  for /l %%i in (1,1,40) do (
    timeout /t 6 /nobreak >nul
    docker info >nul 2>&1 && goto dockerup
  )
  echo   Docker did not come up within 4 minutes -- start Docker Desktop and run again
  echo DOCKER_FAIL >> "%LOG%"
  pause
  goto :eof
)
:dockerup
docker compose -f streaming\docker-compose.yml up -d kafka-1 kafka-2 kafka-3 >> "%LOG%" 2>&1
for /l %%i in (1,1,40) do (
  "%PY%" -c "from kafka import KafkaAdminClient; from streaming.kafka_config import BOOTSTRAP_SERVERS; KafkaAdminClient(bootstrap_servers=BOOTSTRAP_SERVERS, request_timeout_ms=4000).close()" >nul 2>&1 && goto brokerup
  timeout /t 3 /nobreak >nul
)
echo   Kafka brokers not reachable on localhost:9092/9094/9096
echo BROKER_FAIL >> "%LOG%"
pause
goto :eof
:brokerup
echo   broker OK
echo BROKER_OK >> "%LOG%"

echo [2/4] A3/A4/A5 attack plans (built once, offline, with OpenDSS) ...
"%PY%" -m validation.gen_demo_attacks --start 80440 >> "%LOG%" 2>&1
if errorlevel 1 (
  echo   plan build FAILED -- see run_demo_C.log
  echo GEN_FAIL >> "%LOG%"
  pause
  goto :eof
)
echo GEN_OK >> "%LOG%"

echo [3/4] peer detectors (fitted once, cached) ...
"%PY%" -c "from streaming.peer_monitor import load_detectors; d=load_detectors(); print('  V-peer thr %%.3f, P-peer thr %%.3f' %% (d['V'].thr, d['P'].thr))"

echo [4/4] live run (about 15 min) -- the live window opens now
powershell -NoProfile -Command "& '%PY%' -m streaming.run_demo --run C --save-gif 2>&1 | Tee-Object -FilePath '%LOG%' -Append"
echo RUN_EXIT %errorlevel% >> "%LOG%"
echo === run end %date% %time% >> "%LOG%"
echo.
echo Done. Summary: streaming\demo_runs\runC_all\summary.json, timeline.png, live_demo.gif
pause
