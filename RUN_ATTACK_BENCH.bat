@echo off
cd /d "%~dp0"
set PY=%~dp0..\.venv-1\Scripts\python.exe
echo === start %date% %time% > bench_run.log
"%PY%" -c "import opendssdirect, numpy, pandas, scipy; print('deps ok', opendssdirect.__version__)" >> bench_run.log 2>&1
"%PY%" -u -m validation.gen_attack_bench >> bench_run.log 2>&1
echo === end %date% %time% >> bench_run.log
