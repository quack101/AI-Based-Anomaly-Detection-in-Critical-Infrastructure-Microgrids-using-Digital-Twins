@echo off
cd /d "%~dp0"
set PY=%~dp0..\.venv-1\Scripts\python.exe
set BENCH_NAME=attack_bench_a45_small
set A45_MAGS=0.00001,0.00003,0.0001,0.0003,0.001
set A45_REPS=4
set SKIP_A3=1
set BENCH_SEED=20261106
echo === start %date% %time% > bench_run_small.log
"%PY%" -u -m validation.gen_attack_bench >> bench_run_small.log 2>&1
echo === end %date% %time% >> bench_run_small.log
