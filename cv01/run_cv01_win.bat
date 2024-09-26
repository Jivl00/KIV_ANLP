@echo off
setlocal

set models=dense cnn
set lrs=0.1 0.01 0.001 0.0001 0.00001
set opts=sgd adam
set dps=0 0.1 0.3 0.5

for %%m in (%models%) do (
    for %%l in (%lrs%) do (
        for %%o in (%opts%) do (
            for %%d in (%dps%) do (
                echo model=%%m, lr=%%l, optimizer=%%o, dp=%%d
                C:\Users\vladka\Documents\FAV\5.zimak\ANLP\cviceni\anlp-2024_kimlova_vladimira\venv\Scripts\python.exe ..\run_cv01.py --model %%m --lr %%l --optimizer %%o --dp %%d
            )
        )
    )
)

endlocal