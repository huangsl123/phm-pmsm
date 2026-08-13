@echo off
REM 训练最佳配置模型
REM 数据路径已设置为您的实际数据路径

set "PROJECT_ROOT=%~dp0.."
set "SOURCE_DATA=%PROJECT_ROOT%\datasets\dataset2_1.0kW.csv"
set "TARGET_DATA=%PROJECT_ROOT%\datasets\dataset2_3.0kW.csv"

echo ========================================
echo 训练最佳配置模型
echo ========================================
echo.

echo 源域数据: %SOURCE_DATA%
echo 目标域数据: %TARGET_DATA%
echo.

REM 训练所有最佳配置
python "%~dp0train_best_configs.py" ^
    --source_data "%SOURCE_DATA%" ^
    --target_data "%TARGET_DATA%" ^
    --epochs 100

echo.
echo ========================================
echo 训练完成!
echo ========================================
pause
