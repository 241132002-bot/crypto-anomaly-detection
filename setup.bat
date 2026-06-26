@echo off
echo ====================================
echo  Настройка на проекта
echo ====================================

echo.
echo Създаване на виртуална среда...
python -m venv .venv

echo.
echo Активиране...
call .venv\Scripts\activate.bat

echo.
echo Инсталиране на библиотеки...
pip install --upgrade pip
pip install pandas numpy scikit-learn matplotlib seaborn shap joblib

echo.
echo Инсталиране на PyTorch...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo.
echo Инсталиране на PyTorch Geometric...
pip install torch-geometric

echo.
echo ====================================
echo  Готово! Отвори VS Code и избери
echo  .venv като Python interpreter
echo ====================================
pause
