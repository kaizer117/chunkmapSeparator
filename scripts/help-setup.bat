@echo off
echo ==============================================================================
echo   SETUP - activate the environment
echo ==============================================================================
echo.
echo From the repository root:
echo.
echo     setup.bat
echo.
echo It activates venv\ and leaves the shell at the repository root, where
echo main.py can see resources\, configs\ and outputs\.
echo.
echo Expected:
echo     [setup] Environment ready. You are in:
echo     C:\...\chunkMapSeparator
echo     [setup] Try: python main.py --help
echo.
echo If the virtual environment is missing, create it once:
echo.
echo     python -m venv venv
echo     venv\Scripts\python -m pip install -r requirements.txt
echo.
echo Check the tool is importable:
echo.
echo     python main.py --help
echo.
echo You do not need to create outputs\ by hand; the tool creates it on the
echo first write. Tests create outputs\_test\ and clean up after themselves.
echo.
pause
