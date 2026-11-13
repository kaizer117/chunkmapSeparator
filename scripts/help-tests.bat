@echo off
echo ==============================================================================
echo   TESTS - unit and integration suites
echo ==============================================================================
echo.
echo Activate first with setup.bat, then run from the repository root.
echo.
echo Everything:
echo.
echo     python -m pytest -q
echo.
echo Unit tests only - one module at a time:
echo.
echo     python -m pytest tests\unit -q
echo.
echo Integration tests only - the real CLI and batch runner:
echo.
echo     python -m pytest tests\integration -q
echo.
echo One file or one test:
echo.
echo     python -m pytest tests\unit\test_accuracy.py -q -v
echo     python -m pytest tests\integration\test_cli_convert.py::test_cli_bezier_end_to_end -q
echo.
echo Expected:
echo.
echo     .......................................................... [100%%]
echo     77 passed in 15.20s
echo.
echo All scratch files go under outputs\_test\ and are deleted per test, so the
echo suite also works when the user profile is read-only. The suite does not
echo touch outputs\orchestration\ or any checked-in resource.
echo.
echo If pytest is missing:
echo     venv\Scripts\python -m pip install -r requirements.txt
echo.
pause
