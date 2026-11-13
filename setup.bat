@echo off
REM Activate the bundled virtual environment and stay at the repository root.
REM Works when launched from any working directory, including paths with spaces.

SET "SCRIPT_DIR=%~dp0"
SET "VENV_ACTIVATE=%SCRIPT_DIR%venv\Scripts\activate.bat"

IF NOT EXIST "%VENV_ACTIVATE%" (
    ECHO [setup] Virtual environment not found:
    ECHO         "%VENV_ACTIVATE%"
    ECHO [setup] Create it first from the repository root:
    ECHO         python -m venv venv
    ECHO         venv\Scripts\python -m pip install -r requirements.txt
    EXIT /B 1
)

REM Quote the path: cmd treats the space in "Kaveesha Perara" as an argument break.
CALL "%VENV_ACTIVATE%"
IF NOT DEFINED VIRTUAL_ENV (
    ECHO [setup] Failed to activate the virtual environment.
    EXIT /B 1
)

REM Run every command from the repository root so main.py finds resources\ and
REM configs\, and so outputs\ lands in one predictable place.
CD /D "%SCRIPT_DIR%"
IF ERRORLEVEL 1 (
    ECHO [setup] Failed to change directory to "%SCRIPT_DIR%".
    EXIT /B 1
)

ECHO [setup] Environment ready. You are in:
CD
ECHO [setup] Try: python main.py --help

REM Keep the activated environment and new cwd in the calling shell.
EXIT /B 0
