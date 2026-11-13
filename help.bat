@echo off
REM Interactive index for the echo-only help pages in scripts\.
REM Nothing here runs the converter or the tests; it only prints the commands
REM and what to expect. Close the window or press Q to quit.

setlocal
title Chunk Map Separator - help

:menu
cls
echo ==============================================================================
echo   CHUNK MAP SEPARATOR - HELP INDEX
echo ==============================================================================
echo.
echo   These pages only PRINT instructions. They do not process images.
echo   All commands are run from the repository root.
echo.
echo   [1] Setup ......... activate venv\, confirm the install
echo   [2] Convert ....... raster line art -^> bezier SVG
echo   [3] Regions ....... label map -^> filled SVG with shared borders
echo   [4] Generate ...... random island PNGs and label maps
echo   [5] Orchestrate ... batch: islands -^> SVG -^> overlay -^> accuracy
echo   [6] Tests ......... unit tests, integration tests, full suite
echo   [7] Accuracy ...... score an existing SVG without converting
echo   [Q] Quit
echo.
choice /c 1234567Q /n /m "Select 1-7 or Q to quit: "
if errorlevel 8 goto :end
if errorlevel 7 goto :accuracy
if errorlevel 6 goto :tests
if errorlevel 5 goto :orchestrate
if errorlevel 4 goto :generate
if errorlevel 3 goto :regions
if errorlevel 2 goto :convert
if errorlevel 1 goto :setup
goto :menu

:setup
call "%~dp0scripts\help-setup.bat"
goto :menu

:convert
call "%~dp0scripts\help-convert.bat"
goto :menu

:regions
call "%~dp0scripts\help-regions.bat"
goto :menu

:generate
call "%~dp0scripts\help-generate.bat"
goto :menu

:orchestrate
call "%~dp0scripts\help-orchestrate.bat"
goto :menu

:tests
call "%~dp0scripts\help-tests.bat"
goto :menu

:accuracy
call "%~dp0scripts\help-accuracy.bat"
goto :menu

:end
endlocal
