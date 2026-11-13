@echo off
echo ==============================================================================
echo   ACCURACY - score an existing SVG without converting
echo ==============================================================================
echo.
echo The measurement module is also a standalone tool:
echo.
echo     python -m chunkmap.accuracy resources\example-cuba.png outputs\cuba-bezier.svg
echo.
echo Write the full metrics as JSON:
echo.
echo     python -m chunkmap.accuracy resources\example-cuba.png outputs\cuba-bezier.svg --json outputs\report.json
echo.
echo Score the checked-in reference polygon SVG:
echo.
echo     python -m chunkmap.accuracy resources\example-cuba.png resources\cuba_ex.svg
echo.
echo Expected output, abridged:
echo.
echo     coverage   IoU 0.9655   Dice 0.9824 ...
echo     boundary   raster-^>vector mean 0.07 px ...
echo     compress   30333 raster points -^> 3157 segments (9.6x)
echo.
echo It accepts the same mask flags as the converter:
echo --mask, --alpha-threshold, --luminance-threshold, --invert, --json, --quiet.
echo Supported SVG commands: M, L, H, V, C, Z. Anything else is an error.
echo.
pause
