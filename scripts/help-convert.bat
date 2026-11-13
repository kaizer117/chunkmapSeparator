@echo off
echo ==============================================================================
echo   CONVERT - hand-drawn raster line art to a smooth bezier SVG
echo ==============================================================================
echo.
echo The default reads ink from the PNG alpha channel and writes one closed
echo bezier path per contour.
echo.
echo     python main.py resources\example-cuba.png
echo.
echo Choose the output path, stroke and a JSON report:
echo.
echo     python main.py ink.png -o outputs\ink.svg --stroke "#000000" --report-json outputs\ink.json
echo.
echo A black-on-white scan, where alpha is flat and luminance carries the ink:
echo.
echo     python main.py scan.png --mask luminance --luminance-threshold 160
echo.
echo The straight-line baseline, useful for comparison:
echo.
echo     python main.py ink.png --curve polygon
echo.
echo Expected console output, abridged:
echo.
echo     chunkmap-separator  raster -^> bezier svg
echo     input      ink.png  (1024x768, 4 channels)
echo     mask       alpha  -^>  12034 px  (1.530%% of image)
echo     contours   12 kept, 0 skipped, 0 failed
echo     vector     12 paths / 480 curves / 1450 control points
echo     svg        23.4 KB  -^>  outputs\ink.svg
echo       coverage   IoU 0.96 ...
echo       boundary   raster-^>vector mean 0.07 px ...
echo       compress   3200 raster points -^> 480 segments (6.7x)
echo.
echo Output: outputs\<input name>.svg unless -o is given.
echo Exit codes: 0 success, 2 bad input, 130 interrupted.
echo.
echo All ink-band options are listed in README.md, section "Command line tool".
echo.
pause
