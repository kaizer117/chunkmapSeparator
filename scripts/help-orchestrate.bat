@echo off
echo ==============================================================================
echo   ORCHESTRATE - batch integration test
echo ==============================================================================
echo.
echo Generates islands, vectorizes each one through the real CLI, renders an
echo overlay and writes accuracy reports plus a batch summary.
echo.
echo Full sample run:
echo.
echo     python main.py orchestrate --config configs\island-svg-sample.yaml
echo.
echo A quick two-image run in a throwaway space:
echo.
echo     python main.py orchestrate --config configs\island-svg-sample.yaml --count 2 --space outputs\orchestration\quick --clean
echo.
echo Output layout inside the test space:
echo.
echo     islands\island_0001.png      generated raster
echo     regions\island_0001.png      16-bit label map
echo     svg\island_0001.svg          one filled path per region
echo     overlay\island_0001.png      island + SVG paths in a contrasting colour
echo     reports\island_0001.json     full per-image metrics
echo     reports\summary.json         config snapshot, per-image digest, aggregate
echo     reports\summary.csv          one row per image
echo.
echo Expected console tail, abridged:
echo.
echo     summary    2/2 images  2 ok  0 below  0 failed  wall 12.40s
echo     coverage   IoU  mean 0.9991  median 0.9992 ...
echo     vector     1,086 curves total ...
echo     reports    ...\reports\summary.json
echo                ...\reports\summary.csv
echo.
echo Flags: --space DIR, --seed N, --count N, --only 1,3-5, --compact,
echo        --quiet, --no-overlay, --clean, --fail-fast.
echo Exit codes: 0 all good, 1 a failure or below-threshold image,
echo             2 bad config, 130 interrupted.
echo.
pause
