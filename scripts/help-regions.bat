@echo off
echo ==============================================================================
echo   REGIONS - label map to filled SVG with shared borders
echo ==============================================================================
echo.
echo Region mode takes a single-channel integer label map: 0 is water, every
echo positive number is one region. Adjacent regions reuse the exact same
echo cubic border, so there is no gap and no doubled stroke.
echo.
echo     python main.py island.png --regions island-labels.png -o island-regions.svg
echo.
echo The same geography as one stroked boundary network:
echo.
echo     python main.py island.png --regions island-labels.png --region-style boundary -o island-borders.svg
echo.
echo Choose the fill style and hide renderer seams:
echo.
echo     python main.py island.png --regions labels.png --region-fill solid --region-color "#9ec79b" --region-seam 0.5
echo.
echo Expected console output, abridged:
echo.
echo     chunkmap-separator  regions -^> fill svg
echo     input      island.png  (1024x768)
echo     label map  labels.png  (7 regions)
echo     regions    7 paths  18 boundary chains (11 shared, 7 coastline)
echo     vector     540 curves  2400 raster boundary edges
echo     svg        45.2 KB  -^>  island-regions.svg
echo.
echo The input PNG and the label map must have the same width and height.
echo Use "python main.py generate ... --subdivisions N" to create a matching pair.
echo Exit codes: 0 success, 2 bad input or wrong-size label map.
echo.
pause
