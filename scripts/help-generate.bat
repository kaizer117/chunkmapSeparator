@echo off
echo ==============================================================================
echo   GENERATE - deterministic random island PNGs and label maps
echo ==============================================================================
echo.
echo One outline island:
echo.
echo     python main.py generate outputs\island.png 1 1024 1024 --seed 42
echo.
echo Eight images, four islands each, three subdivisions per island:
echo.
echo     python main.py generate outputs\islands 8 1024 768 --islands 4 --subdivisions 3 --seed 42
echo.
echo A solid landmass instead of an outline:
echo.
echo     python main.py generate outputs\filled 4 1024 768 --islands 3 --filled --seed 7
echo.
echo Expected console output, abridged:
echo.
echo     islands    8 x 1024x768  -^>  outputs\islands
echo     seed       42  (natural, outline, 4 per image, 3 subdivisions)
echo     [1/8] island_0001.png  seed 42  coverage 24.4%%  islands 4  regions 12  stroke 2.6  bbox 780x701
echo     ...
echo     done       8/8 in 13.2s
echo.
echo OUTPUT can be a directory, a .png file when count is 1, or a .png prefix.
echo The same seed always produces the same island.
echo The label map that region mode needs is produced by orchestration
echo (regions\<name>.png); "generate" alone writes only the PNG.
echo.
pause
