<h1>Chunk Map Separator</h1>

<p style="text-align:center">
<cite>The good cartographer is both a scientist and an artist. </cite>
<br><b>Erwin Josephus Raisz (1893 – 1968)</b>
</p> 


<img src="resources/read_me_rcs/readme_banner.png" style="display:block;margin:auto;width: 90%; height: 90%"> <br>
<p>
Chunk Map Seprator is a convenient piece of software that can help with creating high-quality maps with beautiful procedurely generated color palattes. Primarily written in python, this software allows you to beautify your maps, create visually striking details and customize color palattes to your hearts content.<br><b>Note: This project is still WIP with many fetaures yet to come</b> </p>

<h2>Acknowledgements</h2>
<p>

- Vectorization algorithm: Heavily inspired by Dr. Tim A. Pastva's thesis "<a href = "https://calhoun.nps.edu/entities/publication/8126011c-a7ec-4cad-8372-4c971bf915a9">Bezier Curve Fitting</a>". <b>I have introduced some optimizations for his algorithms, for faster vectorization.</b>
    - Such as cachehing bernstein matrices to save compute on recalculating bernstien matrices each iteration.
    - And early termination techniques for results that achieve a certain level of tolerance.
- Curvature aware adaptive polygon segmentation: A multiple sources were used
    - Wikipedia's equations for <a href="https://en.wikipedia.org/wiki/Curvature#Plane_curves">planar curvature</a> were used to calculate the curvature of a raster curve in 2D plane.
    - The mulit-scalar curvature based segmentation was implemented from F. Mokhtarin's paper "<a href="https://ieeexplore.ieee.org/document/149591">A theory of multiscale, curvature-based shape representation for planar curves</a>".
</p>



<h2>Setup</h2>
<p>
I suggest you clone this repository and use Visual Studio Code as your IDE. Once successfullly cloned, create an empty <code>outputs</code> folder in your repository. 
</p>

<h2>Files</h2>
<p>

</p>

<h2>Workflow</h2>

<p>
For an example for the workflow, I will be using this map of Cuban Islands, I am currently working on. <br>

<h3>Loading a border map</h3>

For your first step, you should have a map with an transparent background and borders drawn in black color, as shown below.<br>

<img src="resources/read_me_rcs/example-cuba-transparent-background.png" style="display:block;margin:auto;width: 90%; height: 90%"><br>

Save this file under any name you like in the <code>\\resources</code> folder. I've saved the image above as <code>example-cuba.png</code>.<br>

<img src="resources/read_me_rcs/1.png" style="display:block;margin:auto;width: 90%; height: 90%"><br>

Assuming you have followed the above instructions to the dot, open <code>\\code\\main.py</code>.<br><br>
In the <code>main.py</code> code, locate <code>inputImg</code> variable and change the variable name to the file name of your border map.<br>

<img src="resources/read_me_rcs/2.png" style="display:block;margin:auto;width: 90%; height: 90%"><br>

You have now, successfully loaded an image into the program.

<h3>Generating color palatte</h3>

This step requires some familiarity with color theory or atleast color wheel. If you are curious, I would suggest you checkout <a href="http://www.workwithcolor.com/hsl-color-picker-01.htm">this online color wheel</a>. Specifically, take note of hue values and their respective color.<br><br>

For this step, with reference to the color wheel, pick a hue alue of your choosing, 
Next up, you can change the sampling and picking parapmeters for <code>colutils.cmapHex()</code> functions.<br>Here is a list of sampling and picking options.<br>


<b>Sampling:</b>
<ul>
<li>Linear: Generate hue, saturationand lightness values equidistant from one another</li>
<li>Random: Generate hue, saturation and lightness values randomly</li>
</ul>

<b>Picking</b>
<ul>
<li>Linear: Collate hue, saturation and lightness values as is</li>
<li>Random: Shuffle hue,saturation and lightness values before collating</li>
</ul>


If you prefer more randomness, pick both options to be Random. If you prefer a less chaotic color palatte, I would suggest you pick linear options. You are allowed to combine these as you please, so if you are not happy, then you try what combination works for you best.<br><br>


I have picked 12 as my hue value, "random" as my sampling procedure and "shuffle" as my picking procedure. Enter your values into the <code>colutils.cmapHex(..</code> line as shown below.<br>
<img src="resources/read_me_rcs/3.png" style="display:block;margin:auto;width: 90%; height: 90%"><br>

<h3>Generating SVG</h3>

Finally, take a gander at the <code>futils.saveCons(..</code> line. The third and fourth parameters are for the output file location and output file name. As is seen in the last image, I will be saing to the outputs folder inside the repository, and I would be naming the file <code>cuba-ex</code>.<br>

You are now ready to run the <code>main.py</code> file. If all goes well, the python file should runwithout errors. You should be seeing an svg file under the folder <code>outputs</code><br>
<img src="resources/read_me_rcs/4.png" style="display:block;margin:auto;width: 90%; height: 90%"><br>

You can open the svg file with a web browser or any vector editor. I have opened the output file in Inkscape.<br>
<img src="resources/read_me_rcs/svg-open.png" style="display:block;margin:auto;width: 90%; height: 90%"><br>
</p>



<h2>TODO:</h2>
<p>
This project still very much WIP. There is plenty of more to add.
</p>

<ol start=1>
<li>On-Demand colormap: Spits out colors based on a hue value infinetely.</li>
<li>Easy color editing for an already saved SVG.</li>
<li>Rudimentary UI features.</li>
<li>Write SVG files compatible with inkscape layers.</li>
<li>On-click contour selection in-UI</li>
</ol>

<br>
<h2>References</h2>
<a href="https://www.w3.org/TR/SVG2/">SVG file definitions</a><br>
<a href="https://en.wikipedia.org/wiki/HSL_and_HSV">HSL theory and HSl to RGB conversions</a>

