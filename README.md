# ColorGrade
A Tonemapping and Color Correction application

<b>You need :</b>
<ul>
  <li>Python3+</li>
  <li>OpenCV2 (from pip install opencv-python)</li>
  <li>Numba Cuda (I had CUDA 10)</li>  
</ul>

<b>Just run : <br></b>
\- python3 main.py

<br>
&nbsp&nbsp You have to copy paste full path of the video sample in the main.py script. (Cavemen coding) <br>
&nbsp&nbsp When none of the following command is active use mouse to move the slider between original and preview section.<br>
&nbsp&nbsp When any of the following command is active use mouse vertically to + or - that option.<br>
<br>
&nbsp&nbsp Some Refs : (Pressing these keys will activate the respective option)<br>
<b>1. Scale : s</b>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp   Scales the preview window.
<b>2. Edit Channel : r | g | b</b>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp   Curve manipulation.
<b>3. Edit Resolution : Ctrl + r | g | b</b>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp   Changes the resolution of the curve used to modify that channel.
<b>4. Preview Slider : Esc</b>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp   Stops the active command and returns back to preview slider
<b>5. Render : q</b>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp   Renders the video sample with currently applied settings of Tonemapping and ColorCorrection
