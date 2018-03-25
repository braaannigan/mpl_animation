# mpl_animation
A module to create line, color and scatter animations in matplotlib using an intuitive declarative syntax.  Contributions/suggestions are welcome!

This is an example of what you can create with a few lines of code.

![](https://github.com/braaannigan/mpl_animation/blob/master/scatter.gif)


The examples sub-directory shows how to make line, pcolor and scatter plots. 
https://github.com/braaannigan/mpl_animation/tree/master/mpl_animation/examples


## Installation
This is a tentative release of the module.  To use it you can:

```pip install mpl_animation```

or clone from the github source:

1. Clone this repository to your machine
```
git clone https://github.com/braaannigan/mpl_animation
```
2. Confirm that you have numpy and matplotlib installed
3. To do a quick test navigate to the mpl_animation directory and run an example
```
python examples line_example.py
```

## Quick start
To see the module in action quickly:
```python
import mpl_animation.anim as anim
import numpy as np

# x-axis of plot
x = np.arange(0,2*np.pi,1e-1)
# time-axis of plot
t = np.arange(0,8*np.pi,1e-1)

# Use meshgrid to create the data to be plotted
# (anim.plot can also accept 1D coordinate arrays)
T,X = np.meshgrid(t,x)

# Create data - a propagating cosine wave
cos_wave = (t/t.max())*np.cos(X - T)

# Create a title for each frame
titles = ['Wave at time {:.02f}'.format(i) for i in t]
# Create the plot object
cos_line = anim.plot(x, cos_wave, titles = titles)

# Animate the object
anim.animate(cos_line,figsize=(10,7))
```

## Saving to disk as mp4 file
To save a movie to disk as an mp4 file you add the keyword argument anim_save='movie_name.mp4' to 
the anim.animate() function.  In this case the movie will not be shown on the screen.  You need to have ffmpeg to save movies to disk.  Anaconda users may be able to do this with
```
conda install -c menpo ffmpeg
```
## Is there a way to stop the title being partially cut off?
I hope so!  All of the solutions I've found to this require modifying the rcparams, let me know if you have a way of doing so within the plotting routine.

