# mpl_animation
A module to create animations in matplotlib using an intuitive declarative syntax.  Contributions welcome!
See this for an example of what you can create

![](https://github.com/braaannigan/mpl_animation/blob/master/scatter.gif)


The examples sub-directory shows how to make line, pcolor and scatter plots.

## Installation
This is a tentative release of the module.  To use it you:

1. Clone this repository to your machine
```
git clone https://github.com/braaannigan/mpl_animation
```
2. Confirm that you have numpy and matplotlib installed
3. To do a quick test navigate to the mpl_animation directory and run an example
```
python examples line_example.py
```
4. Otherwise you need to add the mpl_animation directory to somewhere on your pythonpath

## Saving to disk as mp4 file
To save a movie to disk as an mp4 file you add the keyword argument anim_save='movie_name.mp4' to 
the anim.animate() function.  In this case the movie will not be shown on the screen.  You need to have ffmpeg to save movies to disk.  Anaconda users may be able to do this with
```
conda install -c menpo ffmpeg
```

