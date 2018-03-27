import mpl_animation.anim as anim
import numpy as np

# Create the coordinate arrays
# Angle
theta = np.linspace(0,2*np.pi,30)
# Time axis
t = np.arange(0,2*np.pi,0.1*np.pi)

# For this simple example loop through the array to create the output
x = np.empty((len(theta),len(t)))
y = np.empty_like(x)
# Create the spiral
for i in np.arange(len(t)):
    x[:,i] = (t[i]/t.max())*np.cos(theta + t[i])
    y[:,i] = (t[i]/t.max())*np.sin(theta + t[i])

# Set the color to change through the animation
scatter_color = []
for i in range(len(t)):
    if i<0.33*len(t):
        scatter_color.append('r')
    elif i<0.67*len(t):
        scatter_color.append('g')
    else:
        scatter_color.append('b')
# Create the titles list that specifies the time
titles = ['' for i in t]
# Specify the size of the scatter points
point_size = 5*np.arange(len(t))
# Axis keyword arguments
axkwargs = {'xlim':(-1.1,1.1),'ylim':(-1.1,1.1),
            'xlabel':'X','ylabel':'Y'}

# Create the single-element list with the scatter object
scat_obj = anim.scatter(x,y,s=point_size, titles = titles,
                                      axkwargs = axkwargs,c=scatter_color)

# This step isn't interesting here because there's only one subplot
anim_list = scat_obj
# Create the animation with 10 frames per second - add the anim_save option to write to disk instead of showing in 
# a window
anim.animate(anim_list,figsize=(8,6),fps = 10, anim_save = 'scatter.mp4')