import anim
import numpy as np

# Create the coordinate arrays
# Angle
theta = np.linspace(0,2*np.pi,30)
# Time axis
t = np.arange(0,2*np.pi,0.02*np.pi)

# For this simple example loop through the array to create the output
scatter_data = np.empty((len(theta),2,len(t)))
# Create the spiral
for i in np.arange(len(t)):
    scatter_data[:,0,i] = (t[i]/t.max())*np.cos(theta + t[i])
    scatter_data[:,1,i] = (t[i]/t.max())*np.sin(theta + t[i])

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
titles = ['Time {:.02f}'.format(i) for i in t]
# Specify the size of the scatter points
point_size = 3*np.arange(len(t))
# Axis keyword arguments
axkwargs = {'xlim':(-1.1,1.1),'ylim':(-1.1,1.1),
            'xlabel':'X','ylabel':'Y'}

# Create the single-element list with the scatter object
scat_obj = anim.create_scatter_object(scatter_data,point_size, 
                                      axkwargs = axkwargs,scatter_color=scatter_color,
                                      titles = titles)

# This step isn't interesting here because there's only one subplot
anim_list = scat_obj
# Create the animation - add the anim_save option to write to disk instead of showing in 
# a window
anim.animate(anim_list,figsize=(10,10), anim_save = 'scatter.mp4')