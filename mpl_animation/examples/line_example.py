import anim
import numpy as np

# x-axis of plot
x = np.arange(0,2*np.pi,1e-1)
# time-axis of plot
t = np.arange(0,8*np.pi,1e-1)

# Use meshgrid to help create the data to be plotted - but
# can also give 1D coordinate arrays to create_line_object
T,X = np.meshgrid(t,x)
# field to be plotted in subplot 1 - propagating cosine wave
cos_wave = (t/t.max())*np.cos(X - T)
# field to be plotted in subplot 2 - propagating sin wave
sin_wave = (t/t.max())*np.sin(X - T)

# Create a title
titles = ['Wave at time {:.02f}'.format(i) for i in t]
# Create the animation objects
cos_line = anim.plot(x, cos_wave, titles = titles)
sin_line = anim.plot(x, sin_wave,'r', titles = titles)

# Concatenate the objects
anim_list = cos_line + sin_line

# Animate the objects
anim.animate(anim_list,figsize=(10,7))#,anim_save='line.mp4', )