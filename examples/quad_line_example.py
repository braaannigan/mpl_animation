import anim
import numpy as np

# x-axis of plot
x = np.arange(0,2*np.pi,2e-1)
# y-axis of plot
y = np.arange(0,2*np.pi,2e-1)
# time-axis of plot
t = np.arange(0,2*np.pi,1e-1)
# Don't need meshgrid in anim.create_quad_object, 
# but use meshgrid here to help create the color data
Y,X,T = np.meshgrid(y,x,t,indexing='ij')

# color field to be plotted in subplot 1 - propagating cosine wave
cos_wave = (Y/(2*np.pi)) + (T/t[-1])*(
           np.cos(Y/(4*np.pi))**2*np.cos((X/(0.25*np.pi)) - T)
           ) + 0.25*np.random.standard_normal(X.shape)
# line field to be plotted in subplot 2 - propagating sine wave
meridional_mean = np.mean(cos_wave,axis=0)

# Create titles for each subplot
quad_titles = ['Wave at time {:.02f}'.format(i) for i in t]
line_titles = ['Meridional mean at time {:.02f}'.format(i) for i in t]

# Create axis keyword arguments
quad_axkwargs = {'xlabel':'x','ylabel':'y'}
line_axkwargs = {'xlabel':'x','ylabel':'Merid. Mean'}

# Create the animation objects
cos_quad = anim.pcolormesh(x,y,cos_wave,clims=[-2,2],
                                   titles = quad_titles,
                                   axkwargs = quad_axkwargs,
                                   cb_label='Energy'
                                  )

mean_line = anim.plot(Y[:,0,:],meridional_mean ,'r',
                                   titles = line_titles,
                                   axkwargs = line_axkwargs
                                  )

# Concatenate the objects to create 2 subplots in a row
anim_list = cos_quad + mean_line

# Animate the objects
anim.animate(anim_list, figsize=(10,7))#,anim_save='quad_line.mp4')