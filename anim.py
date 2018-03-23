# See mpl_animation/examples for example use cases
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from copy import copy
import numpy as np

def set_titles(titles, t_len):
    """Set the title based on the type of input
       Inputs:
       titles - can be list, numpy array, string or None
       tlen - integer, length of time axis
    """
    if isinstance(titles,(list, np.ndarray)):
        # If titles is already a sequence, pass it through
        pass
    elif isinstance(titles,str):
        # For a constant in time title, replicate it to be the length
        # of the time axis
        titles = [titles for i in np.arange(t_len)]
    else: 
        # If title is None then create an integer list for titles
        # with the length of the time axis
        titles = np.arange(t_len)
    return titles

def plot(*args, axkwargs = None, titles = None,**kwargs):
    """Create object for animating a line plot.
    
    Inputs:
    Call signatures
    create_line_object(x, y)        # plot x and y using default line style and color
    create_line_object(x, y, 'bo')  # plot x and y using blue circle markers
    create_line_object(y)           # plot y using x as index array 0..N-1
    create_line_object(y, 'r+')     # ditto, but with red plusses

    x is x-axis data, shape = (N, time_length) or (N,)
    y is y-axis data, shape = (N, time_length) or (N,)
    One of x or y has to be two-dimensional!
    
    axkwargs is a dictionary with keyword arguments for plot axis, e.g xlabel, or ylim
    titles is a list of frame titles - see set_titles function above
    **kwargs are the keyword arguments to plot.plot() e.g. label = 'data' or lw =2
    
    Outputs:
    Single element list where element has line_object type
    """
    if axkwargs:
         assert type(axkwargs) is dict, 'axkwargs must be a dictionary, but is {}'.format(type(axkwargs))
    if len(args) == 1:
        y = args[0]
        x = np.arange(len(y))
        line_style = ''
    elif len(args) == 2 and isinstance(args[1],np.ndarray):
        x,y = args
        line_style = ''
    elif len(args) == 2 and isinstance(args[1],'str'):
        y,line_style = args
        x = np.arange(len(y))
    elif len(args) == 3:
        x,y,line_style = args

    assert max((len(x.shape),len(y.shape))) < 3, "x has shape {} and y has shape {}, but neither should have more than two dimensions".format(x.shape,y.shape)
    assert len(x.shape) ==2 or len(y.shape) ==2,"""One of x or y must be 2D but x has shape {} and y has shape {}""".format(x.shape,y.shape)
    assert isinstance(line_style,str),"Line style must be a string but has type {}".format(type(line_style))
    if axkwargs:
        # If the xlim or ylim is not specified
        if 'xlim' not in axkwargs:
            axkwargs['xlim'] = (x.min(), x.max())
        if 'ylim' not in axkwargs:
            axkwargs['ylim'] = (y.min(), y.max())
    else:
        axkwargs = {}
        axkwargs['xlim'] = (x.min(), x.max())
        axkwargs['ylim'] = (y.min(), y.max())

    class line_object(object):
        """Create the line_object for the animation"""

        def __init__(self,x, y, line_style, axkwargs, titles, kwargs):
            self.__dict__.update({k: v for k, v in list(locals().items())
            if k != 'self'})
            self.kwargs = kwargs
            self.plot_type = 'line'
            # Set limits of the line plot
            self.xlim = axkwargs['xlim']
            self.ylim = axkwargs['ylim']
            # Repeat for every frame
            if len(x.shape) == 1 and len(y.shape) == 2:
                """Repeat x for every frame"""
                self.x = np.tile(x, (y.shape[1], 1)).T
            if len(x.shape) == 2 and len(y.shape) == 1:
                """Repeat y for every frame"""
                self.y = np.tile(y, (x.shape[1], 1)).T
            self.t_len = self.y.shape[-1] #T length
            self.titles = set_titles(titles, self.t_len)
    # Create the animation object
    anim_object = line_object(x,y,line_style, axkwargs, titles, kwargs)
    return [anim_object]

def pcolormesh(*args, 
                       clims = [], cmap = plt.cm.RdBu_r,
                       titles = None,
                       cmap_bad = None, cb_label = None, 
                       axkwargs = None,  **kwargs):
    """Create an object for animating a pcolormesh plot
    
    Inputs:
    Call signatures:
        create_quad_object(C)
        create_quad_object(x, y, C)   
    C is the color field, a np.ndarray with shape (ylen, xlen, tlen)
    x is the x axis, a np.ndarray with shape (xlen) or shape (ylen, xlen)
    y is the y axis, a np.ndarray with shape (ylen) or shape (ylen, xlen)
    
    clims is a two element list with the colourbar limits
    titles is a list of frame titles - see set_titles function above
    cmap_bad sets out how to colour regions outside the colour limits
    cb_label is a string for the colorbar label
    axkwargs is a dictionary with keyword arguments for plot axis, e.g xlabel, or ylim
    **kwargs are keyword arguments for pcolormesh
    """
    if axkwargs:
        assert type(axkwargs) is dict, 'axkwargs must be a dictionary, but is {}'.format(type(axkwargs))
    if cb_label:
         assert type(cb_label) is str, 'cb_label must be a string, but is {}'.format(type(cb_label))
    assert 'vmin' not in kwargs and 'vmax' not in kwargs, "Specify color limits as a two element list with clims = [,] instead of with vmin and vmax"

    #Test for the right number of input arrays in *args
    assert len(args) == 1 or len(args) == 3,'args can have length 1 or 3 but has length {}'.format(len(args))
    C = args[-1]
    if len(C.shape) > 3:
        C = np.squeeze(C)
        assert len(C.shape) == 3, "C does is not 3-dimensional, even with squeezing"
    if len(args) == 1:
        x = np.arange(C.shape[1])
        y = np.arange(C.shape[0])
    else:
        x,y = args[:2]
        # Remove possible singleton dimensions
        x = np.squeeze(x)
        y = np.squeeze(y)
        if len(x.shape) > 1: 
            # If x and y are 2d coordinate arrays
            assert x.shape == y.shape,\
            "Axis arrays need to have same shape, but x.shape = {} and y.shape = {}".format(x.shape,y.shape)
            assert x.shape == C.shape[:2],\
            "Axis arrays need to have same shape, but x.shape = {} and C.shape = {}".format(x.shape,z.shape)
        else: 
            #If x and y are 1d coordinate arrays
            assert (len(y),len(x)) == C.shape[:2],\
        "C has shape {}, but y has shape {} and x has shape {}".format(C.shape,len(y),len(x))         

    if not clims:
        kwargs['vmin'] = C.min()
        kwargs['vmax'] = C.max()
    else:
        kwargs['vmin'] = clims[0]
        kwargs['vmax'] = clims[1]
    
    # Copy the matplotlib colormap to allow NaN values to be masked
    kwargs['cmap'] = copy(cmap)
    C = np.ma.array(C, mask=np.isnan(C))

    if not cmap_bad:
        kwargs['cmap'].set_bad('gray',1.)
    else:
        kwargs['cmap'].set_bad(cmap_bad, 1.)
    if axkwargs:
        if 'xlim' not in axkwargs:
            axkwargs['xlim'] = (x.min(), x.max())
        if 'ylim' not in axkwargs:
            axkwargs['ylim'] = (y.min(), y.max())
    else:
        axkwargs = {'xlim':(x.min(), x.max()),'ylim':(y.min(), y.max())}
    class quad_object(object):

        def __init__(self, C, x, y, titles, axkwargs, cb_label, kwargs):
            self.__dict__.update({k: v for k, v in list(locals().items())
            if k != 'self'})
            self.plot_type = 'quad'
            self.t_len = C.shape[-1]
            self.titles = set_titles(titles, self.t_len)

    anim_object = quad_object(C, x, y, titles, axkwargs, cb_label, kwargs)
    return [anim_object]

def scatter(x,y,scatter_color = None, axkwargs = None,
titles = None,  **kwargs):
    """Create an object for animating a scatter plot
    
    Inputs:
    x - np.ndarray with shape (N, time_length), where N is the number of samples.
    y - np.ndarray with shape (N, time_length), where N is the number of samples.
    scatter_color - can be:
                    1) an array/list with N elements and sets
    axkwargs is a dictionary with keyword arguments for plot axis, e.g xlabel, or ylim
    **kwargs are keyword arguments for plt.scatter
    """
    if axkwargs:
         assert type(axkwargs) is dict, 'axkwargs must be a dictionary, but is {}'.format(type(axkwargs))
    assert x.shape == y.shape, "x and y need to be the same shape but x has shape {} and y has shape {}".format(x.shape,y.shape)
    assert len(x.shape) == 2, "x and y need to be 2 dimensional but are shape {}".format(x.shape)
    # We need the colour argument to be an array/list that is the same length as
    # the time axis to allow colours to change in time. If the colour is a constant in time
    # sting for all points then we turn it into a list.
    if 'c' in kwargs:
        if isinstance(kwargs['c'], (str)):
            scatter_color = [kwargs['c'] for i in range(x.shape[1])]
        if isinstance(kwargs['c'],list):
            scatter_color = kwargs['c']
            assert len(kwargs['c']) == x.shape[1],"List must be same length as time axis"
        if isinstance(kwargs['c'],np.ndarray):
            scatter_color = [kwargs['c'][:,i] for i in range(x.shape[1])]
            print(len(scatter_color),scatter_color[0].shape)
        kwargs.pop('c')
    else:
        scatter_color = ['b' for i in range(x.shape[1])]
    class scatter_object(object):

        def __init__(self, x, y, scatter_color, titles, axkwargs, kwargs):
            self.__dict__.update({k: v for k, v in list(locals().items())
            if k != 'self'})
            self.t_len = x.shape[1]
            self.plot_type = 'scat'
            self.xlim = (self.x.min(), self.x.max())
            self.ylim = (self.y.min(), self.y.max())
            self.titles = set_titles(titles, self.t_len)

    anim_object = scatter_object(x, y, scatter_color, titles, axkwargs, kwargs)
    return [anim_object]

def contour(*args,**kwargs):
    """Create an object that can be passed to the animation routine
    z is the color field with shape (ylen, xlen, tlen)
    x is the x axis with shape (xlen) or shape (ylen, xlen)
    y is the y axis with shape (ylen) or shape (ylen, xlen)
    clims is a two element list with the colour limits
    titles is a list with titles for each frame
    cmap_bad sets out how to colour regions outside the colour limits
    axkwargs is a dictionary with keyword arguments for plot axis, e.g xlabel, or ylim
    *args are positional arguments for pcolormesh
    **kwargs are keyword arguments for pcolormesh
    """

    class cont_object(object):

        def __init__(self, args, kwargs):
            self.__dict__.update({k: v for k, v in list(locals().items())
            if k != 'self'})
            self.plot_type = 'contour'
            self.t_len = None

    anim_object = cont_object(args, kwargs)
    return [anim_object]


def animate(anim_objects, fps = 25, anim_save = None, bitrate = 1800,test=False, **kwargs):
    """Main animation routine
    Inputs:
    anim_objects - can be (see examples below):
                   1) a single object created by e.g. create_line_object
                   2) a list of these objects: this overlays the objects on the same plot
                   3) a list of lists of these objects. Each list is a subplot
    fps - frames per second (integer)
    anim_save - either None or a string. If a string this is the name of the saved file
    e.g. anim_save = 'test.mp4'
    bitrate - integer
    kwargs are the arguments to plt.subplots e.g. nrows,ncols,figsize
    
    Examples
    line = anim.create_line_object(x,y)
    quad = anim.create_quad_object(Z,X,Y)
    Single plot:
    animate(line)
    Separate subplots in a row:
    animate(line + quad)
    Separate subplots in a column:
    animate(line + quad,nrows=2,ncols=1])
    Separate subplots in a grid:
    animate(line + quad + line + quad,nrows=2,ncols=2])
    """
    plt.close('all')
    if anim_save:
        show = False
    else:
        show = True
    #Ensure each element of anim_objects is a list itself
    anim_objects = [ [i] if not isinstance(i,list) else i for i in anim_objects]
    #Specify subplot structure
    if 'nrows' not in kwargs:
        kwargs['nrows'] = 1
    if 'ncols' not in kwargs:
        kwargs['ncols'] = int(len(anim_objects)/kwargs['nrows']) #Default to all subplots in a row

    number_of_subplots = np.product( kwargs['nrows'] * kwargs['ncols'] )
    assert number_of_subplots == len(anim_objects),'{} subplots but {} anim_objects'.format(number_of_subplots,len(anim_objects))
    #Need to check t_lens are the same for each subplot
    t_len_list = []
    for lst in anim_objects:
        for obj in lst:
            if obj.plot_type != 'contour':
                t_len_list.append(obj.t_len)
    assert not np.any(np.array(t_len_list) - t_len_list[0]), 'time lengths are not the same'
    fig, axes = plt.subplots(**kwargs)

    class update_plot(object):

        def __init__(self, axes, anim_objects):
            self.__dict__.update({k: v for k, v in list(locals().items())
            if k != 'self'})
            self.plot_objects = []
            self.plot_types = []
            self.x_limits = []
            self.y_limits = []
            if not isinstance(self.axes, np.ndarray): #For a single subplot
                self.axes = np.array([self.axes])
            for idx, ax in enumerate(self.axes.flatten()):
                self.plot_types.append([])
                self.plot_objects.append([])

                for odx,obj in enumerate(self.anim_objects[idx]):
                    self.plot_types[idx].append(obj.plot_type)#Has same indexing as self.plot_objects
                    if obj.plot_type == 'quad':
                        self.plot_objects[idx].append(
                        ax.pcolormesh(obj.x, obj.y, obj.C[:,:,0], **obj.kwargs) )
                        cb = plt.colorbar(self.plot_objects[idx][odx], ax = ax)
                        if hasattr(obj, 'cb_label'):
                            cb.set_label(obj.cb_label)
                    elif obj.plot_type == 'scat':
                        self.plot_objects[idx].append(
                        ax.scatter(
                        obj.x[:,0],obj.y[:,0],c = obj.scatter_color[0],**obj.kwargs) )  
                    elif obj.plot_type == 'line':
                        self.plot_objects[idx].append(
                        ax.plot(obj.x[:,0], obj.y[:,0], obj.line_style, **obj.kwargs)[0] )
                        if 'label' in obj.kwargs:
                            ax.legend()
                    elif obj.plot_type == 'contour':
                        self.plot_objects[idx].append(
                        ax.contour(*obj.args, **obj.kwargs) )
                if obj.plot_type != 'contour':
                    if obj.axkwargs:
                        ax.set(**obj.axkwargs)

        def plot_init(self):
            for idx, ax in enumerate(self.axes.flatten()):
                # ax.set_xlim(self.x_limits[idx])
                # ax.set_ylim(self.y_limits[idx])
                for odx,obj in enumerate(self.plot_objects[idx]):
                    if self.plot_types[idx][odx] == 'quad':
                        obj.set_array(np.asarray([]))
                    elif self.plot_types[idx][odx] == 'scat':
                        self.plot_objects[idx][odx]
                    elif self.plot_types[idx][odx] == 'line':
                        obj.set_data([], [])


        def __call__(self, i):
            for idx, ax in enumerate(self.axes.flatten()):
                for odx,obj in enumerate(self.plot_objects[idx]):
                    if self.plot_types[idx][odx] != 'contour':
                        if self.plot_types[idx][odx] == 'quad':
                            obj.set_array(self.anim_objects[idx][odx].C[:-1,:-1, i].ravel())
                        elif self.plot_types[idx][odx] == 'scat':
                            obj.set_offsets(np.array((self.anim_objects[idx][odx].x[:,i],self.anim_objects[idx][odx].y[:,i])).T)
                            obj.set_color(self.anim_objects[idx][odx].scatter_color[i])
                        elif self.plot_types[idx][odx] == 'line':
                            obj.set_data(self.anim_objects[idx][odx].x[:,i],
                            self.anim_objects[idx][odx].y[:, i])
                        ax.set_title(self.anim_objects[idx][odx].titles[i])

            return self.plot_objects

    ud = update_plot(axes, anim_objects)

    anim = animation.FuncAnimation(fig, ud, init_func=ud.plot_init,
                                   frames=anim_objects[0][0].t_len,
                                   blit=False, interval = 1e3/fps)

    fig.tight_layout()
    if anim_save:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps = fps, bitrate=1800)
        anim.save(anim_save, writer = writer)
    if show:
        plt.show()


if __name__ == "__main__":
    # test code
    tlen = 100
    x = np.arange(200)
    y = np.arange(420)
    X,Y,T = np.meshgrid(x,y,np.arange(tlen))
    z = np.cos(np.pi*X/20)*np.sin(np.pi*Y/60)*np.cos(2*np.pi*T/50)
    quad_object = create_quad_object(z,x=x,y=y)
    
    r = np.arange(0,2*np.pi,1e-1)
    t = np.arange(0,4*np.pi,1e-1)
    T,R = np.meshgrid(t,r)
    s = np.cos(R - T)
#    line_object = create_line_object(
    scat = 2*np.ones((2,2))[:,:,np.newaxis]*np.arange(tlen)
    x = np.arange(0, 200, 1)
    [T,X] = np.meshgrid(np.arange(tlen),x)
    y = 2*np.arange(tlen)*np.cos((X + T/tlen)/10)
    #y = np.tile(np.arange(0,2,1e-1)[:,np.newaxis],(1,50))
    #scatter_object = scatter_anim(scat,c='r',alpha = 0.5, s = 200, scatter_color = y)
    #line_object = line_anim(x, y)
    #Create the plot list
    anim_list = [[quad_object], [quad_object]] #Each element is a subplot

    animate(anim_list)
