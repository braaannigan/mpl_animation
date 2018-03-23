import anim
import numpy as np
from numpy.testing import assert_array_equal

np.random.seed(123)
def test_set_titles_list():
    assert anim.set_titles('ab',3) == ['ab','ab','ab']

    
def set_line_plot_data():
    # x-axis of plot
    x = np.arange(0,2*np.pi,1e-1)
    # time-axis of plot
    t = np.arange(0,4*np.pi,1e-1)
    T,X = np.meshgrid(t,x)
    # field to be plotted - propagating wave
    s = np.cos(X - T)
    return x,t,X,T,s

def standard_line_tests(line_obj):
    x,t,X,T,s = set_line_plot_data()
    assert_array_equal(X,line_obj.x)
    assert_array_equal(s,line_obj.y)
    assert line_obj.t_len == len(t)
    assert line_obj.plot_type == 'line'

def test_plot():
    """Test line object with no extra arguments"""
    x,t,X,T,s = set_line_plot_data()
    line_obj = anim.plot(x, s)[0]
    standard_line_tests(line_obj)
    assert_array_equal(np.arange(line_obj.t_len),line_obj.titles)
    assert line_obj.xlim == (X.min(),X.max())
    assert line_obj.ylim == (s.min(),s.max())
    assert line_obj.kwargs == {}
    assert list(line_obj.axkwargs) == ['xlim','ylim']
    assert line_obj.axkwargs['xlim'] == line_obj.xlim
    assert line_obj.axkwargs['ylim'] == line_obj.ylim


def test_create_line_object_axkwargs():
    """Test line object with axis limit arguments"""
    xlim = [1,3]
    ylim = [-2,2]
    xlabel = 'x'
    ylabel = 'data'
    x,t,X,T,s = set_line_plot_data()
    axkwargs = {'xlim':tuple(xlim),'ylim':tuple(ylim),
                 'xlabel':xlabel,'ylabel':ylabel}
    line_obj = anim.plot(x, s,axkwargs=axkwargs)[0]
    standard_line_tests(line_obj)
    assert_array_equal(np.arange(line_obj.t_len),line_obj.titles)
    assert line_obj.xlim == tuple(xlim)
    assert line_obj.ylim == tuple(ylim)
    assert line_obj.kwargs == {}
    assert list(line_obj.axkwargs) == ['xlim','ylim','xlabel','ylabel']

def test_create_line_object_titles():
    """Test line object with specified titles"""
    x,t,X,T,s = set_line_plot_data()
    titles = ['data at time {}'.format(i) for i in np.arange(len(t))]
    line_obj = anim.plot(x, s, titles = titles)[0]
    standard_line_tests(line_obj)
    assert titles == line_obj.titles
    assert line_obj.xlim == (X.min(),X.max())
    assert line_obj.ylim == (s.min(),s.max())
    assert line_obj.kwargs == {}
    assert list(line_obj.axkwargs) == ['xlim','ylim']
    assert line_obj.axkwargs['xlim'] == line_obj.xlim
    assert line_obj.axkwargs['ylim'] == line_obj.ylim

def set_quad_plot_data():
    # x-axis of plot
    x = np.arange(0,4*np.pi,1e-1)
    # y-axis of plot
    y = np.arange(0,2*np.pi,1e-1)
    # time-axis of plot
    t = np.arange(0,1*np.pi,1e-1)
    Y,X,T = np.meshgrid(y,x,t,indexing='ij')
    # field to be plotted - propagating wave
    s = np.cos(X - T)
    return x,y,t,X,Y,T,s

def standard_quad_tests(quad_obj):
    x,y,t,X,Y,T,s = set_quad_plot_data()
    assert_array_equal(x,quad_obj.x)
    assert_array_equal(y,quad_obj.y)
    assert_array_equal(s,quad_obj.C)
    assert quad_obj.t_len == len(t)
    assert quad_obj.plot_type == 'quad'

def test_create_quad_object():
    """Test quad object with no extra arguments"""
    x,y,t,X,Y,T,s = set_quad_plot_data()
    quad_obj = anim.pcolormesh(x,y,s)[0]
    standard_quad_tests(quad_obj)
    assert_array_equal(np.arange(quad_obj.t_len),quad_obj.titles)
    assert list(quad_obj.kwargs) == ['vmin', 'vmax', 'cmap']
    assert list(quad_obj.axkwargs) == ['xlim','ylim']
    assert quad_obj.axkwargs['xlim'] == (X.min(),X.max())
    assert quad_obj.axkwargs['ylim'] == (Y.min(),Y.max())
    
def test_create_quad_object_axkwargs():
    xlim = [1,3]
    ylim = [-2,2]
    xlabel = 'x'
    ylabel = 'data'
    axkwargs = {'xlim':tuple(xlim),'ylim':tuple(ylim),
                 'xlabel':xlabel,'ylabel':ylabel}
    x,y,t,X,Y,T,s = set_quad_plot_data()
    titles = ['data at time {}'.format(i) for i in np.arange(len(t))]
    quad_obj = anim.pcolormesh(x,y,s,axkwargs=axkwargs,titles=titles)[0]
    standard_quad_tests(quad_obj)
    assert titles == quad_obj.titles
    assert list(quad_obj.kwargs) == ['vmin', 'vmax', 'cmap']
    assert list(quad_obj.axkwargs) == ['xlim','ylim','xlabel','ylabel']

def test_create_quad_object_clims_cblabel():
    clims = [0,1]
    x,y,t,X,Y,T,s = set_quad_plot_data()
    quad_obj = anim.pcolormesh(x,y,s,clims=clims,cb_label='aaa')[0]
    standard_quad_tests(quad_obj)
    assert quad_obj.kwargs['vmin'] == clims[0]
    assert quad_obj.kwargs['vmax'] == clims[1]
    assert quad_obj.cb_label == 'aaa'
    assert list(quad_obj.kwargs) == ['vmin', 'vmax', 'cmap']

def set_scatter_plot_data():
    np.random.seed(123)
    scatter_data = np.zeros((10,2,30))
    for i in np.arange(1,30):
        scatter_data[:,:,i] = 0.5*(
            scatter_data[:,:,i-1] +
            np.random.multivariate_normal([0,0],[[1,0.5],[0.5,1]],size=10)
        )
    return scatter_data[:,0,:],scatter_data[:,1,:]

def standard_scatter_tests(scatter_obj):
    x,y = set_scatter_plot_data()
    assert_array_equal(x,scatter_obj.x)
    assert_array_equal(y,scatter_obj.y)
    assert scatter_obj.t_len == 30
    assert scatter_obj.plot_type == 'scat'

def test_create_scatter_object():
    """Test scatter object with no extra arguments"""
    x,y = set_scatter_plot_data()
    axkwargs={'xlim':(-5e-1,5e-1),'ylim':(-5e-1,5e-1)}
    scatter_obj = anim.scatter(x,y,axkwargs=axkwargs)[0]
    standard_scatter_tests(scatter_obj)
    assert_array_equal(np.arange(scatter_obj.t_len),scatter_obj.titles)
