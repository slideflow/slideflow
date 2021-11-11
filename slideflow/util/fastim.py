from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import imageio
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

Image.MAX_IMAGE_PIXELS = 100000000000

'''
Fast Plotter for Large Images - Resamples Images to a target resolution on each zoom.
'''
class FastImshow:
    '''Fast plotter for large image buffers

    Example::
        sz = (10000,20000) # rows, cols
        buf = np.arange(sz[0]*sz[1]).reshape(sz)
        extent = (100,150,1000,2000)  # arbitrary extent
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        im = FastImshow(buf,ax,extent=extent,tgt_res=1024)
        im.show()
        plt.show()
    '''
    def __init__(self,buf,ax,extent=None,tgt_res=512):
        '''
        [in] img buffer
        [in] extent
        [in] axis to plot on
        [in] tgt_res(default=512) : target resolution
        '''
        self.buf = buf
        self.sz = self.buf.shape
        self.tgt_res = tgt_res
        self.ax = ax

        # Members required to account for mapping extent to buf coordinates
        if extent:
            self.extent = extent
        else:
            self.extent = [ 0, self.sz[1], 0, self.sz[0] ]
        self.startx = self.extent[0]
        self.starty = self.extent[2]
        self.dx = self.sz[1] / (self.extent[1] - self.startx ) # extent dx
        self.dy = self.sz[0] / (self.extent[3] - self.starty ) # extent dy

    def get_strides( self,xstart=0, xend=-1, ystart=0, yend=-1, tgt_res=512 ):
        '''
        Get sampling strides for a given bounding region. If none is provided,
           use the full buffer size
        '''
        # size = (rows,columns)
        if xend == -1:
            xend = self.sz[1]
        if yend == -1:
            yend = self.sz[0]
        if (xend-xstart) <= self.tgt_res:
            stridex = 1
        else:
            stridex = max(int((xend - xstart) / self.tgt_res),1)

        if (yend-ystart) <= self.tgt_res:
            stridey = 1
        else:
            stridey = max(int((yend - ystart) / self.tgt_res),1)

        return stridex,stridey

    def ax_update(self, ax):
        '''
        Event handler for re-plotting on zoom
        - gets bounds in img extent coordinates
        - converts to buffer coordinates
        - calculates appropriate strides
        - sets new data in the axis
        '''
        ax.set_autoscale_on(False)  # Otherwise, infinite loop

        # Get the range for the new area
        xstart, ystart, xdelta, ydelta = ax.viewLim.bounds
        xend = xstart + xdelta
        yend = ystart + ydelta

        xbin_start = int(self.dx * ( xstart - self.startx ))
        xbin_end   = int(self.dx * ( xend - self.startx ))
        ybin_start = int(self.dy * ( ystart - self.starty ))
        ybin_end   = int(self.dy * ( yend - self.starty ))

        # Update the image object with our new data and extent
        im = ax.images[-1]

        stridex,stridey = self.get_strides( xbin_start,xbin_end,ybin_start,ybin_end)

        im.set_data( self.buf[ybin_start:ybin_end:stridey,xbin_start:xbin_end:stridex] )

        im.set_extent((xstart, xend, ystart, yend))

        ax.figure.canvas.draw_idle()

    def show(self):
        '''
        Initial plotter for buffer
        '''
        stridex, stridey = self.get_strides()
        self.ax.imshow(self.buf[::stridex,::stridey],extent=self.extent,origin='upper', aspect='equal')#self.ratio)
        self.ax.figure.canvas.draw_idle()

        self.ax.callbacks.connect('xlim_changed', self.ax_update)
        self.ax.callbacks.connect('ylim_changed', self.ax_update)