import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import disk, rectangle
from scipy.signal import convolve2d
from pathlib import Path
from os.path import sep
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

def generateSingleSpotPan(dotSizeAng, relDotpos, panHeight_px, panWidth_px):
    
    #convert angular dot size to pixel size and make dot
    dotRad = panWidth_px * 0.5 * (dotSizeAng/360)
    mydot = disk(radius=dotRad)
    
    dotpos_x = round(relDotpos[0])
    dotpos_y = round(relDotpos[1])
    
    pan = np.zeros((panWidth_px,panHeight_px),dtype=np.uint8)
    pan[dotpos_x, dotpos_y] = maxBit
    pan[:,:] = convolve2d(pan[:,:], mydot, boundary = 'wrap', mode='same')

    pan = np.minimum(maxBit, pan)

    return pan

def plotPanorama(pan,panHeight_px,panWidth_px, maxBit, fig=None, return_img=False):
    if fig is None: fig = plt.figure(frameon=False)
    fig.set_size_inches(10,10*panHeight_px/panWidth_px)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    img = ax.imshow(pan.T,origin='lower', cmap='binary', vmin=0, vmax=maxBit)
    if return_img:
        return fig, img
    else:
        return fig

# stretch to length of the screen and correct for distortion
def plotPanoramaStretch(pan,panHeight,panWidth, alphas, maxBit):

    X = np.linspace(0,panWidth, pan.shape[0])
    Y = np.tan(alphas)

    fig = plt.figure(frameon=False)
    fig.set_size_inches(10,10*panHeight/panWidth)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.pcolormesh(X, Y,pan.T,cmap='binary', vmin=0, vmax=maxBit)
    return fig

def createMovie(movieStack,panHeight_px,panWidth_px,maxBit):

    Nframes = movieStack.shape[0]
    fig,img = plotPanorama(movieStack[0,:,:],panHeight_px,panWidth_px, maxBit,return_img=True)

    def animate(t):
        img.set_array(movieStack[t,:,:].T) #transpose because of how plotPanorama is written
        return img, 

    anim = FuncAnimation(
        fig,
        animate,
        frames = np.linspace(0,Nframes-1,np.min([100,Nframes])).astype('int'),
        blit = False
    )

    plt.show()

    return anim