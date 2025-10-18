from astropy.coordinates import cartesian_to_spherical
from skimage.draw import polygon
from skimage.color import rgb2gray
from scipy. io import loadmat 
from skimage.exposure import equalize_hist
from skimage.transform import downscale_local_mean
import skimage as ski
import skimage.filters
import matplotlib.patches as mpatches
import xarray as xr
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython import display
from sklearn import datasets
from sklearn.decomposition import PCA


#### ANGULAR CORRECTIONS AND COORDINATE TRANSFORMATIONS
def angleCorrect(Lon, theta):
    LonNorm = p2p(theta-Lon)
    ind = (np.max(LonNorm,axis=1) - np.min(LonNorm,axis=1))<np.pi
    LonNormChange = LonNorm[~ind,:]
    LonNormChange[LonNormChange<=0] += (2*np.pi)
    LonNormChange[LonNormChange>0] -= (2*np.pi)
    LonNorm[~ind,:] = LonNormChange
    return LonNorm

def p2p(t):
    tn = np.mod(t,2*np.pi)
    tn = tn - ((tn>np.pi)*(2*np.pi))
    return tn

def cart2sph(x,y,z):
    azimuth = np.arctan2(y,x)
    elevation = np.arctan2(z,np.sqrt(x**2 + y**2))
    r = np.sqrt(x**2 + y**2 + z**2)
    return r, elevation, azimuth

def angle_to_scene(lat, lon, down, hfov, res):
    c = (lon-(-hfov/2))/res
    r = (lat-(down))/res
    return r.astype('int'),c.astype('int')

def extractWorld(filename, fix = True):
    matWorld = loadmat(filename)
    world = np.array([matWorld['X'], matWorld['Y'], matWorld['Z'], matWorld['colp']])
    if fix:
        world[2,:,2][world[2,:,2]<0] = -world[2,:,2][world[2,:,2]<0] #flip negative Z values of the 3rd vertex
        world[3,:,:] = (1-world[3,:,:]) #flip green channel
    
    dWorld = xr.DataArray(data = world, coords = {'vars':['X', 'Y', 'Z', 'colp'],'polygons':range(world.shape[1]),'points':range(world.shape[2])})
    return dWorld

def extractRoutes(filename, zAnt):
    matRoutes = loadmat(filename)
    routeNames = list(matRoutes.keys())[3:]
    N = np.max([matRoutes[routeNames[i]].shape[0] for i in range(len(routeNames))]) #maximum length of routes

    height = zAnt*np.ones(N)

    #pad end of routes with nans for xarray 
    for r in range(len(routeNames)):
        p = N - len(matRoutes[routeNames[r]])
        matRoutes[routeNames[r]] = np.append(matRoutes[routeNames[r]],np.array([float('NaN')]*(3*p)).reshape(p,3),0) 
        matRoutes[routeNames[r]] = np.insert(matRoutes[routeNames[r]],2,height,axis=1)
    
    routes = np.array([matRoutes[n][:] for n in routeNames])
    routes[:,:,:2] = routes[:,:,:2]/100 #convert from cm to m
    routes[:,:,-1] = routes[:,:,-1]*np.pi/180 #convert from deg to rad
    dRoutes = xr.DataArray(data = routes, coords = {'routes': routeNames, 'frames': range(routes.shape[1]), 'vars': ['X', 'Y', 'Z', 'th']})
    return dRoutes, N

def top_view(rt, frame, visibility, dWorld, dRoutes, hfovDeg=296, plotOrigin = True, plotWorld=True, plotFov=True, plotRoute=True, fig = None, ax = None,
            alpha=0.1, color='k'):
    
    if (fig is None) and (ax is None):
        fig,ax = plt.subplots(figsize = (5,5))
    
    if plotWorld:
        for p in dWorld['polygons'].values:
            plt.fill(dWorld.loc['X',p,:].values,dWorld.loc['Y',p,:].values,color=dWorld.loc['colp',p,:].values);
            
    if plotRoute:
        plt.plot(dRoutes.loc[rt,:,'X'], dRoutes.loc[rt,:,'Y'],'.',color=color,alpha=alpha,markersize=5);
    
    if plotOrigin:
        plt.plot(dRoutes.loc[rt,0,'X'], dRoutes.loc[rt,0,'Y'],'o',color='seagreen');
    
    if plotFov:
        plt.plot(dRoutes.loc[rt,frame,'X'], dRoutes.loc[rt,frame,'Y'],'ro');
        wedge = mpatches.Wedge((dRoutes.loc[rt,frame,'X'], dRoutes.loc[rt,frame,'Y']
                            ),visibility,dRoutes.loc[rt,frame,'th']*180/np.pi-(hfovDeg/2)
                            ,dRoutes.loc[rt,frame,'th']*180/np.pi+(hfovDeg/2),
                            color='r',alpha=0.5);
        ax.add_patch(wedge);
        
    plt.axis('equal')
    return fig, ax

def getScene(rt, frame, dWorld, dRoutes):
    ant = dRoutes.loc[rt,frame,:]
    dScene = dWorld.loc[['X','Y','Z'],:,:]-ant
    return dScene, ant 

def getColor(dWorld):
    return rgb2gray(dWorld.loc['colp',:,:])

def getSphScene(dScene, ant, visibility):
    r,Lat,Lon0 = np.array(cart2sph(dScene.loc['X'],dScene.loc['Y'],dScene.loc['Z']))
    
    #converting to the right angular values
    Lon = angleCorrect(Lon0, ant.loc['th'].values)
    
    #create dataarray
    dSphScene = xr.DataArray(data = np.array([r,Lat,Lon]), coords = {'vars':['r', 'Lat', 'Lon'],'polygons':range(Lon.shape[0]),'points':range(Lon.shape[1])})
    
    #remove objects that are beyond visibility and then sort in descending order of distance
    dSphScene = dSphScene.loc[:,(dSphScene.loc['r',:,:].min(axis=1).values)<=visibility,:].sortby(dSphScene.loc['r',:,:].min(axis=1),ascending=False) 
    
    return dSphScene

def generateScene(rt, frame, visibility, dWorld, dRoutes, hfovDeg, resDeg, upDeg, downDeg, zAnt, invert=True, equalize=True, normalize=True):
    hfov = hfovDeg*np.pi/180
    up = upDeg*np.pi/180
    down = downDeg*np.pi/180
    res = resDeg*np.pi/180
    vfov = up-down
    
    scene = np.zeros((int(hfov/res), int(vfov/res))).T
    
    #check if the ant locations are nan
    if (np.isnan(dRoutes.loc[rt,frame,:]).sum()==0):
        
        #plot ground and sky
        groundcolor = rgb2gray(np.array([229, 183, 90])/255)
        r,_ = angle_to_scene(np.array([0+np.arctan2(-zAnt,10)]),np.array([0]),down,hfov,res)
        scene[:int(r),:] = groundcolor #grey ground
        scene[int(r):,:] = 1 #bright sky

        #iteratively plot grass blades

        #get color of grasses
        PolCol = getColor(dWorld)

        #get dSpScene
        dSphScene = getSphScene(*getScene(rt, frame, dWorld, dRoutes),visibility)

        for p in dSphScene['polygons'].values:
            r, c = angle_to_scene(dSphScene.loc['Lat',p,:], dSphScene.loc['Lon',p,:], down, hfov, res)
            rr,cc = polygon(r,c,shape=np.shape(scene))
            scene[rr, cc] = PolCol[p]

        if invert:
            scene = 1-scene
            
        #adaptive contrast normalize
        if equalize:
            scene = equalize_hist(scene)
        
    else:
        scene[:,:] = np.nan
    
    #rescale by local averaging
    scene = downscale_local_mean(scene, 2, 2)
    
    #normalize by sum of squared pixels
    if normalize: scene = scene/np.sqrt(np.sum(scene**2))
        
    return scene

def walkThrough(nframes, rt, snapshots, 
                visibility, dWorld, dRoutes, hfovDeg, resDeg, upDeg, downDeg, zAnt,
                animate=False, fps = 8, save=False, savename='sample-walkthrough_',plotFolder='plots/', **kwargs):
    frames = np.linspace(0,nframes-1,snapshots).astype('int')
    scene0 = generateScene(rt, frames[0], visibility, dWorld, dRoutes, hfovDeg, resDeg, upDeg, downDeg, zAnt, **kwargs)
    
    #generate empty scenes
    scenes = np.zeros([scene0.shape[0], scene0.shape[1], len(frames)])
        
    #skip the first scene
    for i,f in enumerate(frames[1:]):
        scenes[:,:,i+1] = generateScene(rt, f, visibility, dWorld, dRoutes, hfovDeg, resDeg, upDeg, downDeg, zAnt, **kwargs)
    
    #set scene0
    scenes[:,:,0] = scene0
        
    if animate:
        nanframes = np.isnan(scenes)[0,0,:]
        scenes_anim = scenes[:,:,~nanframes]
        
        # set up the figure, the axis, and the plot element we want to animate
        fig = plt.figure()
        im = plt.imshow(scene0, cmap='gray',origin='lower', vmin = 0, vmax=1) 

        anim = animation.FuncAnimation(
                                       fig, 
                                       animate_func, 
                                       frames = snapshots-np.sum(nanframes),
                                       interval = 1000 / fps, # in ms
                                       fargs = (fps,im,scenes_anim,)
                                       )

        video = anim.to_html5_video()
        html = display.HTML(video)
        display.display(html)
    
        if save:
            if isinstance(rt,str): savename = savename + rt
            anim.save(plotFolder+savename+'.mp4', fps=fps, extra_args=['-vcodec', 'libx264'])

        plt.close()
    
    return scenes, frames
        

def animate_func(i,fps,im,scenes):
    if (i % fps == 0):
        print( '.', end ='' )
    im.set_array(scenes[:,:,i])
    return [im]

def plot_PCA_sceneVecs(I, y = "r"):
    
    nanframes = np.isnan(I)[0]
    I = I[:,~nanframes]
    if not isinstance(y,str):
        y = y[~nanframes]

    fig = plt.figure(1, figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)

    X_reduced = PCA(n_components=3).fit_transform(I.T)
    ax.scatter(
        X_reduced[:, 0],
        X_reduced[:, 1],
        X_reduced[:, 2],
        c = y,
        edgecolor="k",
        s=40,
    )

    ax.set_xlabel("1st PC");
    ax.xaxis.set_ticklabels([]);
    ax.set_ylabel("2nd PC");
    ax.yaxis.set_ticklabels([]);
    ax.set_zlabel("3rd PC");
    ax.zaxis.set_ticklabels([]);
    return fig, ax

def sparse_images(picsize,n,sparsity=0.9,maxit=1000,N=5,a=1,b=25,beta=10,
                  show=False, seed = 0, sparsityDel = 0.05
                 ):
    np.random.seed(seed)
    X = np.zeros([picsize,picsize,n])
    q = 0
    #scale invariance:
    a = a*picsize/20
    b = b*picsize/20
    beta = beta*picsize/20
    if show:
        fig = plt.figure(figsize=(10,4))
        nrows, ncols = int(np.sqrt(n)), int(np.sqrt(n))+1
    for i in range(maxit):
        if q>=n:
            break
        if ((i==(maxit-1)) & (q<n)):
            print("increase maxit")
        pic = ((np.random.rand(picsize,picsize,N)-0.5)*2)*0.01
        cen=int(np.round(1.5*(a+b))); siz=(2*cen+1)
        gauss1 = np.zeros([siz,siz]); gauss1[cen,cen]=1.0; gauss1 = ski.filters.gaussian(gauss1,a)
        gauss2 = np.zeros([siz,siz]); gauss2[cen,cen]=1.0; gauss2 = ski.filters.gaussian(gauss2,b)
        for i in range(N-1):
            pic[:,:,i+1] = ski.filters.gaussian(pic[:,:,i],a)-ski.filters.gaussian(pic[:,:,i],b)
            #convolving with difference of gaussians (Note- f*(g+h)=f*g+f*h)
            pic[:,:,i+1] = (1-np.exp(-beta*pic[:,:,i+1]))/(1+np.exp(-beta*pic[:,:,i+1]))
            #normalized the image to values from -1 to 1
        p_white = np.sum(pic[:,:,-1]>0)/np.prod(np.shape(pic[:,:,-1]))
        condition = ((p_white>(sparsity-sparsityDel)) & (p_white<(sparsity+sparsityDel)))
        if condition:
            X[:,:,q] = (pic[:,:,-1]<=0)
            if show:
                ax = fig.add_subplot(nrows, ncols, q+1)
                ax.imshow(X[:,:,q],cmap='gray');
            q += 1
    return X