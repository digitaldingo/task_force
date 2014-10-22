import numpy as np
import pandas as pd

#from scipy.ndimage import imread
from skimage import io

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.pipeline import Pipeline
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator

from IPython.html import widgets


### Opsætning ###
plt.style.use('bmh')
cmap = plt.cm.Set1
cseq = [plt.cm.Set1(i) for i in np.linspace(0, 1, 9, endpoint=True)]
plt.rcParams['axes.color_cycle'] = cseq


### Polynomium ###

def true_poly(x):
    return .05*x**3. + 0.02*x**2. - 1.3*x + 2.7


def ukendte_data(antal_data=17, seed=30):
    np.random.seed(seed)

    scale = 10.

    x = np.linspace(-7.5, 7.5, antal_data)
    x += scale*np.random.rand(antal_data)/5 - 0.5*scale/5
    x.sort()

    y = true_poly(x)
    y += scale*np.random.rand(antal_data) - 0.5*scale

    return x,y



def plot_data(x, y, titel=""):

    #fig, ax = plt.subplots()

    fig = plt.figure(42)
    ax = fig.add_subplot(111)

    if titel:
        ax.set_title(titel)

    ax.plot(x, y, ls="", marker="o", alpha=.7, mec=cseq[0])

    ax.set_ylim(-30,40)
    ax.set_xlim(-10,10)

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")

    #plt.show()

    #return plt.gcf(), plt.gca()


def polynomium(x, model):
    return model.predict(x[:,None])


def plot_polynomium(x, fit, data=None, ordener=None, figsize=(10,6)):

    # definitions for the axes
    left, width = 0.1, .8
    bottom_h = 0.1
    height_h = 0.2

    bottom = bottom_h + height_h + 0.12
    height = 1 - bottom

    rect_plot = [left, bottom, width, height]
    rect_hist = [left, bottom_h, width, height_h]

    plt.figure(figsize=figsize)

    ax1 = plt.axes(rect_plot)
    ax2 = plt.axes(rect_hist)

    if not data:
        data = ukendte_data()

    ax1.plot(data[0], data[1], ls="", marker="o", alpha=.7, mec=cseq[0])

    ax1.plot(x, polynomium(x,fit))
    ax1.set_ylim(-30,40)
    ax1.set_xlim(-10,10)

    ax1.set_xlabel(r"$x$")
    ax1.set_ylabel(r"$y$")

    lin_part = fit.named_steps['linear']

    coef = lin_part.coef_
    coef[0] = lin_part.intercept_

    ax2.bar(np.arange(len(coef)) - .25, np.abs(coef), width=.5, log=True, color=cseq[3])

    if ordener:
        ax2.set_xlim(-1, ordener[1]+1)
    else:
        ax2.set_xlim(-1, len(coef))

    ax2.yaxis.set_major_locator(LogLocator(numticks=5))
    ax2.set_xlabel(r"Koefficient")


def poly_data(n_points = 17):
    np.random.seed(23)

    scale = 5.

    x = np.linspace(-7,7,n_points)
    x += scale*np.random.rand(n_points) - 0.5*scale
    x.sort()

    y = true_poly(x)
    y += scale*np.random.rand(n_points) - 0.5*scale

    return x,y


def fit_polynomium(x, y, orden=2):

    model = Pipeline([('poly', PolynomialFeatures(degree=orden)),
                      ('linear', LinearRegression(fit_intercept=True))])

    model = model.fit(x[:, np.newaxis], y)

    return model



def plot_interaktivt_polynomium(x, ordener=(0,20), figsize=(10,6)):

    def f(orden):
        xd, yd = ukendte_data()

        fit = fit_polynomium(xd, yd, orden)
        plot_polynomium(x, fit, data=(xd,yd), ordener=ordener, figsize=figsize)

    widgets.interact(f, orden=widgets.IntSliderWidget(min=ordener[0],
                                                  max=ordener[1], step=1,
                                                  value=ordener[0]))


def plot_ml_polynomium(x, orden=10, figsize=(10,6)):

    xd, yd = ukendte_data()

    kf = cross_validation.KFold(len(xd), n_folds=5, shuffle=True, random_state=42)

    alphas = np.logspace(-4,40,500)

    model = Pipeline([('poly', PolynomialFeatures(degree=orden)),
                      ('linear', RidgeCV(alphas=alphas, cv=kf))])

    fit = model.fit(xd[:,None],yd)

    plot_polynomium(x, fit, data=(xd,yd), ordener=(0,orden), figsize=figsize)


def plot_ml_polynomium2(x, orden=10, figsize=(10,6), max_iter=1000):

    xd, yd = ukendte_data()

    kf = cross_validation.KFold(len(xd), n_folds=5, shuffle=True, random_state=42)

    alphas = np.logspace(-1,30,100)

    model = Pipeline([('poly', PolynomialFeatures(degree=orden)),
                      ('linear', LassoCV(alphas=alphas, cv=kf, max_iter=max_iter))])

    fit = model.fit(xd[:,None],yd)

    plot_polynomium(x, fit, data=(xd,yd), ordener=(0,orden), figsize=figsize)



### k Nearest Neighbour ###


def add_noise(x, y, scale = .5):
    np.random.seed(23)

    n_points = len(x)

    x2 = x + scale*np.random.rand(n_points) - 0.5*scale
    x2.sort()

    y2 = y + scale*np.random.rand(n_points) - 0.5*scale

    return x2,y2


def fit_kNN(x, y, k, form):
    kNN = KNeighborsRegressor(n_neighbors=k, weights=form)
    kNN.fit(x[:,None], y)

    return kNN


def kNN(x, fit):
    return fit.predict(x[:, None])


def plot_kNN(x, fit, data=None):

    if not data:
        xd, yd = ukendte_data()

    fig, ax = plt.subplots()

    ax.plot(data[0], data[1], ls="", marker="o", alpha=.7, mec=cseq[0])

    ax.plot(x, kNN(x,fit))

    ax.set_ylim(-30,40)
    ax.set_xlim(-10,10)


def plot_interaktiv_kNN(x, naboer=(1,17)):

    def f(k, metode):
        xd, yd = ukendte_data()

        fit = fit_kNN(xd, yd, k, metode)
        plot_kNN(x, fit, data=(xd,yd))

    widgets.interact(f, k=widgets.IntSliderWidget(min=naboer[0], max=naboer[1],
                                                  step=1, value=naboer[0]),
                     metode=("uniform", "distance"))



### K-means clustering ###

class Billede:

    def __init__(self, billede):
        self.billede = io.imread(billede)
        self.dims = self.billede.shape


    @property
    def billede_til_pixels(self):
        dx,dy,dz = self.dims
        self.pixels = self.billede.reshape(dx*dy,dz)
        data = pd.DataFrame(data=self.pixels, columns=["r","g","b"])

        return data

    @property
    def pixels_til_billede(self):
        dx,dy,dz = self.dims
        self.nyt_billede = self.pixels.reshape(dx,dy,dz)

        return self.nyt_billede


    def reducer_farver(self, kmeans):
        labels = kmeans.labels_
        centre = kmeans.cluster_centers_
        pixels = np.copy(self.pixels)
        #nyt_billede = copy(self.pixels)
        for label in np.unique(labels):
            idx = np.where(labels == label)
            #nyt_billede[idx] = centre[label]
            pixels[idx] = centre[label]

        dx,dy,dz = self.dims
        return pixels.reshape(dx,dy,dz)


def plot_farver2(billede, centre=None):

    pixels = billede.billede_til_pixels

    colours = ["r", "g", "b"]
    cnames = ["Rød", "Grøn", "Blå"]

    cmap = plt.cm.Blues

    fig, ax = plt.subplots(1,3, figsize=(12,6))

    extent = [0, 255, 0, 255]

    for i in range(2):
        for j in range(i+1,3):

            c1 = colours[i]
            c2 = colours[j]

            c = i + j - 1


            H, xedges, yedges = np.histogram2d(pixels[c2], pixels[c1], 256,
                                               range=[[extent[0], extent[1]],
                                                      [extent[2], extent[3]]])

            H[np.where(H < 1)] = 0.1

            cax = ax[c].imshow(H, extent=extent, aspect='equal', origin='lower',
                               cmap = cmap,
                               norm=LogNorm(vmin=1, vmax=np.nanmax(H)), zorder=10)

            ax[c].set_xlabel(cnames[i])
            ax[c].set_ylabel(cnames[j])

            if centre is not None:
                ax[c].scatter(centre[:,i], centre[:,j], marker='x', s=50,
                              linewidths=3, color=plt.cm.Reds(.9), zorder=20)


            ax[c].set_ylim(0,255)
            ax[c].set_xlim(0,255)

    fig.tight_layout()


def plot_farver(billede, centre=None):

    pixels = billede.billede_til_pixels

    colours = ["r", "g", "b"]
    cnames = ["Rød", "Grøn", "Blå"]

    cmap = plt.cm.Blues

    fig, ax = plt.subplots(1,3, figsize=(12,6))

    extent = [0, 255, 0, 255]

    for i in range(2):
        for j in range(i+1,3):

            c1 = colours[i]
            c2 = colours[j]

            c = i + j - 1

            H, xedges, yedges = np.histogram2d(pixels[c2], pixels[c1], 256,
                                               range=[[extent[0], extent[1]],
                                                      [extent[2], extent[3]]])


            H[np.where(H < 1)] = 0.1

            cax = ax[c].imshow(H, extent=extent, aspect='equal', origin='lower',
                               cmap = cmap,
                               norm=LogNorm(vmin=1, vmax=np.nanmax(H)), zorder=10)

            ax[c].set_xlabel(cnames[i])
            ax[c].set_ylabel(cnames[j])

            if centre is not None:
                ax[c].scatter(centre[:,i], centre[:,j], marker='x', s=50,
                              linewidths=3, color=plt.cm.Reds(.9), zorder=15)

            ax[c].set_ylim(0,256)
            ax[c].set_xlim(0,256)

    fig.tight_layout()


def kMeans(billede, klynger=5):
    kmeans = KMeans(n_clusters=klynger)
    kmeans.fit(billede.billede_til_pixels)

    return kmeans


def plot_kMeans(billede, klynger=5):

    kmeans = kMeans(billede, klynger)
    centre = kmeans.cluster_centers_

    #print(centre)

    plot_farver2(billede, centre)


def plot_reducerede_farver(billede, klynger=5):

    kmeans = kMeans(billede, klynger)

    red_billede = billede.reducer_farver(kmeans)

    plt.imshow(red_billede)



### Random forests ###

def titanic_data(sti="data/titanic.csv"):
    return pd.read_csv(sti)


def udvaelg_features(data, features):
    d = data.dropna(subset=features)
    d2 = d[features]

    d2["who"][d2["who"] == "child"] = 0
    d2["who"][d2["who"] == "woman"] = 1
    d2["who"][d2["who"] == "man"] = 2

    d2["sex"][d2["sex"] == "female"] = 0
    d2["sex"][d2["sex"] == "male"] = 1

    return d2

def plot_feature_selection(data, features, figsize=(10,6)):

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(data[features[:-1]].values, data["survived"])

    n_features = len(features[:-1])

    fig, ax = plt.subplots(figsize=figsize)

    index = np.arange(n_features)
    bar_width = 0.5

    ax.bar(index, clf.feature_importances_, width=bar_width, color=cseq[3])

    plt.xticks(index + bar_width/2, features)

    ax.set_xlim(-0.5, n_features + 0.5 - bar_width)

    fig.tight_layout()
