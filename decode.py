from matplotlib.colors import LinearSegmentedColormap

import matplotlib as mpl
mpl.use('TkAgg')

import numpy as np
import sys
import time

bluegreen = LinearSegmentedColormap('bluegreen', {
    'red': ((0., 0., 0.),
            (1., 0., 0.)),
    'green': ((0., 0., 0.),
              (1., 1., 1.)),
    'blue': ((0., 0.2, 0.2),
             (0.5, 0.5, 0.5),
             (1., 0., 0.))
    })

### Load the Miyawaki dataset #####################################################
import datasets
dataset = datasets.get_miyawaki()

# Keep only random runs
X_random = dataset.func[12:]
y_random = dataset.label[12:]
y_shape = (10, 10)

### Preprocess data ###########################################################
import masking, preprocess
import nibabel

sys.stderr.write("Preprocessing data...")
t0 = time.time()

# Load and mask fMRI data

X_train = []
for x_random in X_random:
    # Mask data
    x_img = nibabel.load(x_random)
    x = masking.apply_mask(x_img, dataset.mask)
    x = preprocess.clean(x)
    X_train.append(x)

# Load target data
y_train = []
for y in y_random:
    y_train.append(np.reshape(np.loadtxt(y, dtype=np.int, delimiter=','),
                              (-1,) + y_shape, order='F'))

X_train = [x[3:] for x in X_train]
y_train = [y[:-3] for y in y_train]

X_train = np.vstack(X_train)
y_train = np.vstack(y_train).astype(np.float)

# Remove rest period
X_train = X_train[y_train[:, 0, 0] != -1]
y_train = y_train[y_train[:, 0, 0] != -1]

y_train = np.reshape(y_train, (-1, y_shape[0] * y_shape[1]))

sys.stderr.write(" Done (%.2fs)\n" % (time.time() - t0))

### Prediction function #######################################################
import pylab as pl
import os

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression as LogR
from sklearn.linear_model import LinearRegression as LinR
from sklearn.feature_selection import f_classif, SelectKBest

sys.stderr.write("Single pixel prediction\n")

# Pixel chosen for the study
p = (4, 2)

# Get index of the chosen pixel in flattened array
i_p = 42

# Logistic Regression
sys.stderr.write("\tLogistic regression...")
t0 = time.time()
cache_path = os.path.join('output', 'logr_coef.npy')
logr = LogR(penalty='l1', C=0.05)
logr.fit(X_train, y_train[:, i_p])
np.save(cache_path, logr.coef_)
logr_coef = np.load(cache_path)
sys.stderr.write(" Done (%.2fs)\n" % (time.time() - t0))

# Linear Regression
sys.stderr.write("\tLinear regression...")
t0 = time.time()
cache_path = os.path.join('output', 'logr_coef.npy')
linr = LinR(normalize=True)
linr.fit(X_train, y_train[:, i_p])
np.save(cache_path, logr.coef_)
linr_coef = np.load(cache_path)
sys.stderr.write(" Done (%.2fs)\n" % (time.time() - t0))

# Support Vector Classifier
sys.stderr.write("\tSupport vector classifier...")
t0 = time.time()
cache_path = os.path.join('output', 'svc_coef.npy')
svc = LinearSVC(penalty='l1', dual=False, C=0.01)
svc.fit(X_train, y_train[:, i_p])
np.save(cache_path, svc.coef_)
svc_coef = np.load(cache_path)
sys.stderr.write(" Done (%.2fs)\n" % (time.time() - t0))

### Output ###################################################################
from matplotlib.lines import Line2D

# Create masks for contour
### Mask of chosen voxels
contour = np.zeros(nibabel.load(dataset.mask).shape, dtype=bool)
for x, y in [(31, 9), (31, 10), (30, 10), (32, 10)]:
    contour[x, y, 10] = 1
### Mask of chosen pixel
pixmask = np.zeros((10, 10), dtype=bool)
pixmask[p] = 1


def plot_lines(mask, linewidth=3, color='b'):
    for i, j in np.ndindex(mask.shape):
        if i + 1 < mask.shape[0] and mask[i, j] != mask[i + 1, j]:
            pl.gca().add_line(Line2D([j - .5, j + .5], [i + .5, i + .5],
                color=color, linewidth=linewidth))
        if j + 1 < mask.shape[1] and mask[i, j] != mask[i, j + 1]:
            pl.gca().add_line(Line2D([j + .5, j + .5], [i - .5, i + .5],
                color=color, linewidth=linewidth))


fig = pl.figure(figsize=(8, 8))
ax1 = pl.axes([0., 0., 1., 1.])
sbrain = masking.unmask(logr_coef[0], dataset.mask)
bg = nibabel.load(os.path.join('bg.nii.gz'))
pl.imshow(bg.get_data()[:, :, 10].T, interpolation="nearest", cmap='gray',
          origin='lower')
pl.imshow(np.ma.masked_equal(sbrain[:, :, 10].T, 0.), interpolation="nearest",
          cmap=bluegreen, origin='lower', vmin=0., vmax=2.6)
plot_lines(contour[:, :, 10].T, color='r')
pl.axis('off')
ax2 = pl.axes([.1, .5, .05, .45])
cb = pl.colorbar(cax=ax2, ax=ax1)
cb.ax.yaxis.set_ticks_position('left')
cb.ax.yaxis.set_tick_params(labelcolor='white')
cb.ax.yaxis.set_tick_params(labelsize=32)
cb.set_ticks([0., 1.3, 2.6])
pl.savefig(os.path.join('output', 'decoding_pixel_logistic.pdf'))
pl.savefig(os.path.join('output', 'decoding_pixel_logistic.png'))
pl.savefig(os.path.join('output', 'decoding_pixel_logistic.eps'))
sys.stderr.write("Logistic regression: %d nonzero voxels\n" %
        np.sum(logr_coef != 0.))
pl.close()

fig = pl.figure(figsize=(8, 8))
ax1 = pl.axes([0., 0., 1., 1.])
sbrain = masking.unmask(linr_coef[0], dataset.mask)
bg = nibabel.load(os.path.join('bg.nii.gz'))
pl.imshow(bg.get_data()[:, :, 10].T, interpolation="nearest", cmap='gray',
          origin='lower')
pl.imshow(np.ma.masked_equal(sbrain[:, :, 10].T, 0.), interpolation="nearest",
          cmap=bluegreen, origin='lower', vmin=0., vmax=2.6)
plot_lines(contour[:, :, 10].T, color='r')
pl.axis('off')
ax2 = pl.axes([.1, .5, .05, .45])
cb = pl.colorbar(cax=ax2, ax=ax1)
cb.ax.yaxis.set_ticks_position('left')
cb.ax.yaxis.set_tick_params(labelcolor='white')
cb.ax.yaxis.set_tick_params(labelsize=32)
cb.set_ticks([0., 1.3, 2.6])
pl.savefig(os.path.join('output', 'decoding_pixel_linear.pdf'))
pl.savefig(os.path.join('output', 'decoding_pixel_linear.png'))
pl.savefig(os.path.join('output', 'decoding_pixel_linear.eps'))
sys.stderr.write("Linear regression: %d nonzero voxels\n" %
        np.sum(logr_coef != 0.))
pl.close()

fig = pl.figure(figsize=(8, 8))
ax1 = pl.axes([0., 0., 1., 1.])
sbrain = masking.unmask(svc_coef[0], dataset.mask)
vmax = np.max(np.abs(sbrain[:, :, 10].T))
pl.imshow(bg.get_data()[:, :, 10].T, interpolation="nearest", cmap='gray',
          origin='lower')
pl.imshow(np.ma.masked_equal(sbrain[:, :, 10].T, 0.), interpolation="nearest",
          cmap=bluegreen, origin='lower', vmin=0., vmax=1.0)
plot_lines(contour[:, :, 10].T, color='r')
pl.axis('off')
ax2 = pl.axes([.1, .5, .05, .45])
cb = pl.colorbar(cax=ax2, ax=ax1)
cb.ax.yaxis.set_ticks_position('left')
cb.ax.yaxis.set_tick_params(labelcolor='white')
cb.ax.yaxis.set_tick_params(labelsize=28)
cb.set_ticks([0., .5, 1.])
pl.savefig(os.path.join('output', 'pixel_svc.pdf'))
pl.savefig(os.path.join('output', 'pixel_svc.png'))
pl.savefig(os.path.join('output', 'pixel_svc.eps'))
sys.stderr.write("SVC: %d nonzero voxels\n" % np.sum(logr_coef != 0.))
pl.close()


### Calcualte the Cross Validation Scores ###################################################
from sklearn.cross_validation import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.externals.joblib import Parallel, delayed

pipeline_LogR = Pipeline([('selection', SelectKBest(f_classif, 500)),
                     ('clf', LogR(penalty="l1", C=0.05))])
pipeline_LinR = Pipeline([('selection', SelectKBest(f_classif, 500)),
                     ('clf', LinR(normalize=True))])
pipeline_SVC = Pipeline([('selection', SelectKBest(f_classif, 500)),
                     ('clf', LinearSVC(penalty='l1', dual=False, C=0.01))])
pipeline_SVCL2 = Pipeline([('selection', SelectKBest(f_classif, 500)),
                     ('clf', LinearSVC(penalty='l2', dual=False, C=0.001))])

sys.stderr.write("Cross validation\n")

sys.stderr.write("\tLogistic regression...")
t0 = time.time()
cache_path = os.path.join('output', 'logR_scores.npy')
if not os.path.exists(cache_path):
    scores_log = Parallel(n_jobs=1)(delayed(cross_val_score)(
        pipeline_LogR, X_train, y, cv=5, verbose=True) for y in y_train.T)
    np.save(cache_path, scores_log)
logr_scores = np.load(cache_path)
sys.stderr.write(" Done (%.2fs)\n" % (time.time() - t0))

sys.stderr.write("\tLinear regression...")
t0 = time.time()
cache_path = os.path.join('output', 'linR_scores.npy')
if not os.path.exists(cache_path):
    scores_log = Parallel(n_jobs=1)(delayed(cross_val_score)(
        pipeline_LinR, X_train, y, cv=5, verbose=True) for y in y_train.T)
    np.save(cache_path, scores_log)
linr_scores = np.load(cache_path)
sys.stderr.write(" Done (%.2fs)\n" % (time.time() - t0))

sys.stderr.write("\tSupport vector classifier...")
t0 = time.time()
cache_path = os.path.join('output', 'svc_scores.npy')
if not os.path.exists(cache_path):
    scores_svc = Parallel(n_jobs=1)(delayed(cross_val_score)(
        pipeline_SVC, X_train, y, cv=5, verbose=True) for y in y_train.T)
    np.save(cache_path, scores_svc)
svc_scores = np.load(cache_path)
sys.stderr.write(" Done (%.2fs)\n" % (time.time() - t0))

sys.stderr.write("\tSupport vector classifier L2...")
t0 = time.time()
cache_path = os.path.join('output', 'svcl2_scores.npy')
if not os.path.exists(cache_path):
    scores_svcl2 = Parallel(n_jobs=1)(delayed(cross_val_score)(
        pipeline_SVCL2, X_train, y, cv=5, verbose=True) for y in y_train.T)
    np.save(cache_path, scores_svcl2)
svcl2_scores = np.load(cache_path)
sys.stderr.write(" Done (%.2fs)\n" % (time.time() - t0))

### Output ####################################################################

fig = pl.figure(figsize=(8, 8))
pl.imshow(np.array(logr_scores).mean(1).reshape(10, 10),
    interpolation="nearest", vmin=.3, vmax=1.)
plot_lines(pixmask, linewidth=6)
pl.axis('off')
pl.hot()
fig.subplots_adjust(bottom=0., top=1., left=0., right=1.)
pl.savefig(os.path.join('output', 'decoding_scores_log.pdf'))
pl.savefig(os.path.join('output', 'decoding_scores_log.eps'))
print('Logistic Regression mean accuracy: %f' % logr_scores.mean())
pl.close()

fig = pl.figure(figsize=(8, 8))
pl.imshow(np.array(linr_scores).mean(1).reshape(10, 10),
    interpolation="nearest", vmin=.3, vmax=1.)
plot_lines(pixmask, linewidth=6)
pl.axis('off')
pl.hot()
fig.subplots_adjust(bottom=0., top=1., left=0., right=1.)
pl.savefig(os.path.join('output', 'decoding_scores_lin.pdf'))
pl.savefig(os.path.join('output', 'decoding_scores_lin.eps'))
print('Linear Regression mean accuracy: %f' % linr_scores.mean())
pl.close()

fig = pl.figure(figsize=(8, 8))
pl.imshow(np.array(svc_scores).mean(1).reshape(10, 10),
        interpolation="nearest", vmin=.3, vmax=1.)
plot_lines(pixmask, linewidth=6)
pl.axis('off')
pl.hot()
fig.subplots_adjust(bottom=0., top=1., left=0., right=1.)
pl.savefig(os.path.join('output', 'decoding_scores_svc.pdf'))
pl.savefig(os.path.join('output', 'decoding_scores_svc.eps'))
print('SVC L1 mean accuracy: %f' % svc_scores.mean())
pl.close()


fig = pl.figure(figsize=(8, 8))
pl.imshow(np.array(svcl2_scores).mean(1).reshape(10, 10),
        interpolation="nearest", vmin=.3, vmax=1.)
plot_lines(pixmask, linewidth=6)
pl.axis('off')
pl.hot()
fig.subplots_adjust(bottom=0., top=1., left=0., right=1.)
pl.savefig(os.path.join('output', 'decoding_scores_svcl2.pdf'))
pl.savefig(os.path.join('output', 'decoding_scores_svcl2.eps'))
print('SVC L2 mean accuracy: %f' % svcl2_scores.mean())
pl.close()

### Colorbar #########################################################
import matplotlib as mpl

fig = pl.figure(figsize=(.6, 3.6))
cmap = mpl.cm.hot
norm = mpl.colors.Normalize(vmin=.3, vmax=1.)
cb = mpl.colorbar.ColorbarBase(pl.gca(), cmap=cmap, norm=norm)
# cb.ax.yaxis.set_ticks_position('left')
cb.set_ticks(np.arange(0.3, 1.1, 0.1))
fig.subplots_adjust(bottom=0.03, top=.97, left=0., right=.5)
pl.savefig(os.path.join('output', 'decoding_scores_colorbar.png'))
pl.savefig(os.path.join('output', 'decoding_scores_colorbar.pdf'))
pl.savefig(os.path.join('output', 'decoding_scores_colorbar.eps'))
