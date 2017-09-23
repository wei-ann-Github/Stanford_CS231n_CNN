# from past.builtins import xrange

from math import sqrt, ceil
import numpy as np

def visualize_grid(Xs, ubound=255.0, padding=1):
  """
  Reshape a 4D tensor of image data to a grid for easy visualization.

  Inputs:
  - Xs: Data of shape (N, H, W, C)
  - ubound: Output grid will have values scaled to the range [0, ubound]
  - padding: The number of blank pixels between elements of the grid
  """
  (N, H, W, C) = Xs.shape
  grid_size = int(ceil(sqrt(N)))
  grid_height = H * grid_size + padding * (grid_size - 1)
  grid_width = W * grid_size + padding * (grid_size - 1)
  grid = np.zeros((grid_height, grid_width, C))
  next_idx = 0
  y0, y1 = 0, H
  for y in range(grid_size):
    x0, x1 = 0, W
    for x in range(grid_size):
      if next_idx < N:
        img = Xs[next_idx]
        low, high = np.min(img), np.max(img)
        grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
        # grid[y0:y1, x0:x1] = Xs[next_idx]
        next_idx += 1
      x0 += W + padding
      x1 += W + padding
    y0 += H + padding
    y1 += H + padding
  # grid_max = np.max(grid)
  # grid_min = np.min(grid)
  # grid = ubound * (grid - grid_min) / (grid_max - grid_min)
  return grid

def vis_grid(Xs):
  """ visualize a grid of images """
  (N, H, W, C) = Xs.shape
  A = int(ceil(sqrt(N)))
  G = np.ones((A*H+A, A*W+A, C), Xs.dtype)
  G *= np.min(Xs)
  n = 0
  for y in range(A):
    for x in range(A):
      if n < N:
        G[y*H+y:(y+1)*H+y, x*W+x:(x+1)*W+x, :] = Xs[n,:,:,:]
        n += 1
  # normalize to [0,1]
  maxg = G.max()
  ming = G.min()
  G = (G - ming)/(maxg-ming)
  return G
  
def vis_nn(rows):
  """ visualize array of arrays of images """
  N = len(rows)
  D = len(rows[0])
  H,W,C = rows[0][0].shape
  Xs = rows[0][0]
  G = np.ones((N*H+N, D*W+D, C), Xs.dtype)
  for y in range(N):
    for x in range(D):
      G[y*H+y:(y+1)*H+y, x*W+x:(x+1)*W+x, :] = rows[y][x]
  # normalize to [0,1]
  maxg = G.max()
  ming = G.min()
  G = (G - ming)/(maxg-ming)
  return G

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import warnings
warnings.simplefilter('ignore', np.RankWarning)

def vis_accuracy(X, Y, title1='', title2='', xlab='', ylab=''):
    '''
    Arg
    X = a list of tuple where a tuple = (X_values, accuracy)
    Y = a list of tuple where a tuple = (y_values, accuracy)
    '''
    # data for the scatter plot
    x = list(map(lambda x: x[0], X))
    y = list(map(lambda x: x[0], Y))
    
    # data for the best fit plot
    x2, acc_x = list(zip(*sorted(X, key=lambda x: x[0])))
    y2, acc_y = list(zip(*sorted(Y, key=lambda x: x[0])))
     
    nullfmt = NullFormatter()         # no labels

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_linex = [left, bottom_h, width, 0.2]
    rect_liney = [left_h, bottom, 0.2, height]

    # start with a rectangular Figure
    plt.figure(1, figsize=(8, 8))

    axScatter = plt.axes(rect_scatter)
    axLinex = plt.axes(rect_linex)#, sharey=axScatter)
    axLiney = plt.axes(rect_liney)#, sharex=axScatter)

    # no labels
    axLinex.xaxis.set_major_formatter(nullfmt)
    axLiney.yaxis.set_major_formatter(nullfmt)

    # the scatter plot:
    axScatter.scatter(x, y)
    axScatter.grid(b=True, which='major', color='gray', linestyle='--')
    axScatter.set_xlabel(xlab)
    axScatter.set_ylabel(ylab)
    
    # for plotting the best fit accuracy curve
    num_pts = 100; order = 3
    coeffs_x = np.polyfit(x2, acc_x, order)
    x3 = np.arange(num_pts+1)*(np.max(x2)-np.min(x2))/num_pts + np.min(x2)
    fit_x = np.polyval(coeffs_x, x3)
    
    coeffs_y = np.polyfit(y2, acc_y, order)
    y3 = np.arange(num_pts+1)*(np.max(y2)-np.min(y2))/num_pts + np.min(y2)
    fit_y = np.polyval(coeffs_y, y3)
    
    # plot the curve and place dots on the curve    
    axLinex.plot(x3, fit_x)
    axLinex.scatter(x2, acc_x)
    axLinex.grid(b=True, which='major', color='gray', linestyle='--')
    axLinex.set_title(title1)
    axLiney.plot(fit_y, y3)
    axLiney.scatter(acc_y, y2)
    axLiney.grid(b=True, which='major', color='gray', linestyle='--')
    axLiney.set_title(title2)

    plt.show()