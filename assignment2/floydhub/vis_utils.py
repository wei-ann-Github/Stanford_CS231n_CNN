import matplotlib.pyplot as plt
import numpy as np
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