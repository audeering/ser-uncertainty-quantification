from __future__ import division, print_function

import seaborn as sns

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri


def plot_legend():
    fig, ax = plt.subplots(1, 1)
    pal1 = sns.color_palette(palette='Blues_r')
    pal2 = sns.color_palette(palette='Reds_r')
    ax.plot([1], label="musan-speech", color=pal1[0])
    ax.plot([1], label="crema-d", color=pal1[1])
    ax.plot([1], label="emodb", color=pal1[2])
    ax.plot([1], label="msp-podcast", color=pal1[3])
    ax.plot([1], label="cochlscene", color=pal2[0])
    ax.plot([1], label="musan-music", color=pal2[1])
    ax.plot([1], label="musan-noise", color=pal2[2])
    ax.plot([1], label="whitenoise", color='gray')
    h, l = ax.get_legend_handles_labels()

    ax.clear()
    ax.legend(h, l, loc='upper left', ncol=8)
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3,
    #           fancybox=True, shadow=True)
    ax.axis(False)
    # get handles and labels for reuse
    label_params = ax.get_legend_handles_labels()

    # plt.tight_layout()
    plt.show()


'''Functions for drawing contours of Dirichlet distributions.'''
# https://gist.github.com/tboggs/8778945
# Author: Thomas Boggs

_corners = np.array([[0, 0], [1, 0], [0.5, 0.75 ** 0.5]])
_AREA = 0.5 * 1 * 0.75 ** 0.5
_triangle = tri.Triangulation(_corners[:, 0], _corners[:, 1])

# For each corner of the triangle, the pair of other corners
_pairs = [_corners[np.roll(range(3), -i)[1:]] for i in range(3)]
# The area of the triangle formed by point xy and another pair or points
tri_area = lambda xy, pair: 0.5 * np.linalg.norm(np.cross(*(pair - xy)))


def xy2bc(xy, tol=1.e-4):
    '''Converts 2D Cartesian coordinates to barycentric.
    Arguments:
        `xy`: A length-2 sequence containing the x and y value.
    '''
    coords = np.array([tri_area(xy, p) for p in _pairs]) / _AREA
    return np.clip(coords, tol, 1.0 - tol)


class Dirichlet(object):
    def __init__(self, alpha):
        '''Creates Dirichlet distribution with parameter `alpha`.'''
        from math import gamma
        from operator import mul
        self._alpha = np.array(alpha)
        self._coef = gamma(np.sum(self._alpha)) / \
                     np.multiply.reduce([gamma(a) for a in self._alpha])

    def pdf(self, x):
        '''Returns pdf value for `x`.'''
        from operator import mul
        return self._coef * np.multiply.reduce([xx ** (aa - 1)
                                                for (xx, aa) in zip(x, self._alpha)])

    def sample(self, N):
        '''Generates a random sample of size `N`.'''
        return np.random.dirichlet(self._alpha, N)


def draw_pdf_contours(dist, border=False, nlevels=200, subdiv=8, **kwargs):
    '''Draws pdf contours over an equilateral triangle (2-simplex).
    Arguments:
        `dist`: A distribution instance with a `pdf` method.
        `border` (bool): If True, the simplex border is drawn.
        `nlevels` (int): Number of contours to draw.
        `subdiv` (int): Number of recursive mesh subdivisions to create.
        kwargs: Keyword args passed on to `plt.triplot`.
    '''
    from matplotlib import ticker, cm
    import math

    refiner = tri.UniformTriRefiner(_triangle)
    trimesh = refiner.refine_triangulation(subdiv=subdiv)
    pvals = [dist.pdf(xy2bc(xy)) for xy in zip(trimesh.x, trimesh.y)]

    plt.tricontourf(trimesh, pvals, nlevels, cmap='jet', **kwargs)
    plt.axis('equal')
    plt.xlim(0, 1)
    plt.ylim(0, 0.8)
    plt.axis('off')
    if border is True:
        plt.triplot(_triangle, linewidth=1)


if __name__ == '__main__':
    f = plt.figure(figsize=(6, 1.8))
    alphas = [[101, 5, 5],
              [50, 50, 50],
              [1.2, 1.2, 1.2]]
    for (i, alpha) in enumerate(alphas):
        plt.subplot(1, len(alphas), i + 1)
        dist = Dirichlet(alpha)
        draw_pdf_contours(dist)
        title = ['(a) Low uncertainty', '(b) High data uncertainty', '(c) Out-of-distribution'][i]
        plt.title(title, fontdict={'fontsize': 10},  y=-0.1)

    plt.tight_layout()
    plt.savefig(f'dirichlet_plot.eps', format='eps', transparent=True)
    print('Wrote plots to "dirichlet_plots.png".')

# if __name__ == "__main__":
#     plot_legend()
