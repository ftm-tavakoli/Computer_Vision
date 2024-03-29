import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from skimage.segmentation import active_contour
import cv2

img3_BGR = cv2.imread('img3.jpg')
img3 = cv2.cvtColor(img3_BGR, cv2.COLOR_BGR2RGB)


s = np.linspace(0, 2*np.pi, 800)
r = 310 + 320*np.sin(s)
c = 550 + 200*np.cos(s)
init = np.array([r, c]).T


snake = active_contour(gaussian(img3, 3, preserve_range=False),
                       init, alpha=0.2, beta=0,w_line=0 , gamma=0.001, boundary_condition='periodic' )


fig, ax = plt.subplots(figsize=(9, 5))
ax.imshow(img3, cmap=plt.cm.gray)
ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, img3.shape[1], img3.shape[0], 0])

plt.show()
