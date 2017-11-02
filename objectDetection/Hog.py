##Histogram of Oriented Gradients，很重要的一章!!!!!!!!!!!
# http://stackoverflow.com/questions/6090399/get-hog-image-features-from-opencv-python
# http://www.juergenwiki.de/work/wiki/doku.php?id=public:hog_descriptor_computation_and_visualization
import numpy as np
import cv2
import matplotlib.pyplot as plt
##HOGs是一種feature discriptor，很常被拿來運用在物體偵測，會把圖像上的物體轉變成一個feature vector
##接著使用SVM等分類演算法，來判斷圖像上是否有出現某物體
#此處有個範例為偵測人類，https://lear.inrialpes.fr/people/triggs/pubs//Dalal-cvpr05.pdf


# Load image then grayscale
image = cv2.imread('../images/elephant.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Show original Image
cv2.imshow('Input Image', image)
cv2.waitKey(0)
##此處使用一個8*8pixel的block(裡面會有2*2的cells總共4個)，從中計算block裡面每個pixel的gradient vector or edge orientation???，
##經過計算後一個block裡有64個gradient vector，把她轉換成histogram
##可以把每個cell轉換成angular bins???，這裡代表每個gradient vector代表一個bin，但在上面那篇混文，把64個gradient vector
##轉換成9個bins(0-180)每一個都是20，所以就可以快速地把64vectors轉換成9個values????
##接著還要作標準化(以整個block來看)，看不太懂....
# h x w in pixels
cell_size = (8, 8)

 # h x w in cells
block_size = (2, 2)

# number of orientation bins
nbins = 9

# Using OpenCV's HOG Descriptor
# winSize is the size of the image cropped to a multiple of the cell size
hog = cv2.HOGDescriptor(_winSize=(gray.shape[1] // cell_size[1] * cell_size[1],
                                  gray.shape[0] // cell_size[0] * cell_size[0]),
                        _blockSize=(block_size[1] * cell_size[1],
                                    block_size[0] * cell_size[0]),
                        _blockStride=(cell_size[1], cell_size[0]),
                        _cellSize=(cell_size[1], cell_size[0]),
                        _nbins=nbins)

# Create numpy array shape which we use to create hog_feats
n_cells = (gray.shape[0] // cell_size[0], gray.shape[1] // cell_size[1])

# We index blocks by rows first.
# hog_feats now contains the gradient amplitudes for each direction,
# for each cell of its group for each group. Indexing is by rows then columns.
hog_feats = hog.compute(gray).reshape(n_cells[1] - block_size[1] + 1,
                        n_cells[0] - block_size[0] + 1,
                        block_size[0], block_size[1], nbins).transpose((1, 0, 2, 3, 4))

# Create our gradients array with nbin dimensions to store gradient orientations
gradients = np.zeros((n_cells[0], n_cells[1], nbins))

# Create array of dimensions
cell_count = np.full((n_cells[0], n_cells[1], 1), 0, dtype=int)

# Block Normalization
for off_y in range(block_size[0]):
    for off_x in range(block_size[1]):
        gradients[off_y:n_cells[0] - block_size[0] + off_y + 1,
                  off_x:n_cells[1] - block_size[1] + off_x + 1] += \
            hog_feats[:, :, off_y, off_x, :]
        cell_count[off_y:n_cells[0] - block_size[0] + off_y + 1,
                   off_x:n_cells[1] - block_size[1] + off_x + 1] += 1

# Average gradients
gradients /= cell_count

# Plot HOGs using Matplotlib
# angle is 360 / nbins * direction
color_bins = 5
plt.pcolor(gradients[:, :, color_bins])
plt.gca().invert_yaxis()
plt.gca().set_aspect('equal', adjustable='box')
plt.colorbar()
plt.show()
cv2.destroyAllWindows()