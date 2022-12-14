import numpy
import numpy as np
import matplotlib.pyplot as plt
import cv2

# --------------------------------------------------Q1-------------------------------------------------- #
import scipy.signal


# Q1.a
# imread() returns uint8 ndarray with BGR color channel
building = cv2.imread(filename='../my_data/building.jpg')

rgb_building = cv2.cvtColor(building, cv2.COLOR_BGR2RGB)
gray_building = cv2.cvtColor(building, cv2.COLOR_BGR2GRAY)

fig11, axes = plt.subplots(1, 2, figsize=(10, 10))
axes[0].imshow(rgb_building)
axes[0].set_title("original image")
axes[1].imshow(gray_building, cmap='gray')
axes[1].set_title("original image converted to grayscale")
plt.tight_layout()
plt.show()

# Q1.b
# use fft2 for 2D fourier transform. use fftshift to bring the low frequencies to the center of the image
fft_gray_building = np.fft.fftshift(np.fft.fft2(gray_building))
fft_gray = np.log(1 + np.abs(fft_gray_building))
plt.imshow(fft_gray, cmap='gray')
plt.title('2D Fourier transform of the grayscale of the building picture')
plt.xlabel(r'$ \omega_1 $')
plt.ylabel(r'$ \omega_2 $')
plt.tight_layout()
plt.show()

# Q1.c

(row, col) = np.shape(fft_gray)

# there are 'column' number of frequencies. so 2% will be 'column * 0.02' columns.
# to get them we want the middle column 'col/2' and then we want to get the following range:
# subtract 'col/2 - round(0.01 * col)'
# and add 'col/2 + round(0.01 * col)'

left_column_idx = int(col / 2 - 0.01 * col)
right_column_idx = int(col / 2 + 0.01 * col)
left_row_idx = int(row / 2 - 0.01 * row)
right_row_idx = int(row / 2 + 0.01 * row)

freq_ldirection = np.zeros((row, col), dtype='complex')
freq_kdirection = np.zeros((row, col), dtype='complex')
freq_lkdirection = np.zeros((row, col))

# now we want to extract this range from the matrix
freq_ldirection[:, left_column_idx:right_column_idx] = fft_gray_building[:, left_column_idx:right_column_idx]
freq_kdirection[left_row_idx:right_row_idx, :] = fft_gray_building[left_row_idx:right_row_idx, :]

mask = np.zeros((row, col))
mask[:, left_column_idx:right_column_idx] = np.ones((row, right_column_idx - left_column_idx))
mask[left_row_idx:right_row_idx, :] = np.ones((right_row_idx - left_row_idx, col))
freq_lkdirection = np.multiply(mask, fft_gray_building)

fig13, axes = plt.subplots(1, 3, figsize=(15, 15))
axes[0].imshow(np.log(1 + np.abs(freq_ldirection)), cmap='gray')
axes[0].set_title('2% of the low frequencies in l direction')
axes[1].imshow(np.log(1 + np.abs(freq_kdirection)), cmap='gray')
axes[1].set_title('2% of the low frequencies in k direction')
axes[2].imshow(np.log(1 + np.abs(freq_lkdirection)), cmap='gray')
axes[2].set_title('2% of the low frequencies in l and k direction')
plt.tight_layout()
plt.show()

x_inv = np.fft.ifft2(np.fft.ifftshift(freq_ldirection))
y_inv = np.fft.ifft2(np.fft.ifftshift(freq_kdirection))
xy_inv = np.fft.ifft2(np.fft.ifftshift(freq_lkdirection))

fig132, axes = plt.subplots(1, 3, figsize=(15, 15))
axes[0].imshow(np.abs(x_inv), cmap='gray')
axes[0].set_title('Grayscale of the building with l filter')
axes[1].imshow(np.abs(y_inv), cmap='gray')
axes[1].set_title('Grayscale of the building with k filter')
axes[2].imshow(np.abs(xy_inv), cmap='gray')
axes[2].set_title('Grayscale of the building with l and k filters')
plt.tight_layout()
plt.show()


# Q1.d

def max_freq_filtering(fshift, precentege):
    """
 Reconstruct an image using only its maximal amplitude frequencies.
 :param fshift: The fft of an image, **after fftshift** -
 complex float ndarray of size [H x W].
 :param precentege: the wanted precentege of maximal frequencies.
 :return:
 fMaxFreq: The filtered frequency domain result -
 complex float ndarray of size [H x W].
 imgMaxFreq: The filtered image - real float ndarray of size [H x W].
 """

    precentege = precentege / 100

    unique_freq = np.unique(abs(fshift))

    bound_maximal_freq = unique_freq[int(unique_freq.shape[0] * (1 - precentege))]

    (row, col) = fshift.shape
    fMaxFreq = np.zeros((row, col), dtype=complex)
    for row_i in range(0, row):
        for col_i in range(0, col):
            if (abs(fshift[row_i][col_i]) >= bound_maximal_freq):
                fMaxFreq[row_i][col_i] = fshift[row_i][col_i]
    imgMaxFreq = np.fft.ifft2(np.fft.ifftshift(fMaxFreq))

    return fMaxFreq, imgMaxFreq


max_10_freq, building_img_mex10freq = max_freq_filtering(fft_gray_building, 10)
fig3, axes = plt.subplots(2, 1, figsize=(10, 10))
axes[0].imshow(np.log(1 + np.abs(max_10_freq)), cmap='gray')
axes[0].set_title('Filtering according to Max 10% fr: fr domain')
axes[1].imshow(np.abs(building_img_mex10freq), cmap='gray')
axes[1].set_title('Filtering according to Max 10% fr: space domain')
plt.tight_layout()
plt.show()

# Q1.e
max_4_freq, building_img_mex4freq = max_freq_filtering(fft_gray_building, 4)
fig4, axes = plt.subplots(2, 1, figsize=(10, 10))
axes[0].imshow(np.log(1 + np.abs(max_4_freq)), cmap='gray')
axes[0].set_title('Filtering according to Max 4% fr: fr domain')
axes[1].imshow(np.abs(building_img_mex4freq), cmap='gray')
axes[1].set_title('Filtering according to Max 4% fr: space domain')
plt.tight_layout()
plt.show()

# Q1.f
# as p gets bigger - we include more and more frequencies from the original image, thus the filtered image is closer
# to the original one and the MSE will be smaller
laxe = np.arange(1, 101)
kaxe = np.zeros(100)
for p in range(1, 101):
    _, img_max_freq = max_freq_filtering(fft_gray_building, p)
    kaxe[p - 1] = np.square(np.subtract(gray_building, np.abs(img_max_freq))).mean()
plt.title('MSE', fontsize=10)
plt.xlabel('p value', fontsize=10)
plt.ylabel('MSE(p)', fontsize=10)
plt.plot(laxe, kaxe)
plt.tight_layout()
plt.show()

# --------------------------------------------------Q2-------------------------------------------------- #

# Q2.a
parrot = cv2.imread(str('../given_data/parrot.png'))
portrait = cv2.resize(cv2.imread(str('../my_data/yours.jpg')), (parrot.shape[0], parrot.shape[1]))
gray_parrot = cv2.cvtColor(parrot, cv2.COLOR_BGR2GRAY)
gray_portrait = cv2.cvtColor(portrait, cv2.COLOR_BGR2GRAY)

fig21, axes = plt.subplots(1, 2, figsize=(10, 10))
axes[0].imshow(gray_parrot, cmap='gray')
axes[0].set_title("parrot gray image")
axes[1].imshow(gray_portrait, cmap='gray')
axes[1].set_title("portrait gray image")
plt.tight_layout()
plt.show()

# Q2.b
phase_parrot = np.angle(np.fft.fftshift(np.fft.fft2(gray_parrot)))
amp_parrot = np.abs(np.fft.fftshift(np.fft.fft2(gray_parrot)))
phase_portrait = np.angle(np.fft.fftshift(np.fft.fft2(gray_portrait)))
amp_portrait = np.abs(np.fft.fftshift(np.fft.fft2(gray_portrait)))

fig22, axes = plt.subplots(1, 2, figsize=(10, 10))
axes[0].imshow(np.log(1 + np.abs(amp_parrot)), cmap='gray')
axes[0].set_title("gray parrot amplitude")
axes[1].imshow(np.log(1 + np.abs(amp_portrait)), cmap='gray')
axes[1].set_title("gray portrait amplitude")
plt.tight_layout()
plt.show()

# Q2.c
ampPort_phaseParrot = np.multiply(amp_portrait, np.exp(1j * phase_parrot))

ampParrot_phasePort = np.multiply(amp_parrot, np.exp(1j * phase_portrait))

fig23, axes = plt.subplots(1, 2, figsize=(10, 10))
axes[0].imshow(np.abs(np.fft.ifft2(np.fft.ifftshift(ampPort_phaseParrot))), cmap='gray')
axes[0].set_title("Portrait Amplitude with Parrot Phase")
axes[1].imshow(np.abs(np.fft.ifft2(np.fft.ifftshift(ampParrot_phasePort))), cmap='gray')
axes[1].set_title("Portrait Phase with Parrot Amplitude")
plt.tight_layout()
plt.show()

# Q2.d
random_phase = np.random.uniform(low=-np.pi, high=np.pi, size=gray_portrait.shape)
random_amp = np.random.uniform(low=0, high=100, size=gray_portrait.shape)

ampPort_phaseRdm = np.multiply(amp_portrait, np.exp(1j * random_phase))

ampRdm_phasePort = np.multiply(random_amp, np.exp(1j * phase_portrait))

fig24, axes = plt.subplots(1, 2, figsize=(10, 10))
axes[0].imshow(np.abs(np.fft.ifft2(np.fft.ifftshift(ampPort_phaseRdm))), cmap='gray')
axes[0].set_title("Portrait Amplitude with Random Phase")
axes[1].imshow(np.abs(np.fft.ifft2(np.fft.ifftshift(ampRdm_phasePort))), cmap='gray')
axes[1].set_title("Portrait Phase with Random Amplitude")
plt.tight_layout()
plt.show()

# --------------------------------------------------Q3-------------------------------------------------- #

def func(x, y):
    pi = 3.141
    return np.sin(2*pi*(2/512)*(x+y))+\
           np.sin(2*pi*(5/512)*x)+\
           np.sin(2*pi*(40/512)*y)

# Q3.a
x = np.arange(0, 512, 1)
y = np.arange(0, 512, 1)
X, Y = np.meshgrid(x, y)  # grid of point
Z = func(X, Y)  # evaluation of the function on the grid
plt.imshow(Z, cmap='gray')
plt.title(r'2D Fourier transform of the grayscale of the $F_1(x,y)$')
plt.xlabel(r'$\omega_1$')
plt.ylabel(r'$\omega_2$')
plt.tight_layout()
plt.show()

# Q3.b
fft_Z = np.fft.fftshift(np.fft.fft2(Z))
fft_gray = np.log(1 + np.abs(fft_Z))
plt.imshow(fft_gray, cmap='gray')
plt.title(r'2D Fourier transform of the grayscale of the $F_1(x,y)$')
plt.xlabel(r'$\omega_1$')
plt.ylabel(r'$\omega_2$')
plt.tight_layout()
plt.show()

# Q3.c
x = np.arange(0, 512, 10)
y = np.arange(0, 512, 10)
X, Y = np.meshgrid(x, y)  # grid of point
Z = func(X, Y)  # evaluation of the function on the grid

fft_Z = np.fft.fftshift(np.fft.fft2(Z))
fft_gray = np.log(1 + np.abs(fft_Z))
fig33, axes = plt.subplots(1, 2, figsize=(10, 10))
axes[0].imshow(Z, cmap='gray')
axes[0].set_title(r"$F_{10}(x,y)$ in space domain")
axes[1].imshow(fft_gray, cmap='gray')
axes[1].set_title(r"$F_{10}(x,y)$ in frequency domain")
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[1].set_xlabel(r'$\omega_1$')
axes[1].set_ylabel(r'$\omega_2$')
plt.tight_layout()
plt.show()

# Q3.e
mandrill = cv2.imread(str('../given_data/mandrill.jpg'))
gray_mandrill = cv2.cvtColor(mandrill, cv2.COLOR_BGR2GRAY)

fft_mandrill = np.fft.fftshift(np.fft.fft2(gray_mandrill))
fft_mandrill = np.log(1 + np.abs(fft_mandrill))
fig35, axes = plt.subplots(1, 2, figsize=(10, 10))
axes[0].imshow(gray_mandrill, cmap='gray')
axes[0].set_title("grayscale Mandrill")
axes[1].imshow(fft_mandrill, cmap='gray')
axes[1].set_title("grayscale Mandrill in frequency domain")
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[1].set_xlabel(r'$\omega_1$')
axes[1].set_ylabel(r'$\omega_2$')
plt.tight_layout()
plt.show()

# Q3.f
down_sampled_mandrill = gray_mandrill[::4, ::4]

fft_down_sampled_mandrill = np.fft.fftshift(np.fft.fft2(gray_mandrill))
fft_down_sampled_mandrill = np.log(1 + np.abs(fft_down_sampled_mandrill))
fig36, axes = plt.subplots(1, 2, figsize=(10, 10))
axes[0].imshow(down_sampled_mandrill, cmap='gray')
axes[0].set_title("grayscale downsampled Mandrill")
axes[1].imshow(fft_down_sampled_mandrill, cmap='gray')
axes[1].set_title("grayscale downsampled Mandrill in frequency domain")
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[1].set_xlabel(r'$\omega_1$')
axes[1].set_ylabel(r'$\omega_2$')
plt.tight_layout()
plt.show()


# --------------------------------------------------Q4-------------------------------------------------- #

# Q4.a
def expand(mat):
    (row, col) = np.shape(mat)

    new_mat = np.zeros((row + 1, col + 1))

    new_mat[0:row, 0:col] = mat
    new_mat[row, :col] = mat[0, :]
    new_mat[:row, col] = mat[:, 0]
    new_mat[row, col] = mat[0, 0]

    return new_mat


def bilinear_displacement(dx, dy, image):
    """
    Calculate the displacement of a pixel using a bilinear interpolation.
    :param dx: the displacement in the x direction. dx in rang [0,1).
    :param dy: the displacement in the y direction. dy in rang [0,1).
    :param image: The image on which we preform the cyclic displacement
    :return:
    displaced_image: The new displaced image
    """

    (rows, cols) = np.shape(image)

    vec_x = np.array([1 - dx, dx])
    vec_y = np.array([1 - dy, dy]).T
    neighbors = np.zeros((2, 2))

    displaced_image = np.zeros((rows, cols))

    expanded_image = expand(image)
    for row in range(rows):
        for col in range(cols):
            neighbors = expanded_image[row:row + 2, col:col + 2]
            displaced_image[row, col] = np.matmul(np.matmul(vec_x, neighbors), vec_y)

    return displaced_image


def general_displacement(dx, dy, image):
    """
    Calculate the displacement of a pixel using a bilinear interpolation.
    :param dx: the displacement in the x direction.
    :param dy: the displacement in the y direction.
    :param image: The image on which we preform the cyclic displacement
    :return:
    displaced_image: The new displaced imag
    """

    (rows, cols) = np.shape(image)
    dx_fraction = float('%0.2f' % (dx % 1))
    dx_int = int(dx)
    dy_fraction = float('%0.2f' % (dy % 1))
    dy_int = int(dy)

    # shift by integer value
    displaced_x_image = np.zeros((np.shape(image)))  # x shift
    displaced_x_image[dy_int:, :] = image[:rows - dy_int, :]
    displaced_x_image[:rows - dy_int, :] = image[dy_int:, :]

    displaced_y_image = np.zeros((np.shape(image)))  # y shift
    displaced_y_image[:, :dx_int] = displaced_x_image[:, cols - dx_int:]
    displaced_y_image[:, cols - dx_int:] = displaced_x_image[:, :dx_int]

    # shift by non-natural value
    displaced_image = bilinear_displacement(dx_fraction, dy_fraction, displaced_y_image)
    return displaced_image


cameraman = cv2.imread(filename='../given_data/cameraman.jpg')

gray_cameraman = cv2.cvtColor(cameraman, cv2.COLOR_BGR2GRAY)

displaced_cameraman = general_displacement(150.7, 110.4, gray_cameraman)

fig43, axes = plt.subplots(1, 2, figsize=(10, 10))
axes[0].imshow(gray_cameraman, cmap='gray')
axes[0].set_title("original cameraman")
axes[1].imshow(displaced_cameraman, cmap='gray')
axes[1].set_title("displaced cameraman by [150.7, 110.4]")
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
plt.tight_layout()
plt.show()

# Q4.d
ryan = cv2.imread(filename='../given_data/Ryan.jpg')
gray_ryan = cv2.cvtColor(ryan, cv2.COLOR_BGR2GRAY)

(row, col) = np.shape(gray_ryan)
filter = np.zeros((row, col))

# empirically determined values
radius = 140
x0 = 50
y0 = 370

x, y = np.indices(gray_ryan.shape)
mask1 = ((radius ** 2 - (y - y0) ** 2) >= (x - x0) ** 2) * (x > x0)
filter[mask1] = 1

ryan_win = np.multiply(filter, gray_ryan)
fig44, axes = plt.subplots(1, 2, figsize=(10, 10))
axes[0].imshow(gray_ryan, cmap='gray')
axes[0].set_title("original grayscale ryan")
axes[1].imshow(ryan_win, cmap='gray')
axes[1].set_title("filtered ryan")
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
plt.tight_layout()
plt.show()


# Q4e
def rotating_img(image, theta):
    """
 Calculate the displacement of a pixel using a bilinear interpolation.
 :param image: the image to rotate.
 :param theta: Angle of rotation in radians.
 :return:
 rotated_image: The new displaced image
 """
    (rows, cols) = np.shape(image)

    radius = 140
    x0 = 50
    y0 = 370

    mid_x = int(y0)
    mid_y = int(x0 + radius/2)


    rotation_matrix = np.array([np.cos(theta), np.sin(theta), -np.sin(theta), np.cos(theta)]).reshape((2, 2))
    rotated_image = np.zeros((rows, cols))

    original_grid = [np.array([row, col]) for row in range(rows) for col in range(cols)]
    shifted_grid = [np.array([point[0] - mid_y, point[1] - mid_x]) for point in original_grid]
    rotated_grid = [np.matmul(rotation_matrix, np.array([point[0], point[1]]).T) for point in shifted_grid]
    rounded_rotated_grid = [np.array([int(round(point[0])), int(round(point[1]))]) for point in rotated_grid]
    transformed_grid = [np.array([point[0] + mid_y, point[1] + mid_x]) for point in rounded_rotated_grid]

    for pt, po in zip(transformed_grid, original_grid):
        if not(pt[0] > rows-1 or pt[0] < 0 or pt[1] > cols-1 or pt[1] < 0):
            rotated_image[pt[0]][pt[1]] = image[po[0]][po[1]]

    return rotated_image

rotated_ryan_30 = rotating_img(ryan_win, np.deg2rad(30))
rotated_ryan_45 = rotating_img(ryan_win, np.deg2rad(45))
rotated_ryan_90 = rotating_img(ryan_win, np.deg2rad(90))

fig45, axes = plt.subplots(1, 3, figsize=(10, 10))
axes[0].imshow(rotated_ryan_30, cmap='gray')
axes[0].set_title(r"rotated windowed Ryan by 30$\degree$")
axes[1].imshow(rotated_ryan_45, cmap='gray')
axes[1].set_title(r"rotated windowed Ryan by 45$\degree$")
axes[2].imshow(rotated_ryan_90, cmap='gray')
axes[2].set_title(r"rotated windowed Ryan by 90$\degree$")
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[2].set_xlabel('x')
axes[2].set_ylabel('y')
plt.tight_layout()
plt.show()



