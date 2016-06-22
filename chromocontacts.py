import argparse
import math
import gzip
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from matplotlib.colors import LogNorm


def read_contacts(file_name, chromosome_number, seg_len, contact_matrix):
    """
    Reads contacts from file_name. Contacts have to be in format:
    chromosome_1 position_1 chromosome_2 position_2
    where chromosome_1, chromosome_2 are in format "chr" + str(number)
        position_1, position_2 are ints
    It changes contact_matrix so that contact_matrix[i,j] contains
    the number of rows that described contact between fragment [seg_len*i:seg_len*(i+1)]
    and [seg_len*j:seg_len*(j+1)]
    :param file_name: string - filename of readsfile with .txt.gz extension
    :param chromosome_number: int - number of chromosome that is evaluated
    :param seg_len: int - length of segment
    :param contact_matrix: numpy style array of shape:
            (math.ceil(chromosome_length/seg_len), math.ceil(chromosome_length/seg_len))
    :return: None, modifies contact_matrix
    """
    with gzip.open(file_name, "rb") as handle:
        for line in handle:
            data = line.strip().split()
            data[0] = data[0][3:]
            data[2] = data[2][3:]
            data = list(map(int, data))
            if (data[0] == chromosome_number) and (data[2] == chromosome_number):
                i = data[1] // seg_len
                j = data[3] // seg_len
                contact_matrix[i][j] += 1
                if i != j:
                    contact_matrix[j][i] += 1


def rebin(mat, shape):
    """
    Squeezes the array's resolution so that new values are the mean of former array's shape[0] x shape[1] subarrays
    :param mat: numpy style array to be modified
    :param shape: desired shape
    :return: Reshaped mat
    """
    sh = shape[0], mat.shape[0] // shape[0], shape[1], mat.shape[1] // shape[1]
    return mat.reshape(sh).mean(-1).mean(1)

def display_contacts(contact_matrix, x_1, x_2, y_1, y_2, radius, colormap, logarithmic, interpolate):
    """
    Displays contact_matrix[x_1:x_2, y_1:y_2]
    :param contact_matrix: numpy style array
    :param x_1: int
    :param x_2: int
    :param y_1: int
    :param y_2: int
    :param radius: float - Gaussian filter sigma
    :param colormap: string - cm.colormap
    :param logarithmic: bool - True if colormap should be logarithmic
    :param interpolate: bool - True if the image should be displayed with interpolation
    :return:
    """
    blurred_matrix = scipy.ndimage.gaussian_filter(contact_matrix, sigma=(radius, radius))
    # if no blur then radius=0
    if logarithmic:
        if interpolate:
            plt.imshow(blurred_matrix[x_1:x_2, y_1:y_2], cmap=colormap, norm=LogNorm())
        else:
            plt.imshow(blurred_matrix[x_1:x_2, y_1:y_2], interpolation="None", cmap=colormap, norm=LogNorm())
    else:
        if interpolate:
            plt.imshow(blurred_matrix[x_1:x_2, y_1:y_2], cmap=colormap)
        else:
            plt.imshow(blurred_matrix[x_1:x_2, y_1:y_2], interpolation="None", cmap=colormap)
    plt.show()


def construct_M(B, T):
    """
    Constructs M matrix from B and T as follows:
    M[i,j] = B[i] * B[j] * T[i,j]
    :param B: numpy style one dimensional array - multiplicative weight vector
    :param T: numpy style array - column-normalized symmetric matrix
    :return: numpy style array M
    """
    size_mat = T.shape[0]
    M = np.zeros((size_mat, size_mat))
    for i in range(size_mat):
        for j in range(size_mat):
            M[i,j] = B[i] * B[j] * T[i,j]
    return M

def normalize(contact_matrix, required_eps, iterations):
    """
    Finds B, T, EPS so that contact_matrix[i,j] = B[i] * B[j] * T[i,j]
    and sum(T[:, i]) = 1 +- EPS for every i (columns are normalized to 1 with EPS precision)
    :param contact_matrix: numpy style array - matrix to be normalized
    :param required_eps: float - desired precision of normalization
    :param iterations: int - up to this amount interations are performed
    :return: (B, T, EPS)
    """
    size_mat = contact_matrix.shape[0]
    B = np.ones(size_mat)
    optimal_epsilons = np.zeros(size_mat)
    T = np.zeros((size_mat, size_mat))
    optimal_sums = np.ones(size_mat)
    k = np.ones(size_mat)
    for i in range(size_mat):
        suma = sum(contact_matrix[:, i])
        if suma == 0:
            optimal_epsilons[i] = -1  # desired epsilons (either 0 or -1 [in case sum(column) = 0])
            optimal_sums[i] = 0  # desired sums (0 or 1)

        B[i] = contact_matrix[i, i]
        k[i] = contact_matrix[i, i]

        if contact_matrix[i, i] == 0:
            B[i] = 1
            k[i] = 1

    for iteration in range(iterations):
        for i in range(size_mat):
            for j in range(size_mat):
                T[i, j] = contact_matrix[i, j] / (B[i]*B[j])

        EPS = 0
        for i in range(size_mat):
            suma = sum(T[:, i])
            epsilon = suma - optimal_sums[i]

            if abs(epsilon) > EPS:
                EPS = abs(epsilon)
            if k[i] > 0.001:
                k[i] = k[i] / 2
            if epsilon < optimal_epsilons[i]:
                B[i] -= k[i]
            elif epsilon > optimal_epsilons[i]:
                B[i] += k[i]

        if EPS <= required_eps:
            return (B, T, EPS)
    return (B, T, EPS)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", type=str, help="filename of readsfile with .txt.gz extension", default="test.txt.gz")
    parser.add_argument("-c", type=int, help="chromosome number", required=True)
    parser.add_argument("-s", type=int, help="segment length", required=True)
    parser.add_argument("-l", type=int, help="chromosome length", required=True)
    parser.add_argument("-x_1", type=int, help="display submatrix[x_1:x_2, y_1:y_2", default=0)
    parser.add_argument("-x_2", type=int, help="display submatrix[x_1:x_2, y_1:y_2", default=-1)
    parser.add_argument("-y_1", type=int, help="display submatrix[x_1:x_2, y_1:y_2", default=0)
    parser.add_argument("-y_2", type=int, help="display submatrix[x_1:x_2, y_1:y_2", default=-1)
    parser.add_argument("-r", type=float, help="blur image with Gaussian filter (sigma=(r, r, 0))", default=0)
    parser.add_argument("--cm", type=str, help="colormap from matplotlib", default="gist_earth")
    parser.add_argument("--logcm", type=bool, help="True if logarithmic colormap should be applied", default="False")
    parser.add_argument("--shrink", type=int, help="shrink resolution by factor shrink", default=1)
    parser.add_argument("--interp", type=bool, help="True if images should be shown with interpolation", default=False)
    parser.add_argument("--save", type=str, help="Filename if matrix should be saved instead of being displayed", default="No")
    parser.add_argument("--eps", type=float, help="Precision of normalization - required epsilon", default=0.0)
    parser.add_argument("--it", type=int, help="Number of iterations for normalization", default=100000)
    parser.add_argument("--normal", type=bool, help="True if normalized matrix should be displayed", default=False)

    args = parser.parse_args()
    f = args.f
    c = args.c
    s = args.s
    l = args.l
    x_1 = args.x_1
    x_2 = args.x_2
    y_1 = args.y_1
    y_2 = args.y_2
    r = args.r
    cm = args.cm
    logcm = args.logcm
    shrink = args.shrink
    interp = args.interp
    save = args.save
    eps = args.eps
    it = args.it
    normal = args.normal

    n = math.ceil(l/s)

    # if x_2, y_2 were not set, display full image
    if x_2 == -1:
        x_2 = n
    if y_2 == -1:
        y_2 = n

    M = np.zeros((n, n))

    read_contacts(f, c, s, M)

    K = normalize(M, eps, it)

#    print("B:")
#    print(K[0])
#    print("T:")
#    print(K[1])
#    print("epsilon")
#    print(K[2])

    if normal:
        M = K[1]

    if save == "No":
        if shrink != 1:  # when shrinking is needed
            shape = n // shrink
            display_contacts(rebin(M, (shape, shape)), x_1, x_2, y_1, y_2, r, cm, logcm, interp)
        else:
            display_contacts(M, x_1, x_2, y_1, y_2, r, cm, logcm, interp)
    else:
        np.savetxt(save, M)
