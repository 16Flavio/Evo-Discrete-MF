import numpy as np
import scipy.io
import matplotlib.pyplot as plt

W = np.loadtxt('experiment/result_experiment_RELU/W_r40_RELU.txt')
H = np.loadtxt('experiment/result_experiment_RELU/H_r40_RELU.txt')
mnist_data = scipy.io.loadmat('data/MNIST_numpy.mat')
X = mnist_data['X']

n_img = 16
n_total = X.shape[0]

random_indices = np.random.choice(n_total, n_img, replace=False)
X_sample = X[random_indices]

W_sample = W[random_indices]
X_reconstruit = np.maximum(0, W_sample @ H)

X_ReLuNMD = scipy.io.loadmat('data/X_ReLuNMD_r5.mat')
X_ReLuNMD = X_ReLuNMD['X_reconstructed']
print(f"Relative error : {(np.linalg.norm(X-X_ReLuNMD)/(1e-12+np.linalg.norm(X)))*100:.2f}")


def show_grid_with_lines(data):
    # Paramètres
    n_row, n_col = 4, 4
    h, w = 28, 28
    
    canvas = np.zeros((n_row * h, n_col * w))
    
    for i in range(n_row):
        for j in range(n_col):
            idx = i * n_col + j
            if idx < len(data):
                img = data[idx].reshape((h, w))
                canvas[i*h:(i+1)*h, j*w:(j+1)*w] = img
    
    plt.figure(figsize=(5, 5))
    
    plt.imshow(canvas, cmap='gray_r', vmin=0, vmax=np.max(data))
    
    for x in range(n_col + 1):
        plt.axvline(x * w - 0.5, color='black', linewidth=1)
        
    for y in range(n_row + 1):
        plt.axhline(y * h - 0.5, color='black', linewidth=1)
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()

show_grid_with_lines(X_sample)
show_grid_with_lines(X_ReLuNMD[random_indices])
show_grid_with_lines(X_reconstruit)