import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

from training_data import read_data

def main():
    # covariance matrix
    RX = np.array([[2, -1.2],
                   [-1.2, 1]])
    
    # generate n = 1000 samples of iid N(0, I) gaussian random vectors
    n = 1000
    p = 2
    W = np.random.randn(p, n)
    
    # calculate the eigen-decomposition of RX
    eigenvalues, eigenvectors = np.linalg.eig(RX)
    
    # generate scaled random vectors X~
    X_tilde = np.sqrt(np.diag(eigenvalues)).dot(W)
    
    # generate samples Xi by applying transformation Xi = E X~
    X = eigenvectors.dot(X_tilde)
    
    
    
    # # Scatter plot of W
    # plt.figure()
    # plt.plot(W[0, :], W[1, :], '.', label='W')
    # plt.axis('equal')
    # plt.title('Scatter plot of W')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.show()

    # # Scatter plot of X~
    # plt.figure()
    # plt.plot(X_tilde[0, :], X_tilde[1, :], '.', label='X~')
    # plt.axis('equal')
    # plt.title('Scatter plot of X~')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.show()

    # # Scatter plot of X
    # plt.figure()
    # plt.plot(X[0, :], X[1, :], '.', label='X')
    # plt.axis('equal')
    # plt.title('Scatter plot of X')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.show()
    
    # 1: estimate covariance
    # Compute the sample mean
    mu_hat = np.mean(X, axis=1, keepdims=True)

    # Compute the mean-centered data vectors
    Z = X - mu_hat

    # Compute the covariance estimate
    R_hat = (1 / (n - 1)) * np.dot(Z, Z.T)

    # Print the covariance estimate
    print("Covariance Estimate (R_hat):", R_hat)

    # Step 2: Compute the transformation to decorrelate the samples
    # Eigen-decomposition of the covariance estimate
    eigenvalues_hat, eigenvectors_hat = np.linalg.eig(R_hat)

    # Compute the decorrelation transformation
    D = np.diag(1 / np.sqrt(eigenvalues_hat))
    V = eigenvectors_hat
    T_decorrelation = np.dot(np.dot(V, D), V.T)

    # Decorrelate the samples
    X_tilde_decorrelated = np.dot(T_decorrelation, Z)

    # Step 3: Compute the transformation to fully whiten the samples
    T_whitening = np.dot(np.diag(1 / np.sqrt(eigenvalues_hat)), V.T)

    # Whiten the samples
    W_whitened = np.dot(T_whitening, Z)

    # # Step 4: Scatter plots of X~ and Wi
    # # Scatter plot of decorrelated samples
    # plt.figure()
    # plt.plot(X_tilde_decorrelated[0, :], X_tilde_decorrelated[1, :], '.', label='X~')
    # plt.axis('equal')
    # plt.title('Scatter plot of decorrelated samples')
    # plt.xlabel('X~_1')
    # plt.ylabel('X~_2')
    # plt.show()

    # # Scatter plot of whitened samples
    # plt.figure()
    # plt.plot(W_whitened[0, :], W_whitened[1, :], '.', label='W')
    # plt.axis('equal')
    # plt.title('Scatter plot of whitened samples')
    # plt.xlabel('W_1')
    # plt.ylabel('W_2')
    # plt.show()

    # Step 5: Compute the covariance estimate of the whitened samples
    R_hat_whitened = (1 / (n - 1)) * np.dot(W_whitened, W_whitened.T)
    print("Covariance Estimate of Whitened Samples (R_hat_whitened):\n", R_hat_whitened)
    
    image_classify()
    
# Section 4    
def eigenimages():
    # Read the images into a vector X
    X = read_data.read_data()

    # Compute the mean image over the entire dataset
    mu_hat = np.mean(X, axis=1, keepdims=True)

    # Center the data by subtracting the mean image from each column of X
    X_centered = X - mu_hat

    # Compute the SVD of the centered data matrix X
    U, Sigma, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # Display the eigenimages associated with the 12 largest eigenvalues
    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    axes = axes.flatten()

    for i in range(12):
        # Reshape the eigenimage column vector into a 64x64 matrix
        eigenimage = U[:, i].reshape((64, 64))
        
        # Display the eigenimage
        axes[i].imshow(eigenimage, cmap=plt.cm.gray)
        axes[i].set_title(f"Eigenvalue {Sigma[i]:.2f}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()
    
    # Compute the projection coefficients Y = U^T(X - mu_hat)
    Y = np.dot(U.T, X_centered)

    # Plot the first 10 projection coefficients for the first four images
    plt.figure(figsize=(10, 6))
    for i in range(4):
        plt.plot(range(1, 11), Y[:10, i], label=f'Image {chr(97 + i)}')

    plt.title('Projection Coefficients for the First Four Images')
    plt.xlabel('Projection Coefficient Index')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Synthesize the original image using the first m eigenvectors for m = 1, 5, 10, 15, 20, 30
    m_values = [1, 5, 10, 15, 20, 30]

    fig, axs = plt.subplots(3, 2, figsize=(12, 12))

    for i, m in enumerate(m_values):
        # Synthesize the image using the first m eigenvectors
        synthesized_image = np.dot(U[:, :m], Y[:m, 0][:, np.newaxis]) + mu_hat
        synthesized_image = synthesized_image.reshape((64, 64))
        
        # Display the synthesized image
        axs[i // 2, i % 2].imshow(synthesized_image, cmap=plt.cm.gray)
        axs[i // 2, i % 2].set_title(f'Synthesized Image (m = {m})')
        axs[i // 2, i % 2].axis('off')

    plt.tight_layout()
    plt.show()

    # show the orig image
    original_image = X[:, 0].reshape((64, 64))
    plt.figure()
    plt.imshow(original_image, cmap=plt.cm.gray)
    plt.title('Original Image')
    plt.axis('off')
    plt.show()
    
# Section 5
def image_classify():
    # Read the training images
    X_train = read_data.read_data()

    # Compute the mean image over the entire dataset
    mu_hat = np.mean(X_train, axis=1, keepdims=True)

    # Center the training data by subtracting the mean image
    X_centered_train = X_train - mu_hat

    # Compute the SVD of the centered training data matrix
    U, Sigma, Vt = np.linalg.svd(X_centered_train, full_matrices=False)

    # Form the transformation matrix A using the first 10 eigenvectors
    A = U[:, :10]

    # Transform each of the original training images to a lower-dimensional representation
    Y_train = np.dot(A.T, X_centered_train)

    # Compute class means and covariances for each of the 26 classes
    class_data = {}
    for i in range(26):
        class_data[i] = {'mean': np.mean(Y_train[:, i*10:(i+1)*10], axis=1),
                        'cov': np.cov(Y_train[:, i*10:(i+1)*10])}

    # Read the test images
    X_test = []
    for i in range(26):
        image = plt.imread(f'test_data/veranda/{chr(97+i)}.tif')
        X_test.append(image.flatten())

    X_test = np.array(X_test).T

    # Use the same A and mu_hat!
    # Center the test data by subtracting the mean image
    X_centered_test = X_test - mu_hat

    # Transform the test data to a lower-dimensional representation
    Y_test = np.dot(A.T, X_centered_test)

    # Classification
    misclassified = []
    for i in range(26):
        diff = Y_test - class_data[i]['mean'][:, np.newaxis]
        distances = np.sum(np.dot(diff.T, np.linalg.inv(class_data[i]['cov'])) * diff.T, axis=1)
        min_distance_index = np.argmin(distances)
        
        if min_distance_index != i:
            misclassified.append((chr(97 + i), chr(97 + min_distance_index)))

    # Display misclassified characters
    print("Misclassified characters:")
    for item in misclassified:
        print(f"Character '{item[0]}' misclassified as '{item[1]}'")
        
    # modified image classification section
    # Initialize a list to store misclassified characters for each modification
    misclassified_modifications = []

    # Modification 1: Bk = Λk
    Bk_mod1 = [np.diag(class_data[i]['cov']) for i in range(26)]

    # Modification 2: Bk = Rwc
    Rwc = np.mean([class_data[i]['cov'] for i in range(26)], axis=0)
    Bk_mod2 = [Rwc] * 26

    # Modification 3: Bk = Λ
    Bk_mod3 = [np.diag(Rwc)] * 26

    # Modification 4: Bk = I
    Bk_mod4 = [np.eye(class_data[i]['cov'].shape[0]) for i in range(26)]

    # Classify test data using each modification
    for Bk, mod_name in [(Bk_mod1, 'Modification 1: Λk'), (Bk_mod2, 'Modification 2: Rwc'),
                        (Bk_mod3, 'Modification 3: Λ'), (Bk_mod4, 'Modification 4: I')]:
        misclassified = []
        for i in range(26):
            diff = Y_test - class_data[i]['mean'][:, np.newaxis]
            distances = np.sum(np.dot(diff.T, np.linalg.inv(Bk[i])) * diff.T, axis=1)
            min_distance_index = np.argmin(distances)
            
            if min_distance_index != i:
                misclassified.append((chr(97 + i), chr(97 + min_distance_index)))
        
        # Append misclassified characters for this modification to the list
        misclassified_modifications.append((mod_name, misclassified))

    # Display misclassified characters for each modification
    for mod_name, misclassified in misclassified_modifications:
        print(f"Misclassified characters for {mod_name}:")
        for item in misclassified:
            print(f"Character '{item[0]}' misclassified as '{item[1]}'")
        print()

if __name__ == "__main__":
    main()