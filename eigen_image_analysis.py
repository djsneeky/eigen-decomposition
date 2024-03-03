import numpy as np
import matplotlib.pyplot as plt

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
    # (4096, 312) - columns of image data, every 26 is the same character

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
    # (10, 312) - reduced column vectors, 12 for each class

    # Compute class means and covariances for each of the 26 classes
    class_data = {}
    for i in range(26):
        class_cols = np.array(get_every_26th_column(Y_train, i))
        # print(class_cols.shape)
        class_mean = np.mean(class_cols, axis=1, keepdims=True)
        # print(class_mean.shape)
        class_cov = np.cov(class_cols, rowvar=True)
        # col_sum = np.zeros((10, 10))
        # for j in range (0, len(class_cols[0])):
        #     col = np.array(class_cols[:, j])
        #     col_sum += (np.matmul(col - class_mean, (col - class_mean).T))
        # class_cov = col_sum / 25
        # print(class_cov.shape)
        class_data[i] = {'mean': class_mean, 'cov': class_cov}

    # Read the test images
    X_test = read_data.read_test_data()
    # (4096, 26)

    # Use the same A and mu_hat!
    # Center the test data by subtracting the mean image
    X_centered_test = X_test - mu_hat

    # Transform the test data to a lower-dimensional representation
    Y_test = np.dot(A.T, X_centered_test)
    # (10, 26)

    # Classification
    misclassified = []
    for i in range(26):
        distances = []
        for j in range(26):
            diff = Y_test[:, [i]] - class_data[j]['mean']
            distances.append(diff.T @ np.linalg.inv(class_data[j]['cov']) @ diff + np.log(np.linalg.det(class_data[j]['cov'])))
        min_distance_index = np.argmin(distances)
        
        if min_distance_index != i:
            misclassified.append((chr(97 + i), chr(97 + min_distance_index)))

    # Display misclassified characters
    print("Misclassified characters:")
    for item in misclassified:
        print(f"Character '{item[0]}' misclassified as '{item[1]}'")
        
    # modified image classification section
    # 1. Let Bk = Λk
    lambda_k = []
    for i in range(26):
        diag = np.diag(np.diag(class_data[i]['cov']))
        lambda_k.append(diag)
    misclassified_1 = classify(Y_test, class_data, lambda_k)

    # 2. Let Bk = Rwc
    Rwc = np.zeros((10, 10))
    for i in range(26):
        Rwc += class_data[i]['cov']
    Rwc = Rwc / 26
    Rwc_list = [Rwc.copy()] * 26
    misclassified_2 = classify(Y_test, class_data, Rwc_list)

    # 3. Let Bk = Λ (Diagonal of Rwc)
    Rwc_diag_list = [np.diag(np.diag(Rwc)).copy()] * 26
    misclassified_3 = classify(Y_test, class_data, Rwc_diag_list)

    # 4. Let Bk = I
    I = np.eye(10)
    I_list = [I.copy()] * 26
    misclassified_4 = classify(Y_test, class_data, I_list)

    # Display misclassified characters for each modification
    print("Misclassified characters with different Bk matrices:")
    print("1. Bk = Λk:")
    print(misclassified_1)
    print("2. Bk = Rwc:")
    print(misclassified_2)
    print("3. Bk = Λ (Diagonal of Rwc):")
    print(misclassified_3)
    print("4. Bk = I:")
    print(misclassified_4)
        
def classify(test_data, class_data, Bk):
    misclassified = []
    for i in range(26):
        distances = []
        for j in range(26):
            diff = test_data[:, [i]] - class_data[j]['mean']
            distances.append(diff.T @ np.linalg.inv(Bk[j]) @ diff + np.log(np.linalg.det(Bk[j])))
        min_distance_index = np.argmin(distances)
        
        if min_distance_index != i:
            misclassified.append((chr(97 + i), chr(97 + min_distance_index)))

    return misclassified

def get_every_26th_column(matrix, start_offset):
    result = []
    for i in range(start_offset, len(matrix[0]), 26):
        result.append([row[i] for row in matrix])
    return np.array(result).T

if __name__ == "__main__":
    main()