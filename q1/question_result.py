from q1.generate_data import generate_data
from q1.plot_dataset import plot_dataset
from q1.split_dataset import train_test_split
from q1.calculate_mean_cov import compute_class_stats_matrices, compute_shared_covariance
from q1.qdc import qda_predict, plot_decision_boundary
from q1.calculate_error import classification_error
from q1.ldc import lda_predict, plot_lda_decision_boundary


def run():
    mus = [[1, 2], [3, 5]]
    sigmas = [[[2, 1], [1, 3]], [[1, -0.8], [-0.8, 3]]]
    number_of_classes = 2
    number_of_samples = 1000

    X, y = generate_data(mus, sigmas, number_of_class=number_of_classes, number_of_samples=number_of_samples)

    # question 1-a: plot dataset
    plot_dataset(X, y, mus, sigmas)
    # question 1-b: split data into %80 train and %20 test and calculate mean, cov matrices without using built-in methods
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_ratio=0.2)
    mus, sigmas = compute_class_stats_matrices(x_train, y_train, number_of_class=number_of_classes)
    for i in range(number_of_classes):
        print(f"Mean vector of class {i} is: {mus[i]}")
        print(f"Covariance matrices of class {i} is: {sigmas[i]}")

    # question 1-c: Design a quadratic discriminant classifier using different covariance matrices. Obtain the training and test errors. Draw the decision boundary.
    y_train_pred = qda_predict(x_train, mus, sigmas)
    y_test_pred = qda_predict(x_test, mus, sigmas)

    train_err = classification_error(y_train, y_train_pred)
    test_err = classification_error(y_test, y_test_pred)

    print(f"Train error: {train_err:.4f}")
    print(f"Test error: {test_err:.4f}")

    plot_decision_boundary(x_train, y_train, mus, sigmas)

    # question 1-d: Design a linear discriminant classifier using a shared covariance matrix. Obtain the training and test errors. Draw the decision boundary
    shared_sigma = compute_shared_covariance(x_train, y_train, mus, number_of_class=number_of_classes)
    y_train_pred_lda = lda_predict(x_train, mus, shared_sigma)
    y_test_pred_lda = lda_predict(x_test, mus, shared_sigma)

    test_err = classification_error(y_test, y_test_pred_lda)
    train_err = classification_error(y_train, y_train_pred_lda)

    print(f"Train error: {train_err:.4f}")
    print(f"Test error: {test_err:.4f}")

    plot_lda_decision_boundary(x_train, y_train, mus, shared_sigma)


if __name__ == "__main__":
    run()
