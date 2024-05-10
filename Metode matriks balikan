import numpy as np

def inverse_matrix_method(A, b):
    # Kode pengecekan apakah matriks A memiliki invers
    if np.linalg.det(A) == 0:
        raise ValueError("Matriks koefisien tidak memiliki invers.")
    
    # Kode untuk menghitung invers dari matriks A
    A_inv = np.linalg.inv(A)
    
    # Kode untuk mencari solusi x dengan menggunakan invers matriks A
    x = np.dot(A_inv, b)
    
    return x

# Kode testing
if __name__ == "__main__":
    # Contoh soal
    A = np.array([[2, 1], [1, -1]])
    b = np.array([[4], [1]])

    # Solusi
    try:
        x = inverse_matrix_method(A, b)
        print("Solusi x:")
        print(x)
    except ValueError as e:
        print(e)
