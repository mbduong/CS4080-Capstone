import numpy as np

class GaussianElimination:
    
    def __init__(self):
        self.size = 0
        self.option = 0
        self.augcoeff = None
        self.sol = None
        self.bVal = None
        self.index = None
    
    def run(self):
        print("Welcome to the Gaussian Elimination with Scaled Partial Pivoting Program")
        print("")
        
        # Ask user for the number of linear equations to solve (10 or below)
        while self.size > 10 or self.size < 1:
            self.size = int(input("Please enter the number of size that will be needed (Note: Must be between 1 and 10): "))
        
        self.augcoeff = np.zeros((self.size, self.size))
        self.bVal = np.zeros(self.size)
        self.index = np.arange(self.size)
        self.sol = np.zeros(self.size)
        
        # Ask user the choice to enter file or own input
        while self.option != 1 and self.option != 2:
            print("Choose method of input:")
            print("1. Enter coefficients manually")
            print("2. Enter a file name")
            self.option = int(input())
        
        if self.option == 1:
            self.manualCoeff()
        elif self.option == 2:
            file_name = input("Please Enter File Name. Note: don't add .txt: ")
            try:
                self.augcoeff = self.fileScan(file_name)
            except FileNotFoundError:
                print("Error Reading File, closing program")
                return
        
        print("")
        print("Entered coefficients: ")
        self.printcoeff()
        
        print("Now Solving...")
        self.SPPGaussian()
        self.printResult()
    
    def SPPGaussian(self):
        self.SPPFwdElimination()
        self.SPPBackSubst()
    
    def SPPFwdElimination(self):
        n = self.augcoeff.shape[0]
        scale = np.zeros(n)
        
        # Scaling the rows
        for i in range(n):
            smax = 0
            for j in range(n):
                smax = max(smax, abs(self.augcoeff[i][j]))
            scale[i] = smax
        
        # Forward elimination
        for k in range(n - 1):
            rmax = 0
            j = k
            for i in range(k, n):
                r = abs(self.augcoeff[self.index[i]][k] / scale[self.index[i]])
                if r > rmax:
                    rmax = r
                    j = i
            # Swap indices
            self.swap(j, k)
            
            for i in range(k + 1, n):
                xmult = self.augcoeff[self.index[i]][k] / self.augcoeff[self.index[k]][k]
                self.augcoeff[self.index[i], k] = xmult
                for j in range(k + 1, n):
                    self.augcoeff[self.index[i], j] -= xmult * self.augcoeff[self.index[k], j]
            
            # Print out scaled ratio and pivot row
            print("\nScaled Ratio: ", rmax)
            print("Pivot Row: ", self.index[k] + 1)
            self.printcoeff()
    
    def swap(self, j, k):
        temp = self.index[j]
        self.index[j] = self.index[k]
        self.index[k] = temp
    
    def SPPBackSubst(self):
        n = self.augcoeff.shape[0]
        
        # Perform forward substitution
        for k in range(n - 1):
            for i in range(k + 1, n):
                self.bVal[self.index[i]] -= self.augcoeff[self.index[i], k] * self.bVal[self.index[k]]
        
        self.sol[n - 1] = self.bVal[self.index[n - 1]] / self.augcoeff[self.index[n - 1], n - 1]
        
        # Perform back substitution
        for i in range(n - 2, -1, -1):
            sum = self.bVal[self.index[i]]
            for j in range(i + 1, n):
                sum -= self.augcoeff[self.index[i], j] * self.sol[j]
            self.sol[i] = sum / self.augcoeff[self.index[i], i]
    
    def printcoeff(self):
        for i in range(self.size):
            for j in range(self.size + 1):
                if j < self.size:
                    print(f"{self.augcoeff[i][j]:.2f}", end=" ")
                else:
                    print(f"{self.bVal[i]:.2f}", end=" ")
            print()
    
    def printResult(self):
        letter_list = "xyzabcdefghijklmnopqrstuvw"
        print("\nFinal Output: ")
        for i in range(len(self.sol)):
            print(f"{letter_list[i]} = {self.sol[i]:.2f}")
    
    def manualCoeff(self):
        print("Please enter coefficient: ")
        for i in range(self.size):
            for j in range(self.size):
                print(f"Row: {i}")
                print(f"COL: {j}")
                self.augcoeff[i][j] = float(input())
            print(f"Please enter b-value for equation {i}:")
            self.bVal[i] = float(input())
    
    def fileScan(self, file_name):
        try:
            with open(f"{file_name}.txt", "r") as file:
                for i in range(self.size):
                    line = file.readline().strip()
                    row_values = list(map(float, line.split()))  # Split the row by space and convert to float
                    self.augcoeff[i, :] = row_values[:-1]  # All except the last value are coefficients
                    self.bVal[i] = row_values[-1]  # The last value is the b-value
        except FileNotFoundError:
            raise FileNotFoundError
        return self.augcoeff


# Main execution
if __name__ == "__main__":
    gauss = GaussianElimination()
    gauss.run()
