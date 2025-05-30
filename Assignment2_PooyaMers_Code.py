import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

#Change wd as per personal file directory
os.chdir("C:/Users/4dmer/Desktop/Study material/Year 2/Multivariate Statistics/Assignment/2")

# Data preparation
read = pd.read_csv("popmusic.csv")
billie_index = read.loc[read['artist'] == 'Billie Eilish'].index
song_count = read['artist'].value_counts()
print("The number of artists in df: ", np.count_nonzero(read['artist'].unique()))
print("The number of songs in df per artist:\n", song_count)
print("Total song count: ", np.sum(song_count))
X = read.select_dtypes(include=np.number)
X = X.drop(columns=['popul'])

Col_names = list(X.columns.values)


X = X.values
n = len(X)

"""
--------------------------------------------------------------------------------------------
Principal Component Analysis: Reduce dimension of X from 10 to 4
--------------------------------------------------------------------------------------------
"""

#Normalize the data
def Normalize(x):
    H = np.identity(n) - (1 / n) * np.ones((n, n))
    D = np.diag(np.var(x, axis=0))

    return 1/np.sqrt(n) * np.linalg.multi_dot([H, x, np.linalg.inv(np.sqrt(D))])

#Perform PCA
def PCA(X):
    # Normalize data
    X = Normalize(X)

    X_T_X = np.dot(np.transpose(X), X)
    eigenvalues, eigenvectors = np.linalg.eig(X_T_X)
    biggest_4 = np.sort(eigenvalues)[::-1][:4]

    # Take index of four biggest eigenvalues
    index = np.zeros(len(biggest_4), dtype=int)
    for i in range(len(biggest_4)):
        if biggest_4[i] in eigenvalues:
            index[i] = (np.where(eigenvalues == biggest_4[i])[0][0])

    # Responses corresponding to 4 largest ev.s
    #y = np.take(eigenvectors, index.astype(int), axis=0)
    y = eigenvectors[:, index]

    # The factors:
    F = np.dot(X, eigenvectors)

    # Taking only the most important variables of X
    PCA_X = np.dot(X, y)


    # print(np.transpose(y))

    return PCA_X, eigenvalues, eigenvectors, F, y, index, biggest_4

#Calculate proportion of variance retained
def var_explained(eigenvalues, num):
    eigenvalues = np.sort(eigenvalues)[::-1]
    return (sum(eigenvalues[:num]) / sum(eigenvalues))

#Make scree plot
def scree_plot(eigenvalues):
    x = np.arange(1, len(eigenvalues), 1)
    fig, ax1 = plt.subplots()
    varex = [var_explained(eigenvalues, i) for i in range(1, len(eigenvalues))]

    ax1.plot(x, varex, marker='o', linestyle='dashed', label="Variance Explained")

    # ax1.plot(x, np.sort(eigenvalues)[::-1], marker='o', linestyle='dashed')
    ax1.set_xlabel("PC index")
    ax1.set_ylabel("Cumulative Explained Variance Ratio")
    ax1.set_title("Scree Plot of the Principal Components")
    ax1.axhline(y=0.6635, xmin= 0, xmax= 0.39, color='gray', linestyle='dashed')
    ax1.set_yticks(np.append(np.arange(0, 1.2, 0.2), 0.6635))
    ax1.set_ylim([(var_explained(eigenvalues, 1) - 0.1 * var_explained(eigenvalues, 1)), 1])
    fig.tight_layout()

    x2 = np.arange(1, len(eigenvalues)+1, 1)
    fig2, ax2 = plt.subplots()
    ax2.set_ylabel("Eigenvalue")
    ax2.set_xlabel("PC index")
    ax2.plot(x2, np.sort(eigenvalues)[::-1], marker='x', linestyle='dashed', color='red', label="Eigenvalue")
    ax2.set_title("Eigenvalues of the Principal Components")

    plt.xticks(x)
    fig2.tight_layout()
    plt.show()


#Plot the PCs
def plot_vars(X, billie=False):
    NX = X
    index = NX[0]
    if billie:
        min_billie = min(billie_index)
        max_billie = max(billie_index)
        for i in range(len(index)):
            for j in range(i + 1, len(index)):
                plt.scatter(NX[:min_billie, i], NX[:min_billie, j], color = 'blue', label = 'By Other Pop-Stars')
                plt.scatter(NX[max_billie+1:, i], NX[max_billie+1:, j], color = 'blue')

                plt.scatter(NX[min_billie:max_billie+1, i], NX[min_billie:max_billie+1, j], marker = 'x', color = 'red', label = 'By Billie Eilish')
                plt.xlabel("PC " + str(i+1))
                plt.ylabel("PC " + str(j+1))
                plt.title("PC " + str(i+1) + " and PC " + str(j+1) + " of the " + '$\it{normalized}$ ' + "and " +'$\it{compressed}$ ' + "pop-music data")
                #plt.xlim((-0.21, 0.21))
                #plt.ylim((-0.21, 0.21))
                plt.legend(loc = 3)

                #plt.savefig(('d' + str(i+1) + str(j+1)))
                plt.show()

    else:
        for i in range(len(index)):
            for j in range(i + 1, len(index)):
                plt.scatter(NX[:, i], NX[:, j], color = 'blue')
                plt.xlabel("PC " + str(i+1))
                plt.ylabel("PC " + str(j+1))
                plt.title("PC " + str(i+1) + " and PC " + str(j+1) + " of the " + '$\it{normalized}$ ' + "and " +'$\it{compressed}$ ' +  "pop-music data")
                #plt.xlim((-0.3, 0.3))
                #plt.ylim((-0.3, 0.3))
                plt.show()


#Output:

print("Compressed dataset using four Principal Components:\n", PCA(X)[0])
print("The % variance retained is: ", np.round(var_explained(PCA(X)[1], 4) * 100, 2))
scree_plot(PCA(X)[1])
#plot_vars_1(X, True)


#Present loadings chart
Chart = pd.DataFrame(np.round(PCA(X)[4],3), columns = ("PC1", "PC2", "PC3", "PC4"), index = Col_names)
#print(Chart)
#Chart.to_csv("Chart.csv")

#Compute squared correlation for PC/variable
gamma = PCA(X)[4]
L = PCA(X)[6]
Corr = np.zeros(((len(gamma)), len(gamma[0])))
for k in range(len(gamma[0])):
    for j in range(len(gamma)):
        Corr[j][k] = (np.dot(np.sqrt(L[k]), gamma[j][k]))

Corrsq = pd.DataFrame(np.round(np.power(Corr, 2),2), columns = ("PC1", "PC2", "PC3", "PC4"), index = Col_names)
#Corrsq.to_csv("Corrsq.csv")
#print(Corrsq)

#PC plots and Billie
plot_vars(PCA(X)[0], billie = False)
plot_vars(PCA(X)[0], billie = True)
