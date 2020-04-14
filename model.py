import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data1 = open("ratings_data.txt","r")
lines = data1.readlines()
# 49,290 users who rated a total of
# 139,738 different items at least once, writing
# 664,824 reviews and
# 487,181 issued trust statements.

users = 49290
movies = 139738
reviews = 664824
trusts = 487181

user_movie = np.zeros((users,movies),dtype=np.int8)
rating = np.zeros((users,movies), dtype= np.int8)
for i in range(664824):
    try:
        user,movie,Rating = lines[i+1].split(" ")
        user_movie[int(user)-1][int(movie)-1] = Rating
        rating[(int(user)-1)][int(movie)-1] = 1
#         np.savetxt('rating.txt', user_movie, fmt="%d", header=header)
    except:
        print("No")
        continue

user_movie = user_movie[:500,:10000]
rating = rating[:500,:10000]

print("Rating Matrix")
print(user_movie)
print(np.shape(user_movie))
print("Rating Done or not")
print(rating)

data2 = open("trust_data.txt","r")
lines2 = data2.readlines()

user_user = np.zeros((users,users),dtype = np.int8)
for i in range(trusts):
    lines2[i+1] = lines2[i+1].lstrip()
    try:
        user1,user2,trust = lines2[i+1].split(" ")
        user_user[int(user1)-1][int(user2)-1] = trust

    except:
        print("no")
        continue

user_user = user_user[:500,:500]
print(user_user)
print(np.shape(user_user))

user_movie = user_movie / np.max(user_movie)
def sigmoid(X):
    X = 1 + np.exp(-X)
    X = 1/X
    return X
def sigmderivative(X):
    return sigmoid(X)*(1-sigmoid(X))

theta = np.zeros((np.shape(user_movie)))
print(theta)
def costFunction(X,Y,R,theta,lmd):

    J = np.sum(((sigmoid(X@theta)-Y)*R).transpose()@((sigmoid(X@theta)-Y)*R))/2+ lmd*(np.sum(theta.transpose()@theta))/2
#     print(J)
    grad = np.zeros(np.shape(theta))
    grad = X.transpose()@sigmderivative(X@theta)*(sigmoid(X@theta) - Y)*R + lmd*theta

    return J/Y.shape[0],grad

print(costFunction(user_user,user_movie,rating,theta,0))

def gradientDescent(X,Y,R,theta,alpha,lmd,iterations):
    J_history = np.zeros(iterations)
    for iteration in range(iterations):
        print(iteration)
        J_history[iteration] ,grad = costFunction(X,Y,R,theta,lmd)
        theta = theta - alpha*grad
    return theta,J_history

theta = np.zeros((np.shape(user_movie)))

# theta, J_history = gradientDescent(user_user,user_movie,rating,theta,0.005,0,50)
# plt.plot(J_history)
# print(J_history)

max = np.max(sigmoid(user_user@theta))
min = np.min(sigmoid(user_user@theta))

new_Y = sigmoid(user_user@theta)*(1-rating) >= (max+min)/2

for i in range(500):
    if np.sum(new_Y[i]==True) == 0:
        if np.sum(user_user[i]==1) != 0:
             print("False")
        else:
            print("No trust")
