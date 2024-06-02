# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 18:55:25 2023

@author: HP-PC
"""

import evaluation as ev
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
import helper as hp

v_list = []
mark_list = []
# def main():
print("--------------Welcome to the Exam Paper Evaluation Project!------------------")
ls = os.listdir("Result//")
print(ls)
if os.path.exists("Result//EvaluatedAnswers.txt"):
    # ch = input("Previous Result Already Exists\nDo you want to Delete Previous? ('y' to delete, 'n' to exit) : ")
    ch = 'y'
    if(ch == 'y'):
        os.remove("Result//EvaluatedAnswers.txt")
        print("Program Continuing...")
    else:
        print("Program Exiting...")
        sys.exit(0)

# no = int(input("Enter Number of Questions : "))
no = 5

for i in range(5, no+1):
    v_list, mark_list = ev.evaluation_main(i)
    # print()
    break

print("Your Result is saved in the text file named EvaluatedAnswers.txt")


# PLOT









lm = len(mark_list)
x = np.arange(1, lm+1)
y1 = mark_list
plt.figure(figsize=(10,5))
plt.plot(x, y1, label="Predicted")
plt.plot(x, real, label="Original")
plt.xlabel("Answer No. $\longrightarrow$")
plt.ylabel("Marks $\longrightarrow$")
plt.title("Question 5 Original and Predicted Marks")
plt.xticks(x[::5], rotation='vertical')
plt.grid()
plt.legend()

acc_list = []
sm = 0
for i, j in zip(y1, real):
    sm += (i-j)**2
    acc_list.append(sm)
        
# for i in range(1, 104):
#     sm += acc_list[i]
    
print(sm/lm)
print(mse(y1, real))
    
    

plt.savefig("q5_Marking.png", dpi=1200, format='png', bbox_inches = 'tight')










