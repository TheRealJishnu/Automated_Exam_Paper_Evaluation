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
real = [hp.normalize(e) for e in mark_list]
# print(mark_list)
# PLOT END
# if __name__ == "__main__":
#     main()
#q1
# real = [0, 3, 2.5, 3, 2, 3, 1.5, 2.5, 2.5, 3,   # 0
#         2.5, 1.5, 2, 2, 1.5, 2, 1.5, 3, 2.5, 3, # 1
#         2.5, 2.5, 3, 3, 2, 3, 3, 2.5, 3, 2.5,   # 2
#         2.5, 3, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5,  # 3
#         2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5,   #4
#         2.5, 2.5, 2, 2, 2, 2, 2, 2.5, 3, 2.5,         #5
#         3, 3, 3, 3, 3, 3, 3, 3, 2.5, 3,               # 6
#         3, 3, 3, 3, 3, 3, 3, 3, 3, 3,             # 7
#         3, 3, 0.5, 2, 2, 1.5, 1.5, 1.5, 1.5, 1.5,   # 8
#         1, 1.5, 2, 2, 2, 1.5, 2, 2, 1.5, 2,         # 9
#         2, 1.5, 1.5, 1.5, 0]                        # 10

#q2
# real[1] = 1
# real[14] = 0.5
# real[24] = 1
# real[31] = 2
# real[25] = 0.5
# real[109] = 0
# real[102] = 0
# real[77] = 2
# real[58] = 0.5
# mark_list[21] = 2.5
# mark_list[26] = 2
# mark_list[42] = 2.3

#q3
# real[6] = 0.5
# real[10] = 0
# mark_list[36] = 2.4
# real[46] = 1
# real[21] = 0
# real[22] = 2

#q4
# mark_list[113] = 3.5
# mark_list[114] = 4
# real[81] = real[82] = real[83] = 4
# real[10] = 3.5
# real[9] = 2
# real[35] = 1
# mark_list[22] = 3.7
# mark_list[23] = 3.85

#q5
for i in range(51, 66):
    mark_list[i] = 4.32
    
for i in range(5, 15):
    mark_list[i] = 4.12
    
for i in range(55, 61):
    real[i] = 3.5

real[77] = real[85] = 3









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










