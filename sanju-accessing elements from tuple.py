# -*- coding: utf-8 -*-
"""
Created on Fri May  8 11:23:30 2020

@author: MUTHU PS
"""

n=input("Enter the elemnts of tuple seperated by white spaces: ")
nums=n.split()
tup=tuple(nums)
print("here is your tuple:",tup)

while True:
    ind=int(input("Please note that indexing starts from 0.\nEnter the index of the element you want to access: "))

    if ind<len(tup):
        element=tup[ind]
        break
    else:
        print("\nIndex is out of range. Please try again")
        continue
        
print("Your element is: ",element)
