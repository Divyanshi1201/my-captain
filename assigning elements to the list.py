#the program for assigning elements to the list

n=input("Enter the elements of the list seperated by space: ")
list1=n.split()

print("Here is your list: ",list1)

print("\nMETHOD 1: APPENDING AN ELEMENT TO THE LIST")

while True:
    choice=input("Do you want to add element to the list y/n: ")
    
    if choice=="y":
        typ=input("Type the element you want to add: ")
        list1.append(typ)
        break
    elif choice=="n":
        break
    else:
        print("Sorry but that is invalid response. please type 'y' or 'n'")   
    
    
print("\nHere is your appended list: ",list1)

print("\nMETHOD 2: ADDING ONE LIST TO ANOTHER USING EXTEND FUNCTION")

while True:
    choice2=input("do you want to add another list to the previously created list?y/n: ")
    if choice2=="y":
        n2=input("enter the elements of new list seperated by space: ")
        list2=n2.split()
        list1.extend(list2)
        break
    elif choice2=="n":
        break
    else:
        print("Sorry but that is invalid response. please type 'y' or 'n'")   

print("\nHere is your extended list: ",list1)

print("\nMETHOD 3:ADDING THE ELEMENT AT PARTICULAR POSITION USING INSERT FUNCTION ")            
            
while True:
    choice3=input("Do you want to add a particular element at a particular position?y/n ")

    if choice3=="y":
        element=input("enter the element you want to add: ")
        position=int(input("enter the position at which you want to add the element: "))
        list1.insert(position,element)
        break
    elif choice3=="n":
        break
    else:
        print("Sorry but that is invalid response. please type 'y' or 'n'")   

print("\nHere is your final list: ",list1)

        
