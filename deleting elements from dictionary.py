#DELETING DIFFERENT DICTIONARY ELEMENTS
n=int(input("enter the number of elements you want in dictionary:"))
dict1={}
for i in range(n):
    key=input("enter the key: ")
    value=input("enter the value: ")
    dict1[key]=value
print("here is your dictionary:",dict1)
while True:
    element=input("enter the key you want to delete:")
    del dict1[element]
    print("Here is how your dictionary looks like now: ",dict1)
    choice=input("do you want to delete any other element?y/n ")
    if choice=="y":
        True
    else:
        break
print("Here is your modified dictionary: ",dict1)
    
