n=int(input("Enter number of elements you want in input list:"))
mylist=[]
for i in range(n):
    l=int(input("Enter the element:"))
    mylist.append(l)
final=[]
for number in mylist:
    if number>=0:
        final.append(number)
print("List of positive numbers:",final)

        
