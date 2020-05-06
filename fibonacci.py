n=int(input("enter the number of digits you want in fibonacci series:"))

a=0
b=1
mylist=[a,b]
limit=2
while (limit<n):
    c=a+b
    a=b
    b=c
    mylist.append(c)
    limit+=1
for i in mylist:
    print(i,end=",")
