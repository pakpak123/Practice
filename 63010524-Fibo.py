def fibo(n):
    if n <= 1:
        return n
    else:
        return fibo(n-1) + fibo(n-2)


i = int(input("Enter Number : "))
print("fibo("+str(i)+") = " + str(fibo(i)))