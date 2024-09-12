1.#write a program to find GCD of two numbers

def gcd(a,b):
   while b:
        a, b = b, a % b
   return a

a =int(input("enter the first number"))
b =int(input("enter the second number"))
print(gcd(a,b))



2.#palindrome or not 
def isPalindrome(s):
    return s == s[::-1]

s = "river"
ans = isPalindrome(s)

if ans:
   print("Yes")
else:
   print("No")



3.#maximum of list of numbers
list_nums=[]
num = int(input("Enter the num of elements in list: "))

for i in range (num):
  ele = int (input("Enter the element: "))
  list_nums.append(ele)

print("Maximum element in the list is: ", max(list_nums))

4.#Menu program
def add(x, y):
  return x + y

def subtract(x, y):
  return x - y

def multiply(x, y):
  return x * y

def divide(x, y):
  if y == 0:
    return "Division by zero is not allowed."
  else:
    return x / y
3
print("Select operation:")
print("1. Add")
print("2. Subtract")
print("3. Multiply")
print("4. Divide")


choice = input("Enter choice(1/2/3/4): ")

num1 = float(input("Enter first number: "))
num2 = float(input("Enter second number: "))

if choice == '1':
  print(num1, "+", num2, "=", add(num1, num2))

elif choice == '2':
  print(num1, "-", num2, "=", subtract(num1, num2))

elif choice == '3':
  print(num1, "*", num2, "=", multiply(num1, num2))

elif choice == '4':
  print(num1, "/", num2, "=", divide(num1, num2))

else:
  print("Invalid input")


5.#write a program to find squareroot using newton method

def sqrt (n):
  a = 0.5*n
  b = 0.5*(a+n/a)
  while a!=b:
    a = b
    b= 0.5*(a+n/a)
  return a

n = int(input("enter the number"))
print(sqrt(n))



6.#factorial of number using function
def fact (n):
  f = 1
  for i in range (1 , n+1):
      f = f * i
  return f

print ( fact(6))


7.#check equality of list
list1 = [1, 2, 3, 4, 5]
list2 = [2, 3, 4, 5, 1]

if list1==list2:
    print ('True')

else:
  print("False")


8.#Merage two python dictionary
def merge_dicts(dict1,dict2):
  result = dict1.copy()
  result.update(dict2)
  return result

dict1 = {"a": 2, "b": 3}
dict2 = {"c": 4, "d": 5}
merged_dict = merge_dicts(dict1,dict2)
print(merged_dict)


9.#Sort of dictionary by values using the function

def sort_dict_by_values(dictionary):
    return dict(sorted(dictionary.items(), key=lambda item: item[1]))

my_dict = {'apple': 5, 'banana': 2, 'cherry': 8}
sorted_dict = sort_dict_by_values(my_dict)
print(sorted_dict)


10.#Check given key exists in a dictionary
def check_key(dict, key):
  if key in dict:
    print("The key is presnt:", dict[key])

  else:
    print("the key is not present")

sample = {'a':6, 'b': 7, 'c': 9}

check_key(sample,'a')
check_key(sample,'c')


11.# to find shortest list of values with the keys in a given dictionary

def check1(dict):
 minval=min([len(dict[ele])for ele in dict])
 res=[]
 for ele in dict1:
    if len(dict1[ele])==minval:
      res.append(ele)
 print("the list of keys with lowest values are:",res)
dict1={'k':[30,40],'b':[56,43,232],'c':[12,34],'d':[32,23,21],'g':[43,23]}
check1(dict1)

12.#to check a number is spy numbers or not
n=int(input("enter a number:"))
s=0
p=1
while n>0:
   r=n%10
   s=s+r
   p=p*r
   n=n//10
if s==p:
   print("the number is a spy number")
else:
   print("the numbers is not a spy numbers")


13.#to prime all twin prime numbers
def prime(n):
  for i in range(2,n):
    if n%i==0:
      return False
  return True
n=int(input("enter a numbers:"))
for p in range (2,n-1):
  if prime(p) and prime(p+2):
    print(p,p+2)

14.#to convert decimal to hexadecimal using class and methods
def decimal_to_hexadecimal(decimal):
    hexadecimal = ""
    hex_digits = "0123456789ABCDEF"
    while decimal > 0:
        remainder = decimal % 16
        hexadecimal = hex_digits[remainder] + hexadecimal
        decimal //= 16
    return hexadecimal

decimal_number = 2000
hexadecimal_number = decimal_to_hexadecimal(decimal_number)
print(hexadecimal_number)


15.#to find sum of two numbers using and class and methods
class sum(object):
  def __init__ ( self, num1, num2):
    self.num1=num1
    self.num2=num2
  def find_sum(self):
    val=self.num1+self.num2
    print(val)
s=sum(10,20)
s.find_sum()

16.# Student result using classes and object
class Student:
  reg=int(input("Enter registration number:"))
  m1=int(input("Enter mark1:"))
  m2=int(input("Enter mark2:"))
  m3=int(input("Enter mark3:"))
  tot=m1+m2+m3
  avg=tot/3
  if(avg<40):
    result="Fail"
  else:
    result="Pass"

s=Student()
print("Registration number:",s.reg)
print("Mark1:",s.m1)
print("Mark2:",s.m2)
print("Mark3:",s.m3)
print("Total:",s.tot)
print("Average:",s.avg)
print("Result:",s.result)

17.#To find area of rectangle circle and square using hierarchical inheritance
class shape:
  def __init__(self):
     pass

class rectangle(shape):
  def __init__(self,length,width):
    self.length=length
    self.width=width
  def area(self):
    return self.length*self.width

class circle(shape):
  def __init__(self,radius):
    self.radius=radius
  def area(self):
    return 3.14*self.radius*self.radius

class square(shape):
  def __init__(self,side):
    self.side=side
  def area(self):
    return self.side*self.side
l=int(input("enter the length="))
b=int(input("enter the breadth="))
r=int(input("enter the radius="))
s=int(input("enter the side="))
print("area of rectangle",rectangle(l,b).area())
print("area of circle",circle(r).area())
print("area of square",square(s).area())

18.1# Addition, Subtraction and multiplication using multilevel and multiple inheritance.

"multilevel inheritance"
class add:
 def sum(self,a,b):
   print("The Sum is :",a+b)
class sub(add):
 def diff(self,a,b):
    print("The difference is:",a-b)
24
class prod(sub):
 def mult(self,a,b):
   print("The product is :",a*b)
a=add()
b=sub()
c=prod()
a.sum(20,64)
b.diff(100,3)
c.mult(394,9)


18.2# multilevel inheritance
class a:
 def get(self):
   self.x=int(input("Enter the x value:"))
   self.y=int(input("Enter the y value:"))
class b:
 def add(self):
   self.addition=self.x+self.y
 def sub(self):
   self.diff=self.x-self.y
 def mult(self):
   self.prod=self.x*self.y
class c(a,b):
 def output(self):
   print("The sum is:",self.addition)
   print("The difference is:",self.prod)
   print("The product is:",self.mult)
obj=c()
25
obj.get()
obj.add()
obj.sub()
obj.mult()
obj.output()


19.# To write a python program to find sum of the two numbers and product of the three numbers using method overloading.

def func(a=None,b=None,c=None):
 if a!=None and b!= None and c==None:
  print("The sum is :",a+b)
 elif a!=None and b!= None and c!=None:
  print("The product of 3 numbers is:",a*b*c)
 else:
  print("Nothing to find")
func(83,32)
func(20,30,40)



20.# To write a python programming to find area of Square and area of Triangle using method overloading

def area(a=None,b=None):
 if a!=None:
  print("The area of the square is:",a*a)
 elif a!=None and b!= None:
  print("The area of the triangle is:",0.5*a*b)
area(11)
area(45,7)
