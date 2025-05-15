# function = A block of reusable code
# place () after the function name to invoke it

# def happy_birthday(num):
#
#     for x in range(0, num):
#         print("Happy birthday to you!")
#         print("You are old!")
#         print("Happy birthday to you!")
#         print()
#
# happy_birthday(2)
#

# return statement used to end a function and send a result back to the caller

# def create_name(first, last):
#     first = first.capitalize()
#     last = last.capitalize()
#     return first + " " + last
#
# full_name = create_name("bro", "code")
# print(full_name)

# default arguments = A default value for certain parameters default is used when that argument is omitted
# A1 A make your functions more flexible, reduces # of arguments 1. positional, 2. DEFAULT, 3. keyword, 4. arbitrary

# def net_price(list_price, discount=0, tax=0.05):
#     return list_price * (1 - discount) * (1 + tax)
# print(net_price(500))
# print(net_price(500, 0.1))
# print(net_price(500, 0.1, 0))

# import time
# def count(end, start=0):
#     for x in range(start, end+1):
#         print(x)
#         time.sleep(1)
#     print("DONE!")
# count (30, 15)

#keyword arguments an argument preceded by an identifier helps with readability
# order of arguments doesn't matter
# 1. positional 2. default 3. KEYWORD 4. arbitrary

# def hello(greeting, title, first, last):
#     print(f"{greeting} {title}{first} {last}")
# hello("Hello", title="Mr.", last="John", first="James")
# print("Hello", "Mr.", "John", "James", sep="_")

# *args = allows you to pass multiple non-key arguments
# **kwargs allows you to pass multiple keyword-arguments * unpacking operator
# 1. positional 2. default 3. keyword 4. ARBITRARY

# def add(*args):
#     total = 0
#     for arg in args:
#         total += arg
#     return total

# print(add(1, 2, 3))

# def display_name(*args):
#     for arg in args:
#         print(arg, end=" ")
# display_name("Dr.", "Spongebob", "Harold", "Squarepants", "III")
#
#
# def print_address(**kwargs):
#     for key, value in kwargs.items():
#         print(f"{key}: {value}")
#
# print_address(street="123 Fake St.",
#               city="Detroit",
#               state="MI",
#               zip="54321")

# def shipping_label(*args, **kwargs):
#     for arg in args:
#         print(arg, end=" ")
#     print()
#     for value in kwargs.values():
#         print(value, end=" ")
#
# shipping_label("Dr.", "Spongebob", "Squarepants", "III",
#     street="123 Fake St.",
#     apt="100",
#     city="Detroit",
#     state="MT")


# Iterables = An object/collection that can return its elements one at a time,
# allowing it to be iterated over in a loop

# numbers = [1, 2, 3, 4, 5]
# for number in numbers:
#     print(number)

# fruits={"apple", "orange", "banana", "coconut"}
# for fruit in reversed(fruits):
#     print(fruit)


# my_dictionary={"A": 1, "B": 2, "C": 3}
# for key, value in my_dictionary.items():
#     print(key, value)


# Membership operators used to test whether a value or variable is found in a sequence
# (string, list, tuple, set, or dictionary) 1. in 2. not in

# word = "APPLE"
# letter = input("Guess a letter in the secret word: ")
# if letter in word:
#     print(f"There is a {letter}")
# else:
#     print(f"{letter} was not found")

# students = {"Spongebob", "Patrick", "Sandy"}
# student = input("Enter the name of a student: ")
# if student not in students:
#     print(f"{student} was not found")
# else:
#     print (f" {student} is a student")

# grades = {"Sandy": "A",
# "Squidward": "B",
# "Spongebob": "C",
# "Patrick": "D"}
#
# student = input("Enter the name of a student: ")
# if student in grades:
#     print(f" {student}'s grade is {grades[student]}")
# else:
#     print(f"{student} was not found")

email = "BroCode@gmail.com"
if "@" in email and "." in email:
    print("Valid email")
else:print("Invalid email")