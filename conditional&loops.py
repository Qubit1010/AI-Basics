# Conditional Statements
# x = int(input("Enter a number: "))
# if x > 0:
#     print("Positive Number")
# elif x == 0:
#     print("Zero")
# else:
#     print("Negative Number")

# operator = input("Enter an operator (+ - * /): ")
# num1 = float(input("Enter the 1st Number: "))
# num2 = float(input("Enter the 2nd Number: "))
#
# if operator == "+":
#     result = num1 + num2
#     print(result)
# elif operator == "-":
#     result = num1 - num2
#     print(result)
# elif operator == "*":
#     result = num1 * num2
#     print(result)
# elif operator == "/":
#     result = num1 / num2
#     print(result)
# else:
#     print("Not Valid Operator!")


# logical operators
# temp = 25
# is_raining = False
#
# if temp > 35 or temp < 0 or is_raining:
#     print("The outdoor event is cancelled")
# else:
#     print("The outdoor event is still scheduled")
#
# temp2 = 2
# is_sunny = False
#
# if temp2 > 35 and is_sunny:
#     print("Its hot outdoor")
#     print("Its sunny")
# elif temp2 < 5 and not is_sunny:
#     print("Its Cold  outdoor")
#     print("Its Cloudy")
# else:
#     print("Not Sunny")

# num = 7
# result = "EVEN" if num % 2 == 0 else "ODD"
# print(result)


# name = input("Enter your name: ")
# if name == "":
#     print("You did not enter your name")
# else:
#     print(f"Hello {name}")

# while loop= execute some code WHILE some condition remains true

# while name == "":
#     print("You did not enter your name")
#     name = input("Enter your name: ")
# print(f"Hello {name}")

# food = input("Enter a food you like (q to quit): ")
# while not food == "q":
#     print(f"You like {food}")
#     food =input("Enter another food you like (q to quit): ")
# print("bye")

# principle = 0
# rate = 0
# time = 0
#
# while True:
#     principle = float(input("Enter the principle amount: "))
#     if principle <= 0:
#         print("Principle can't be less than or equal to zero")
#     else:
#         break
#
# while True:
#     rate = float(input("Enter the Interest rate amount: "))
#     if rate <= 0:
#         print("Interest rate can't be less than or equal to zero")
#     else:
#         break
#
# while True:
#     time = float(input("Enter the Time in years: "))
#     if time <= 0:
#         print("Time can't be less than or equal to zero")
#     else:
#         break
#
# print(principle)
# print(rate)
# print(time)
#
# total = principle * pow((1 + rate / 100), time)
# print(f"Balance after {time} year/s: ${total:.2f}")


# for loops = execute a block of code a fixed number of times. You can iterate over a range, string, sequence, etc.
# for x in (range(1, 11)):
#     print(x)

# for x in reversed (range(1, 11)):
#     print(x)

# for x in (range(1, 11, 2)):
#     print(x)

# for x in (range(1, 11, )):
#     even = x % 2 == 0
#     if even:
#         print(x)

# credit_number = "1234-5678-9012-3456"
# for x in credit_number:
#     print(x)

# fruits = ["apple", "orange", "banana", "coconut"]
# for fruit in fruits:
#     print(fruit)

# for x in range(1,11):
#     if x == 3:
#         continue
#     print(x)

# for x in range(1,11):
#     if x == 4:
#         break
#     print(x)

# import time
# my_time = int(input("Enter the time in seconds:"))
#
# for x in range(my_time, 0, -1):
#     seconds = x % 60
#     minutes = int(x / 60) % 60
#     hours = int(x / 3600)
#     print(f"{hours:02}:{minutes:02}:{seconds:02}")
#     time.sleep(1)
#
# print("TIME'S UP!")


# nested loop = A loop within another loop (outer, inner)
#       outer loop:
#           inner loop:

# rows = int(input("Enter the # of rows: "))
# columns = int(input("Enter the # of columns: "))
# symbol = input("Enter a symbol to use: ")
#
# for x in range(rows):
#     for y in range(columns):
#         print(symbol, end="")
#     print()



# Match-case statement (switch): An alternative to using many 'elif' statements
#Execute some code if a value matches a 'case'
# Benefits: cleaner and syntax is more readable

# def day_of_week (day):
#     if day == 1:
#         return "It is Sunday"
#     elif day == 2:
#         return "It is Monday"
#     elif day == 3:
#         return "It is Tuesday"
#     elif day == 4:
#         return "It is Wednesday"
#     elif day == 5:
#         return "It is Thursday"
#     elif day == 6:
#      return "It is Friday"
#     elif day == 7:
#         return "It is Saturday"
#     else:
#         return "Not a valid day"

def day_of_week (day):
    match day:
      case 1:
        return "It is Sunday"
      case 2:
        return "It is Monday"
      case 3:
        return "It is Tuesday"
      case 4:
        return "It is Wednesday"
      case 5:
        return "It is Thursday"
      case 6:
        return "It is Friday"
      case 7:
        return "It is Saturday"
      case _:
        return "Not a valid day"


print(day_of_week(3))
