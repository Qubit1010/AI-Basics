
# collection = single "variable" used to store multiple values
# List = [] ordered and changeable. Duplicates OK
# Set = {} unordered and immutable, but Add/Remove OK. NO duplicates
# Tuple = () ordered and unchangeable. Duplicates OK. FASTER I


# fruits = ["apple", "orange", "banana", "coconut"]
# fruits2 = {"apple", "orange", "banana", "coconut"}
# fruits3 = ("apple", "orange", "banana", "coconut")

# print (fruits)
# print (dir(fruits))
# print (help(fruits))
# print("pineapple" in fruits)

# print (len(fruits))
# print (fruits [0:3])
# print (fruits [::2])
# print (fruits [::-1])

# fruits [0] = "pineapple"
# fruits.append("mango")
# fruits.remove("banana")
# fruits.insert(2,"apricot")
# fruits.sort()
# fruits.reverse()
# fruits.clear()
# print(fruits.index("coconut"))
# print(fruits)
# print(fruits2)
# fruits2.add("grapes")
# fruits2.remove("coconut")
#
# print(fruits2)
# print(fruits3)
#
# for fruit in fruits:
#     print(fruit)



# Shopping cart program
# foods = []
# prices = []
# total = 0
# while True:
#     food = input("Enter a food to buy (q to quit): ")
#     if food.lower() == "q":
#         break
#     else:
#         price = float(input(f"Enter the price of a {food}: $"))
#         foods.append(food)
#         prices.append(price)
#
# print("---- Your Cart ----")
#
# for food in foods:
#     print(food, end=" ")
#
# for price in prices:
#     print(price, end=" ")
#     total += price
#
# print(f"\nYour total is: ${total}")



# fruits = ["apple", "orange", "banana", "coconut"]
# vegetables = ["celery", "carrots", "potatoes"]
# meats = ["chicken", "fish", "turkey"]
# groceries = [fruits, vegetables, meats]
# groceries2 = [["apple", "orange", "banana", "coconut"],
#               ["celery", "carrots", "potatoes"],
#               ["chicken", "fish", "turkey"]]


# print(groceries[0])
# print(groceries[1])
# print(groceries[2])
# print(groceries[2][1])
# print(groceries2)
#
# for collection in groceries2:
#     for food in collection:
#         print(food, end=" ")
#


# num_pad =((1, 2, 3),
#           (4, 5, 6),
#           (7, 8, 9),
#           ("*", 0, "#"))
#
# for row in num_pad:
#     for num in row:
#         print(num, end=" ")
#     print()



# Python quiz game
# questions = ("How many elements are in the periodic table?:", "Which animal lays the largest eggs?: ",
# "What is the most abundant gas in Earth's atmosphere?:", "How many bones are in the human body?: ",
# "Which planet in the solar system is the hottest?: " )
# options= (("A. 116", "B. 117", "C. 118", "D. 119"),
# ("A. Whale", "B. Crocodile", "C. Elephant", "D. Ostrich"),
# ("A. Nitrogen", "B. Oxygen", "C. Carbon-Dioxide", "D. Hydrogen"),
# ("A. 206", "B. 207", "C. 208", "D. 209"),
# ("A. Mercury", "B. Venus", "C. Earth", "D. Mars"))
#
#
# answers = ("C", "D", "A", "A", "B")
# guesses = []
# score = 0
#
# question_num = 0
# for question in questions:
#     print("---------------------")
#     print(question)
#
#     for option in options[question_num]:
#         print(option)
#
#     guess = input("Enter (A, B, C, D): ").upper()
#     guesses.append(guess)
#     if guess == answers [question_num]:
#         score += 1
#         print("CORRECT!")
#     else:
#         print("INCORRECT!")
#         print (f" {answers [question_num]} is the correct answer")
#     question_num += 1
#
#
# for answer in answers:
#     print(answer, end=" ")
# print()
#
# print("guesses: ", end="")
# for guess in guesses:
#     print(guess, end=" ")
# print()
# score = int(score / len(questions) * 100)
# print (f"Your score is: {score}%")



# dictionary  = a collection of {key:value} pairs ordered and changeable. No duplicates
capitals = {"USA": "Washington D.C.",
"India": "New Delhi",
"China": "Beijing",
"Russia": "Moscow"}
# print(dir(capitals))
# print(help (capitals))
# print(capitals.get("USA"))
# print(capitals.get("Japan"))



# if capitals.get("Russia"):
#   print("That capital exists")
#else:
#   print("That capital doesn't exist")

# capitals.update({"Germany": "Berlin"})
# capitals.update({"USA": "Detroit"})
# capitals.pop("China")
# capitals.popitem()
# capitals.clear()

# keys = capitals.keys()
# for key in capitals.keys():
#     print(key)
#
# values = capitals.values()
# for value in capitals.values():
#     print(value)


#items = capitals.items()
# for key, value in capitals.items():
#     print(f" {key}: {value}")

# Concession stand program
menu = {"pizza": 3.00,
    "nachos": 4.50,
    "popcorn": 6.00,
    "fries": 2.50,
    "chips": 1.00,
    "pretzel": 3.50,
    "soda": 3.00,
    "lemonade": 4.25}

cart =[]
total = 0

print("------- MENU -------")
for key, value in menu.items():
    print(f"{key:10}: ${value:.2f}")
print("------------------")

while True:
    food =input("Select an item (q to quit): ").lower()
    if food == "q":
        break
    elif menu.get(food) is not None:
        cart.append(food)


print("----YOUR ORDER----")
for food in cart:
    total += menu.get(food)
    print (food, end=" ")

print()
print (f"Total is: ${total:.2f}")

print(cart)