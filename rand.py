
import random

# low = 1
# high = 100
# options = ("rock", "paper", "scissors")
# cards = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A",]
# number = random.randint(low, high) # number = random.random()
# option = random.choice(options)
# random.shuffle(cards)

# print(cards)


# Python number guessing game
# lowest_num = 1
# highest_num = 100
# answer = random.randint(lowest_num, highest_num)
# guesses = 0
# is_running = True
#
# print("Python Number Guessing Game")
# print(f"Select a number between {lowest_num} and {highest_num}")
#
# while is_running:
#
#     guess = input("Enter your guess: ")
#     if guess.isdigit():
#         guess = int(guess)
#         guesses += 1
#
#         if guess < lowest_num or guess > highest_num:
#             print("That number is out of range")
#             print(f"Please select a number between {lowest_num} and {highest_num}")
#         elif guess < answer:
#             print("Too low! Try again!")
#         elif guess > answer:
#             print("Too high! Try again!")
#         else:
#             print(f"Correct! The answer was {answer}")
#             print(f"Number of guesses {guesses}")
#             is_running = False
#     else:
#         print("Invalid guess")
#         print(f"Please select a number between {lowest_num} and {highest_num}")

# 
def get_computer_choice():
    choices = ["rock", "paper", "scissors"]
    return random.choice(choices)

def get_user_choice():
    user_choice = input("Enter rock, paper, or scissors: ").lower()
    while user_choice not in ["rock", "paper", "scissors"]:
        print("Invalid choice! Please try again.")
        user_choice = input("Enter rock, paper, or scissors: ").lower()
    return user_choice

def determine_winner(user, computer):
    if user == computer:
        return "It's a tie!"
    elif (user == "rock" and computer == "scissors") or \
            (user == "scissors" and computer == "paper") or \
            (user == "paper" and computer == "rock"):
        return "You win!"
    else:
        return "Computer wins!"

def play_game():
    print("Welcome to Rock, Paper, Scissors!")
    while True:
        user_choice = get_user_choice()
        computer_choice = get_computer_choice()
        print(f"Computer chose: {computer_choice}")
        print(determine_winner(user_choice, computer_choice))

        play_again = input("Do you want to play again? (yes/no): ").lower()
        if play_again != "yes":
            print("Thanks for playing! Goodbye!")
            break

play_game()