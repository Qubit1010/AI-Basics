# Python Banking Program

# def show_balance(balance):
#     print(f"Your balance is ${balance:.2f}")
#
# def deposit():
#     amount = float(input("Enter an amount to be deposited: "))
#     if amount < 0:
#         print("That's not a valid amount")
#         return 0
#     else:
#         return amount
#
#
# def withdraw(balance):
#     amount = float(input("Enter amount to be withdrawn: "))
#     if amount > balance:
#         print("Insufficient funds")
#         return 0
#     elif amount < 0:
#         print("Amount must be greater than 0")
#         return 0
#     else:
#         return amount
#
# def main():
#     balance = 0
#     is_running = True
#
#     while is_running:
#         print("Banking Program")
#         print("1.Show Balance")
#         print("2.Deposit")
#         print("3.Withdraw")
#         print("4.Exit")
#
#         choice = input("Enter your choice (1-4): ")
#
#         if choice == '1':
#             show_balance(balance)
#         elif choice == '2':
#             balance += deposit()
#         elif choice == '3':
#             balance -= withdraw(balance)
#         elif choice == '4':
#             is_running = False
#         else:
#             print("That is not a valid choice")
#     print("Thank you! have a nice day!")
#
# if __name__ == '__main__':
#     main()


# import random
#
# # Slot symbols
# symbols = ["🍒", "🍋", "🔔", "💎", "7️⃣"]
#
#
# def spin_slot_machine():
#     return [random.choice(symbols) for _ in range(3)]
#
#
# def check_win(result):
#     # All 3 symbols match
#     if result[0] == result[1] == result[2]:
#         return "🎉 JACKPOT! You win!"
#     # Any 2 symbols match
#     elif result[0] == result[1] or result[1] == result[2] or result[0] == result[2]:
#         return "😊 Nice! You matched two!"
#     else:
#         return "😢 Sorry, try again."
#
#
# def slot_machine_game():
#     print("🎰 Welcome to the Slot Machine!")
#     name = input("What's your name? ")
#     balance = 100
#
#     while True:
#         print(f"\n{name}, your current balance is: ${balance}")
#         bet = input("Place your bet (or type 'q' to quit): ")
#
#         if bet.lower() == 'q':
#             print("Thanks for playing! Goodbye 👋")
#             break
#
#         if not bet.isdigit() or int(bet) <= 0:
#             print("Please enter a valid positive number.")
#             continue
#
#         bet = int(bet)
#
#         if bet > balance:
#             print("You don't have enough balance!")
#             continue
#
#         # Spin the slot machine
#         result = spin_slot_machine()
#         print("Spinning...")
#         print(" | ".join(result))
#
#         message = check_win(result)
#         print(message)
#
#         # Adjust balance
#         if "JACKPOT" in message:
#             balance += bet * 5
#         elif "matched two" in message:
#             balance += bet * 2
#         else:
#             balance -= bet
#
#         if balance <= 0:
#             print("😢 You're out of money! Game over.")
#             break


# Run the game
# slot_machine_game()


def encrypt(message, key):
    encrypted = ""

    for char in message:
        if char.isalpha():
            # Preserve case
            base = ord('A') if char.isupper() else ord('a')
            shifted = (ord(char) - base + key) % 26 + base
            encrypted += chr(shifted)
        else:
            # Leave non-alphabet characters unchanged
            encrypted += char
    return encrypted

def decrypt(message, key):
    return encrypt(message, -key)  # Reverse the shift for decryption

# Driver code
print("🔐 Caesar Cipher Encryption Program 🔐")
msg = input("Enter the message: ")
while True:
    try:
        key = int(input("Enter the key (number): "))
        break
    except ValueError:
        print("Please enter a valid number.")

encrypted_msg = encrypt(msg, key)
print("✅ Encrypted Message:", encrypted_msg)

# Optional: Decryption
choice = input("Do you want to decrypt the message? (y/n): ")
if choice.lower() == 'y':
    decrypted_msg = decrypt(encrypted_msg, key)
    print("🔓 Decrypted Message:", decrypted_msg)
