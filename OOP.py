"""
COMPREHENSIVE OBJECT-ORIENTED PROGRAMMING (OOP) IN PYTHON
==========================================================
This file covers all major OOP concepts with practical examples.
"""

# ============================================================================
# 1. CLASS AND OBJECT
# ============================================================================
print("=" * 60)
print("1. CLASS AND OBJECT")
print("=" * 60)


class Dog:
    """A simple class representing a dog"""

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def bark(self):
        return f"{self.name} says Woof!"


# Creating objects (instances)
dog1 = Dog("Buddy", 3)
dog2 = Dog("Max", 5)

print(f"Dog 1: {dog1.name}, Age: {dog1.age}")
print(dog1.bark())
print(f"Dog 2: {dog2.name}, Age: {dog2.age}")
print(dog2.bark())

# ============================================================================
# 2. ENCAPSULATION (Access Modifiers)
# ============================================================================
print("\n" + "=" * 60)
print("2. ENCAPSULATION")
print("=" * 60)


class BankAccount:
    """Demonstrates encapsulation with public, protected, and private attributes"""

    def __init__(self, account_number, balance):
        self.account_number = account_number  # Public
        self._bank_name = "MyBank"  # Protected (convention)
        self.__balance = balance  # Private (name mangling)

    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount
            return f"Deposited ${amount}. New balance: ${self.__balance}"
        return "Invalid amount"

    def get_balance(self):
        """Getter method for private balance"""
        return self.__balance

    def set_balance(self, balance):
        """Setter method for private balance"""
        if balance >= 0:
            self.__balance = balance
        else:
            raise ValueError("Balance cannot be negative")


account = BankAccount("ACC123", 1000)
print(f"Account: {account.account_number}")
print(f"Protected attribute: {account._bank_name}")
print(account.deposit(500))
print(f"Balance via getter: ${account.get_balance()}")
# print(account.__balance)  # This would raise AttributeError


# ============================================================================
# 3. INHERITANCE
# ============================================================================
print("\n" + "=" * 60)
print("3. INHERITANCE")
print("=" * 60)


# Single Inheritance
class Animal:
    """Base class"""

    def __init__(self, name, species):
        self.name = name
        self.species = species

    def make_sound(self):
        return "Some generic sound"

    def info(self):
        return f"{self.name} is a {self.species}"


class Cat(Animal):
    """Derived class inheriting from Animal"""

    def __init__(self, name, color):
        super().__init__(name, "Cat")
        self.color = color

    def make_sound(self):
        return "Meow!"


cat = Cat("Whiskers", "Orange")
print(cat.info())
print(cat.make_sound())
print(f"Color: {cat.color}")


# Multiple Inheritance
class Flyable:
    def fly(self):
        return "Flying in the sky"


class Swimmable:
    def swim(self):
        return "Swimming in water"


class Duck(Animal, Flyable, Swimmable):
    """Multiple inheritance example"""

    def __init__(self, name):
        super().__init__(name, "Duck")

    def make_sound(self):
        return "Quack!"


duck = Duck("Donald")
print(f"\n{duck.info()}")
print(duck.make_sound())
print(duck.fly())
print(duck.swim())


# Multilevel Inheritance
class Vehicle:
    def __init__(self, brand):
        self.brand = brand


class Car(Vehicle):
    def __init__(self, brand, model):
        super().__init__(brand)
        self.model = model


class ElectricCar(Car):
    def __init__(self, brand, model, battery_capacity):
        super().__init__(brand, model)
        self.battery_capacity = battery_capacity


tesla = ElectricCar("Tesla", "Model S", "100 kWh")
print(f"\nElectric Car: {tesla.brand} {tesla.model}, Battery: {tesla.battery_capacity}")

# ============================================================================
# 4. POLYMORPHISM
# ============================================================================
print("\n" + "=" * 60)
print("4. POLYMORPHISM")
print("=" * 60)


# Method Overriding
class Shape:
    def area(self):
        return 0


class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height


class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return 3.14159 * self.radius ** 2


# Polymorphism in action
shapes = [Rectangle(5, 10), Circle(7), Rectangle(3, 4)]

for shape in shapes:
    print(f"{shape.__class__.__name__} area: {shape.area()}")


# Operator Overloading
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        """Overloading the + operator"""
        return Vector(self.x + other.x, self.y + other.y)

    def __str__(self):
        """Overloading the str() function"""
        return f"Vector({self.x}, {self.y})"

    def __eq__(self, other):
        """Overloading the == operator"""
        return self.x == other.x and self.y == other.y


v1 = Vector(2, 3)
v2 = Vector(4, 5)
v3 = v1 + v2
print(f"\n{v1} + {v2} = {v3}")

# ============================================================================
# 5. ABSTRACTION
# ============================================================================
print("\n" + "=" * 60)
print("5. ABSTRACTION")
print("=" * 60)

from abc import ABC, abstractmethod


class PaymentMethod(ABC):
    """Abstract base class"""

    @abstractmethod
    def process_payment(self, amount):
        """Abstract method that must be implemented by subclasses"""
        pass

    @abstractmethod
    def refund(self, amount):
        """Another abstract method"""
        pass


class CreditCard(PaymentMethod):
    def __init__(self, card_number):
        self.card_number = card_number

    def process_payment(self, amount):
        return f"Processing ${amount} via Credit Card ending in {self.card_number[-4:]}"

    def refund(self, amount):
        return f"Refunding ${amount} to Credit Card"


class PayPal(PaymentMethod):
    def __init__(self, email):
        self.email = email

    def process_payment(self, amount):
        return f"Processing ${amount} via PayPal account {self.email}"

    def refund(self, amount):
        return f"Refunding ${amount} to PayPal account"


# Cannot instantiate abstract class
# payment = PaymentMethod()  # This would raise TypeError

cc = CreditCard("1234567890123456")
pp = PayPal("user@example.com")

print(cc.process_payment(100))
print(pp.process_payment(50))

# ============================================================================
# 6. PROPERTIES AND DECORATORS
# ============================================================================
print("\n" + "=" * 60)
print("6. PROPERTIES AND DECORATORS")
print("=" * 60)


class Temperature:
    """Using @property decorator for getters and setters"""

    def __init__(self, celsius=0):
        self._celsius = celsius

    @property
    def celsius(self):
        """Getter for celsius"""
        return self._celsius

    @celsius.setter
    def celsius(self, value):
        """Setter for celsius with validation"""
        if value < -273.15:
            raise ValueError("Temperature below absolute zero is not possible")
        self._celsius = value

    @property
    def fahrenheit(self):
        """Computed property"""
        return (self._celsius * 9 / 5) + 32

    @fahrenheit.setter
    def fahrenheit(self, value):
        self._celsius = (value - 32) * 5 / 9


temp = Temperature(25)
print(f"Temperature: {temp.celsius}°C = {temp.fahrenheit}°F")
temp.fahrenheit = 98.6
print(f"After setting to 98.6°F: {temp.celsius}°C")

# ============================================================================
# 7. CLASS METHODS AND STATIC METHODS
# ============================================================================
print("\n" + "=" * 60)
print("7. CLASS METHODS AND STATIC METHODS")
print("=" * 60)


class Employee:
    """Demonstrates class variables, class methods, and static methods"""

    company_name = "TechCorp"  # Class variable
    employee_count = 0

    def __init__(self, name, salary):
        self.name = name
        self.salary = salary
        Employee.employee_count += 1

    @classmethod
    def from_string(cls, emp_str):
        """Alternative constructor using class method"""
        name, salary = emp_str.split('-')
        return cls(name, int(salary))

    @classmethod
    def get_company_name(cls):
        """Class method to access class variable"""
        return cls.company_name

    @staticmethod
    def is_workday(day):
        """Static method - doesn't access instance or class"""
        return day not in ['Saturday', 'Sunday']

    def display(self):
        return f"{self.name}: ${self.salary}"


emp1 = Employee("Alice", 50000)
emp2 = Employee.from_string("Bob-60000")

print(emp1.display())
print(emp2.display())
print(f"Company: {Employee.get_company_name()}")
print(f"Total employees: {Employee.employee_count}")
print(f"Is Monday a workday? {Employee.is_workday('Monday')}")
print(f"Is Sunday a workday? {Employee.is_workday('Sunday')}")

# ============================================================================
# 8. MAGIC/DUNDER METHODS
# ============================================================================
print("\n" + "=" * 60)
print("8. MAGIC/DUNDER METHODS")
print("=" * 60)


class Book:
    """Demonstrates various magic methods"""

    def __init__(self, title, author, pages):
        self.title = title
        self.author = author
        self.pages = pages

    def __str__(self):
        """String representation for users"""
        return f"'{self.title}' by {self.author}"

    def __repr__(self):
        """String representation for developers"""
        return f"Book('{self.title}', '{self.author}', {self.pages})"

    def __len__(self):
        """Length of the book"""
        return self.pages

    def __eq__(self, other):
        """Equality comparison"""
        return self.title == other.title and self.author == other.author

    def __lt__(self, other):
        """Less than comparison"""
        return self.pages < other.pages

    def __add__(self, other):
        """Adding books combines their pages"""
        return Book(
            f"{self.title} & {other.title}",
            f"{self.author} & {other.author}",
            self.pages + other.pages
        )

    def __getitem__(self, key):
        """Makes the object subscriptable"""
        if key == 0:
            return self.title
        elif key == 1:
            return self.author
        elif key == 2:
            return self.pages
        raise IndexError("Index out of range")

    def __call__(self):
        """Makes the object callable"""
        return f"Reading '{self.title}'..."


book1 = Book("Python Programming", "John Doe", 500)
book2 = Book("Data Science", "Jane Smith", 450)

print(f"str: {str(book1)}")
print(f"repr: {repr(book1)}")
print(f"Length: {len(book1)} pages")
print(f"book1 < book2: {book1 < book2}")
print(f"Subscript [0]: {book1[0]}")
print(f"Callable: {book1()}")

combined = book1 + book2
print(f"Combined book: {combined}")

# ============================================================================
# 9. COMPOSITION
# ============================================================================
print("\n" + "=" * 60)
print("9. COMPOSITION (HAS-A Relationship)")
print("=" * 60)


class Engine:
    def __init__(self, horsepower):
        self.horsepower = horsepower

    def start(self):
        return "Engine started"


class Wheel:
    def __init__(self, size):
        self.size = size


class AutoCar:
    """Car has-a Engine and has-a set of Wheels"""

    def __init__(self, brand, horsepower, wheel_size):
        self.brand = brand
        self.engine = Engine(horsepower)  # Composition
        self.wheels = [Wheel(wheel_size) for _ in range(4)]  # Composition

    def drive(self):
        return f"{self.brand} car with {self.engine.horsepower}HP engine is driving on {self.wheels[0].size}-inch wheels"


car = AutoCar("Toyota", 200, 18)
print(car.engine.start())
print(car.drive())

# ============================================================================
# 10. AGGREGATION
# ============================================================================
print("\n" + "=" * 60)
print("10. AGGREGATION (Weak HAS-A Relationship)")
print("=" * 60)


class Student:
    def __init__(self, name):
        self.name = name


class Department:
    """Department has students but they can exist independently"""

    def __init__(self, name):
        self.name = name
        self.students = []

    def add_student(self, student):
        self.students.append(student)

    def show_students(self):
        return [student.name for student in self.students]


student1 = Student("Alice")
student2 = Student("Bob")

dept = Department("Computer Science")
dept.add_student(student1)
dept.add_student(student2)

print(f"{dept.name} department students: {dept.show_students()}")

# ============================================================================
# 11. METHOD RESOLUTION ORDER (MRO)
# ============================================================================
print("\n" + "=" * 60)
print("11. METHOD RESOLUTION ORDER (MRO)")
print("=" * 60)


class A:
    def method(self):
        return "Method from A"


class B(A):
    def method(self):
        return "Method from B"


class C(A):
    def method(self):
        return "Method from C"


class D(B, C):
    pass


d = D()
print(f"D's method: {d.method()}")
print(f"MRO for class D: {[cls.__name__ for cls in D.__mro__]}")

# ============================================================================
# 12. DATACLASSES (Python 3.7+)
# ============================================================================
print("\n" + "=" * 60)
print("12. DATACLASSES")
print("=" * 60)

from dataclasses import dataclass, field


@dataclass
class Product:
    """Dataclass automatically generates __init__, __repr__, __eq__, etc."""
    name: str
    price: float
    quantity: int = 0
    tags: list = field(default_factory=list)

    def total_value(self):
        return self.price * self.quantity


product = Product("Laptop", 999.99, 5, ["electronics", "computers"])
print(product)
print(f"Total value: ${product.total_value()}")

# ============================================================================
# 13. SLOTS (Memory Optimization)
# ============================================================================
print("\n" + "=" * 60)
print("13. SLOTS (Memory Optimization)")
print("=" * 60)


class Point:
    """Using __slots__ to reduce memory usage"""
    __slots__ = ['x', 'y']

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Point({self.x}, {self.y})"


point = Point(10, 20)
print(point)
# point.z = 30  # This would raise AttributeError


# ============================================================================
# 14. DESCRIPTORS
# ============================================================================
print("\n" + "=" * 60)
print("14. DESCRIPTORS")
print("=" * 60)


class Validator:
    """A descriptor for validating values"""

    def __init__(self, min_value=None, max_value=None):
        self.min_value = min_value
        self.max_value = max_value

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj.__dict__.get(self.name)

    def __set__(self, obj, value):
        if self.min_value is not None and value < self.min_value:
            raise ValueError(f"{self.name} must be >= {self.min_value}")
        if self.max_value is not None and value > self.max_value:
            raise ValueError(f"{self.name} must be <= {self.max_value}")
        obj.__dict__[self.name] = value


class Person:
    age = Validator(min_value=0, max_value=150)

    def __init__(self, name, age):
        self.name = name
        self.age = age


person = Person("John", 30)
print(f"Person: {person.name}, Age: {person.age}")

try:
    person.age = 200
except ValueError as e:
    print(f"Error: {e}")

# ============================================================================
# 15. CONTEXT MANAGERS
# ============================================================================
print("\n" + "=" * 60)
print("15. CONTEXT MANAGERS")
print("=" * 60)


class FileManager:
    """Custom context manager"""

    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None

    def __enter__(self):
        self.file = open(self.filename, self.mode)
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
        return False


# Using the context manager
with FileManager('/home/claude/test.txt', 'w') as f:
    f.write('Hello from custom context manager!')

print("File written using custom context manager")

# Alternative using contextlib
from contextlib import contextmanager


@contextmanager
def timer():
    import time
    start = time.time()
    yield
    end = time.time()
    print(f"Execution time: {end - start:.4f} seconds")


with timer():
    sum([i ** 2 for i in range(1000000)])

# ============================================================================
# 16. MIXINS
# ============================================================================
print("\n" + "=" * 60)
print("16. MIXINS")
print("=" * 60)


class JSONMixin:
    """Mixin to add JSON serialization capability"""

    def to_json(self):
        import json
        return json.dumps(self.__dict__)


class LogMixin:
    """Mixin to add logging capability"""

    def log(self, message):
        print(f"[LOG] {self.__class__.__name__}: {message}")


class User(JSONMixin, LogMixin):
    def __init__(self, username, email):
        self.username = username
        self.email = email


user = User("johndoe", "john@example.com")
print(user.to_json())
user.log("User object created")

# ============================================================================
# 17. SINGLETON PATTERN
# ============================================================================
print("\n" + "=" * 60)
print("17. SINGLETON PATTERN")
print("=" * 60)


class Singleton:
    """Singleton pattern - only one instance can exist"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.value = None


s1 = Singleton()
s2 = Singleton()
s1.value = 100

print(f"s1 is s2: {s1 is s2}")
print(f"s2.value: {s2.value}")

# ============================================================================
# 18. METACLASSES
# ============================================================================
print("\n" + "=" * 60)
print("18. METACLASSES")
print("=" * 60)


class UpperAttrMetaclass(type):
    """Metaclass that converts all attribute names to uppercase"""

    def __new__(cls, name, bases, attrs):
        uppercase_attrs = {
            key.upper() if not key.startswith('__') else key: value
            for key, value in attrs.items()
        }
        return super().__new__(cls, name, bases, uppercase_attrs)


class MyClass(metaclass=UpperAttrMetaclass):
    hello = "world"

    def greet(self):
        return "Hello!"


obj = MyClass()
print(f"Has HELLO: {hasattr(obj, 'HELLO')}")
print(f"HELLO value: {obj.HELLO}")
print(f"GREET method: {obj.GREET()}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("OOP CONCEPTS COVERED:")
print("=" * 60)
concepts = [
    "1. Class and Object",
    "2. Encapsulation (Public, Protected, Private)",
    "3. Inheritance (Single, Multiple, Multilevel)",
    "4. Polymorphism (Method Overriding, Operator Overloading)",
    "5. Abstraction (Abstract Classes)",
    "6. Properties and Decorators",
    "7. Class Methods and Static Methods",
    "8. Magic/Dunder Methods",
    "9. Composition",
    "10. Aggregation",
    "11. Method Resolution Order (MRO)",
    "12. Dataclasses",
    "13. Slots",
    "14. Descriptors",
    "15. Context Managers",
    "16. Mixins",
    "17. Singleton Pattern",
    "18. Metaclasses"
]

for concept in concepts:
    print(f"✓ {concept}")

print("\nAll OOP concepts demonstrated successfully!")