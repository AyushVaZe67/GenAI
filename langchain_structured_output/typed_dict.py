from typing import TypedDict

class Person(TypedDict):
    name:str
    age:int

new_dict : Person = {'name':'ayush','age':11}
print(new_dict)
