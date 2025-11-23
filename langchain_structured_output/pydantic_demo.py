from pydantic import BaseModel
from typing import Optional

class Student(BaseModel):
    name : str
    age : Optional[int] = None

new_student1 = {'name':'ayush','age':11}
new_student2 = {'name':'ayush'}
student1 = Student(**new_student1)
student2 = Student(**new_student2)

print(student1)
print(student2)