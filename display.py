# Global variable to store the class name
class_name = None

# Function to set the class name
def set_class_name(name):
    global class_name
    class_name = name

# Function to print the class name
def print_class_name():
    global class_name
    if class_name is not None:
        return class_name
    else:
        return "no class detected"


