import multiprocessing
import time

# Function to modify the attribute of a custom object
def add_items(shared_list, lock, manager):
    for i in range(0, 5):
        with lock:
            new_obj = CustomObject(shared_list, 5, manager)
            shared_list.append(new_obj)
            time.sleep(1)

def change_value(shared_list, lock):
    for i in range(0, 2):
        time.sleep(2)
        with lock:
            for obj in shared_list:
                obj.value.value = 7 #.value

class CustomObject:
    def __init__(self, shared_list, value, manager):
        self.value = manager.Value('i', value)
        self.shared_list = shared_list

if __name__ == "__main__":
    # Create a multiprocessing Manager

    #multiprocessing.freeze_support()
    manager = multiprocessing.Manager()

    # Create a shared list of custom objects
    shared_list = manager.list()

    lock = multiprocessing.Lock()

    # Create a process to modify an attribute
    process1 = multiprocessing.Process(target=add_items, args=(shared_list, lock, manager))
    process2 = multiprocessing.Process(target=change_value, args=(shared_list, lock))
    process1.start()
    process2.start()
    process1.join()
    process2.join()

    # Print the modified attribute
    for obj in shared_list:
        print(obj.value.value) #.value