import multiprocessing
import time

# Function to modify the attribute of a custom object
def add_items(shared_dict, lock):
    for i in range(0, 5):
        with lock:
            shared_dict[i] = '-'
            time.sleep(1)

def change_value(shared_dict, lock):
    for i in range(0, 1):
        time.sleep(2)
        with lock:
            if len(shared_dict.keys()) > 0:
                for key in shared_dict.keys():
                    shared_dict[key] = 7


if __name__ == "__main__":

    #multiprocessing.freeze_support()
    manager = multiprocessing.Manager()
    shared_dict = manager.dict()
    lock = multiprocessing.Lock()

    # Create a process to modify an attribute
    process1 = multiprocessing.Process(target=add_items, args=(shared_dict, lock))
    process2 = multiprocessing.Process(target=change_value, args=(shared_dict, lock))
    process1.start()
    process2.start()
    process1.join()
    process2.join()

    # Print the modified attribute
    for key in shared_dict:
        print('shared_dict[key]', shared_dict[key])