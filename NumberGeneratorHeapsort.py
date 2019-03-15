# Number Generator - 2
# July 18, 2018
# Shakeel Alibhai

import argparse
import random
import sys
import time

DEFAULT_OUTPUT_FILE = "output.txt"
DEFAULT_NUMS = 10000000

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--decreasing", help="Makes all the numbers be in decreasing order\nNote: Use of this argument alongside the increasing argument or the max_time argument is not supported.\"", action="store_true")
parser.add_argument("-i", "--increasing", help="Makes all the numbers be in increasing order\nNote: Use of this argument alongisde the decreasing argument or the max_time argument is not supported.\"", action="store_true")
parser.add_argument("-k", "--keep", help="Attempts to prevent a file from being overwritten. If a file already exists with the name of the output file, that file will (hopefully) not be replaced and the program will exit. (Note: Not guaranteed.)", action="store_true")
parser.add_argument(
    "-m", "--max_time", help="Allows the user to specify the maximum time the program will run. The program will hopefully stop running if it exceeds the user's maximum time input (prompted during program execution). This is not guaranteed, however. The time limit will only be checked when going through the numbers written to the file.", action="store_true")
parser.add_argument("-n", "--numbers", help="Allows the user to specify the number of numbers they want to be generated", action="store_true")
parser.add_argument("-o", "--output", help="Allows the user to specify the output file", action="store_true")
parser.add_argument("-p", "--decimal", help="Allows the user to specify the maximum number of digits to have after the decimal point.", action="store_true")
parser.add_argument("-r", "--range", help="Allows the user to specify the minimum and maximum values of the generated numbers", action="store_true")
parser.add_argument("-s", "--seed", help="Allows the user to specify a seed", action="store_true")
parser.add_argument("-t", "--timer", help="Displays the time it took to execute the program", action="store_true")
parser.add_argument("-v", "--verbose", help="Prints messages at various points in the program execution", action="store_true")
args = parser.parse_args()

# Heapsort code from https://www.geeksforgeeks.org/heap-sort/
 
# To heapify subtree rooted at index i.
# n is size of heap
def heapify(arr, n, i):
    largest = i  # Initialize largest as root
    l = 2 * i + 1     # left = 2*i + 1
    r = 2 * i + 2     # right = 2*i + 2
 
    # See if left child of root exists and is
    # greater than root
    if l < n and arr[i] < arr[l]:
        largest = l
 
    # See if right child of root exists and is
    # greater than root
    if r < n and arr[largest] < arr[r]:
        largest = r
 
    # Change root, if needed
    if largest != i:
        arr[i],arr[largest] = arr[largest],arr[i]  # swap
 
        # Heapify the root.
        heapify(arr, n, largest)
 
# The main function to sort an array of given size
def heapSort(arr):
    n = len(arr)
 
    # Build a maxheap.
    for i in range(n, -1, -1):
        heapify(arr, n, i)
 
    # One by one extract elements
    for i in range(n-1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]   # swap
        heapify(arr, i, 0)
 
# Driver code to test above
'''arr = [ 12, 11, 13, 5, 6, 7]
heapSort(arr)
n = len(arr)
print ("Sorted array is")
for i in range(n):
    print ("%d" %arr[i])'''
# This code is contributed by Mohit Kumra





# Check to make sure that arguments that are not supported together are not used
if args.decreasing and args.increasing and args.max_time:
    print("%s" % "Error: \"Decreasing,\" \"increasing,\" and \"max_time\" arguments are not supported together.")
    sys.exit()
if args.decreasing and args.increasing:
    print("%s" % "Error: \"Decreasing\" and \"increasing\" arguments are not supported together.")
    sys.exit()
if args.decreasing and args.max_time:
    print("%s" % "Error: \"Decreasing\" and \"max_time\" arguments are not supported together.")
    sys.exit()
if args.increasing and args.max_time:
    print("%s" % "Error: \"Increasing\" and \"max_time\" arguments are not supported together.")
    sys.exit()

if args.max_time or args.timer:
    start_time = time.time()
    if args.verbose:
        print("Timer started.")

decimal = False
# If the user chooses to specify the number of digits to have after the decimal point
if args.decimal:
    try:
        input_str = input("Please enter the maximum number of digits to have after the decimal point: ")
    except KeyboardInterrupt:
        print()
        sys.exit()
    try:
        digits_decimal = int(input_str)
        decimal = True
    except ValueError:
        print("%s" % "Error: Integer not recognized; will use the default.")
        decimal = False

# If the user chooses to specify a maximum time
if args.max_time:
    try:
        input_str = input("Please enter the maximum amount of time (in seconds) you want this program to run (not guaranteed): ")
    except KeyboardInterrupt:
        print()
        sys.exit()
    try:
        max_time = float(input_str)
    except ValueError:
        print("Invalid input; exiting program.")
        sys.exit()

nums = DEFAULT_NUMS
# If the user chooses to specify the number of numbers to be generated
if args.numbers:
    try:
        input_str = input("Please enter the number of numbers you would like to be generated: ")
    except KeyboardInterrupt:
        print()
        sys.exit()
    # Attempt to convert the user's input to an int
    # If the conversion fails, print a message and use DEFAULT_NUMS as the value
    try:
        nums = int(input_str)
    except ValueError:
        print("Error: Integer not recognized; the default number of %d will be used." % DEFAULT_NUMS)
        nums = DEFAULT_NUMS
    # If the number of numbers to be generated is less than 1, print a message and exit the program
    if nums < 1:
        print("%s" % "Error: Number of numbers to be generated is less than 1; exiting program.")
        sys.exit()

file_path = DEFAULT_OUTPUT_FILE
# If the user chooses to specify the output file
if args.output:
    try:
        file_path = input("Please enter the file to output the numbers to: ")
    except KeyboardInterrupt:
        print()
        sys.exit()
    # If no file path was entered, print a message and use DEFAULT_OUTPUT_FILE as the file path
    if(file_path == ""):
        print("Error: No file path entered; the default file path of %s will be used." % DEFAULT_OUTPUT_FILE)
        file_path = DEFAULT_OUTPUT_FILE

# Checks whether a file with the same path and name as file_name exists
# Paramter: file_name, the path/name of a file to check existence of
# Returns True if such a file exists and False otherwise
def check_file_exists(file_name):
    if args.verbose:
        print("%s" % "Checking whether a file with the same path and name as the output file exists.")
    try:
        temp_file = open(file_path, "x")
        temp_file.close()
        return False
    except FileExistsError:
        return True

# If the user chooses not to overwrite an existing file, check whether a file with the same path/name as the output file exists. If so, print a message and exit the program.
if args.keep:
    file_exists = check_file_exists(file_path)
    if(file_exists):
        print("%s" % "File with the name of the output file already exists; exiting program.")
        sys.exit()

custom_range = False
# If the user chooses to specify the minimum and maximum values of the generated numbers
if args.range:
    try:
        min_str = input("Please enter the minimum value of the generated numbers: ")
        max_str = input("Please enter the maximum value of the generated numbers: ")
    except KeyboardInterrupt:
        print()
        sys.exit()
    # Attempt to convert min_str and max_str to floats
    # If the conversion fails, print a message and use the default (between 0 and 1)
    try:
        min = float(min_str)
        max = float(max_str)
        custom_range = True
    except ValueError:
        print("%s" % "Error: Number not recognized; the generated numbers will be between 0 and 1.")
        custom_range = False

# If the user chooses to specify a seed
if args.seed:
    try:
        seed = input("Please enter a seed: ")
    except KeyboardInterrupt:
        print()
        sys.exit()
    random.seed(a=seed)

# Attempt to open the file
# If an OSError is raised, print a message and exit the program
try:
    output_file = open(file_path, "w+")
except OSError:
    print("%s" % "Error opening file; exiting program.")
    sys.exit()

if args.verbose:
    print("%s" % "File opened.")

if args.decreasing:
    if args.verbose:
        print("%s" % "Generating numbers...")
    num_list = []
    for i in range(1, nums + 1):
        # If the numbers to generate are in a custom range
        if custom_range:
            # If a specified number of digits after the decimal point is requested
            if decimal:
                num_list.append(round(random.uniform(min, max), 2))
            # If no specified number of digits after the decimal point is requested
            else:
                num_list.append(random.uniform(min, max))
        # If the numbers to generate are not in a custom range
        else:
            # If a specified number of digits after the decimal point is requested
            if decimal:
                num_list.append(round(random.random(), 2))
            # If no specified number of digits after the decimal point is requested
            else:
                num_list.append(random.random())
    if args.verbose:
        print("%s" % "Sorting numbers...")
    heapSort(nums)
    if args.verbose:
        print("%s" % "Printing numbers...")
    # Print the numbers to the file
    for i in range(nums):
        output_file.write(str(num_list[i]))
        # If this is not the last number, write a newline character to the file
        if i != nums - 1:
            output_file.write("\n")
elif args.increasing:
    if args.verbose:
        print("%s" % "Generating numbers...")
    num_list = []
    for i in range(1, nums + 1):
        # If the numbers to generate are in a custom range
        if custom_range:
            # If a specified number of digits after the decimal point is requested
            if decimal:
                num_list.append(round(random.uniform(min, max), 2))
            # If no specified number of digits after the decimal point is requested
            else:
                num_list.append(random.uniform(min, max))
        # If the numbers to generate are not in a custom range
        else:
            # If a specified number of digits after the decimal point is requested
            if decimal:
                num_list.append(round(random.random(), 2))
            # If no specified number of digits after the decimal point is requested
            else:
                num_list.append(random.random())
    if args.verbose:
        print("%s" % "Sorting numbers...")
    heapSort(num_list)
    if args.verbose:
        print("%s" % "Printing numbers...")
    # Print the numbers to the file
    for i in range(nums):
        output_file.write(str(num_list[i]))
        # If this is not the last number, write a newline character to the file
        if i != nums - 1:
            output_file.write("\n")
else:
    for i in range(1, nums + 1):
        # If the numbers to generate are in a custom range
        if custom_range:
            # If a specified number of digits after the decimal point is requested
            if decimal:
                output_file.write(str(round(random.uniform(min, max), 2)))
            # If no specified number of digits after the decimal point is requested
            else:
                output_file.write(str(random.uniform(min, max)))
        # If the numbers to generate are not in a custom range
        else:
            # If a specified number of digits after the decimal point is requested
            if decimal:
                output_file.write(str(round(random.random(), 2)))
            # If no specified number of digits after the decimal point is requested
            else:
                output_file.write(str(random.random()))
        # If args.verbose is true, print i every 1,000,000 values
        if (i % 1000000 == 0) and args.verbose:
            print("At number #%d" % i)
        # If this is not the last number, write a newline character to the file
        if i != nums:
            output_file.write("\n")
        if args.max_time:
            if((time.time() - start_time) + 0.15) > max_time:
                output_file.close()
                print("Approaching time limit (%f of %f seconds); file closed; last number printed was number %d of %d numbers" % ((time.time() - start_time), max_time, i, nums))
                sys.exit()

output_file.close()

if args.verbose:
    print("%s" % "File closed.")

if args.timer:
    print("Program Time: %f seconds (accuracy not guaranteed)" % (time.time() - start_time))
    print("%s" % "Note: This time includes the time it took to collect user input(s) after the start of program execution.")
