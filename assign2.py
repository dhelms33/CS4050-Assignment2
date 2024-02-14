"""
Dereck Helms
CS 4050
"""

import time
import random
import sys

sys.setrecursionlimit(100000)


def bubbleSort(alist):
    """Perform Bubble Sort on a list."""
    start_time = time.time()

    n = len(alist)
    for i in range(n):
        for j in range(0, n - i - 1):
            if alist[j] > alist[j + 1]:
                # Swap elements if they are in the wrong order
                alist[j], alist[j + 1] = alist[j + 1], alist[j]

    elapsed_time = time.time() - start_time
    return alist, elapsed_time


def insertionSort(alist):
    """Perform Insertion Sort on a list."""
    start_time = time.time()

    for i in range(1, len(alist)):
        key = alist[i]
        j = i - 1
        while j >= 0 and key < alist[j]:
            # Shift elements greater than key to the right
            alist[j + 1] = alist[j]
            j -= 1
        # Insert the key at its correct position
        alist[j + 1] = key

    elapsed_time = time.time() - start_time
    return alist, elapsed_time


def mergeSort(alist):
    """Perform Merge Sort on a list."""
    start_time = time.time()

    # Helper function for merging two sorted sublists
    def merge(left, right):
        result = []
        i = j = 0

        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1

        result.extend(left[i:])
        result.extend(right[j:])
        return result

    if len(alist) <= 1:
        return alist, time.time() - start_time

    mid = len(alist) // 2
    left, left_time = mergeSort(alist[:mid])
    right, right_time = mergeSort(alist[mid:])

    # Merge the sorted sublists
    result = merge(left, right)

    elapsed_time = time.time() - start_time
    return result, elapsed_time + left_time + right_time


def hybridSort(alist):
    """Perform Hybrid Sort (combination of Insertion Sort and Merge Sort) on a list."""
    start_time = time.time()

    # Threshold for switching between Insertion Sort and Merge Sort
    threshold = 100

    # Helper function for merging two sorted sublists
    def merge(left, right):
        result = []
        i = j = 0

        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1

        result.extend(left[i:])
        result.extend(right[j:])
        return result

    # Helper function for insertion sort
    def insertion_sort(arr):
        for i in range(1, len(arr)):
            key = arr[i]
            j = i - 1
            while j >= 0 and key < arr[j]:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key

    if len(alist) <= threshold:
        insertion_sort(alist)
        return alist, time.time() - start_time
    else:
        mid = len(alist) // 2
        left = hybridSort(alist[:mid])
        right = hybridSort(alist[mid:])
        result = merge(left[0], right[0])

        elapsed_time = time.time() - start_time
        return result, elapsed_time


def quickSort(alist, pivot='first'):
    """Perform Quick Sort on a list."""
    start_time = time.time()

    # Helper function for partitioning the list
    def partition(arr, low, high, pivot):
        if pivot == 'middle':
            # Use middle element as pivot
            pivot_index = (low + high) // 2
            arr[low], arr[pivot_index] = arr[pivot_index], arr[low]

        pivot_value = arr[low]
        left = low + 1
        right = high

        done = False
        while not done:
            while left <= right and arr[left] <= pivot_value:
                left = left + 1
            while arr[right] >= pivot_value and right >= left:
                right = right - 1
            if right < left:
                done = True
            else:
                arr[left], arr[right] = arr[right], arr[left]

        arr[low], arr[right] = arr[right], arr[low]
        return right

    # Helper function for quick sort
    def quick_sort(arr, low, high, pivot):
        if low < high:
            # Find partition index
            pi = partition(arr, low, high, pivot)

            # Recursively sort the two partitions
            quick_sort(arr, low, pi - 1, pivot)
            quick_sort(arr, pi + 1, high, pivot)

    quick_sort(alist, 0, len(alist) - 1, pivot)

    elapsed_time = time.time() - start_time
    return alist, elapsed_time


def radixSort(alist):
    """Perform Radix Sort on a list."""
    start_time = time.time()

    # Find the maximum number to know the number of digits
    max_num = max(alist)
    exp = 1

    while max_num // exp > 0:
        counting_sort(alist, exp)
        exp *= 10

    elapsed_time = time.time() - start_time
    return alist, elapsed_time


def counting_sort(alist, exp):
    """
    Perform counting sort on a list of integers based on the specified exponent.

    Parameters:
    - alist (List[int]): The input list of integers to be sorted.
    - exp (int): The exponent used to determine the digit's place value for sorting.

    Returns:
    - List[int]: The sorted list in ascending order.

    Counting sort is a linear time sorting algorithm that sorts integers by counting
    the occurrences of each digit at a particular place value. The `exp` parameter
    determines the place value (e.g., units, tens, hundreds) for sorting.

    Note: This implementation assumes that the input integers are non-negative.
    """
    n = len(alist)
    output = [0] * n
    count = [0] * 10

    # Count the occurrences of each digit
    for i in range(n):
        index = alist[i] // exp
        count[index % 10] += 1

    # Update count to store the position of each digit in the output array
    for i in range(1, 10):
        count[i] += count[i - 1]

    # Build the output array in a stable manner (start from the end)
    i = n - 1
    while i >= 0:
        index = alist[i] // exp
        output[count[index % 10] - 1] = alist[i]
        count[index % 10] -= 1
        i -= 1

    # Copy the sorted elements back to the original array
    for i in range(n):
        alist[i] = output[i]

    return alist


if __name__ == '__main__':
    """ Check if the program is being run directly (i.e. not being imported) """

    def testFunction(sort_function, alist):
        """ A test utility function. """
        alist2 = alist.copy()
        res = sort_function(list(alist))
        print(f"Using {sort_function.__name__} to sort list: {alist[:10]}... w/ {len(alist)} items")
        print(f"    sort time: {res[1]:.4f} seconds")
        alist2.sort()
        print(f"    sorted correctly?: {'y :)' if res[0] == alist2 else 'n :('}")

    list1 = [54, 26, 93, 17, 77, 31, 44, 55, 20]  # helpful for early testing
    testFunction(bubbleSort, list(list1))
    testFunction(insertionSort, list(list1))
    testFunction(mergeSort, list(list1))
    testFunction(hybridSort, list(list1))
    testFunction(quickSort, list(list1))
    testFunction(radixSort, list(list1))

    random.seed(1)
    list2 = list(range(5000))
    random.shuffle(list2)
    testFunction(bubbleSort, list(list2))
    testFunction(insertionSort, list(list2))
    testFunction(mergeSort, list(list2))
    testFunction(hybridSort, list(list2))
    testFunction(quickSort, list(list2))
    testFunction(radixSort, list(list2))

    list3 = list(range(6000, 1000, -1))
    testFunction(bubbleSort, list(list3))
    testFunction(insertionSort, list(list3))
    testFunction(mergeSort, list(list3))
    testFunction(hybridSort, list(list3))
    testFunction(quickSort, list(list3))
    testFunction(radixSort, list(list3))
