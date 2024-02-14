## Assignment 2

### Coding directions

In this second assignment we will be implementing several common sorting
algorithms but with some slight variations. As with the first assignment,
you'll use assign2.py as a starting point and complete the functions from
there. Note that for this second assignment, you'll likely want to implement
other "helper" functions that can be called from the main sorting functions you
will implment.  For example, with `quickSort()` you will likely want to
implement a separate `partition()` function. 

The exact sorting algorithms you will be implementing are: 
* bubbleSort(alist) 
* insertionSort(alist) 
* mergeSort(alist) 
* quickSort(alist)
* hybridSort(alist) 
* radixSort(alist)


Note that your `hybridSort()` function should be a combination of your
insertionSort() and mergeSort() functions. That is, when the size of input list
is small (e.g. n < 100), then it should use insertionSort(). When the size of
the input is large (e.g. n > 100), then it should use mergeSort(). This is
actually how the built-in sorting method in Python and Java (and probably other
languages) is implemented. Be sure that you implement this correctly!

As with Assignment 1, be sure to document your code well, include your name, etc.

Also as with Assignment 1, you will need to return the runtime so that we can
compare the performance of the algorithms. Additionally, you'll again be using
Generative AI to help with this (e.g.  ChatGPT). As before, you can ask the
Generative AI to help with coding and implementation. But, as before, you
should be sure to ask the Generative AI to explain the differences between the
algorithms, their strengths, weaknesses, etc. 

However, this time when you use ChatGPT, you will also want to ask it to quiz
you on your knowledge of these sorting algorithms. Here is a link to a dialogue
of what this might look like: 
* [https://chat.openai.com/share/549c66e4-6352-4997-bea7-eabb6e7fdafe](https://chat.openai.com/share/549c66e4-6352-4997-bea7-eabb6e7fdafe)

See the Assignment in Canvas for more details, and be sure to look closely at the Rubric. 


