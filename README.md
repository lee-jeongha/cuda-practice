## cuda documentation <br>
https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html

## thread, threadblock, grid <br>
Consider adding 1 to the values of all the elements in the 100x100 matrix <br>
> Let's assume it's implemented in a two-dimensional array.<br>
> Adding each element is completely independent.<br>
> This case is especially called the 'Embarassingly parallel' because it is too simple among parallel problems.

If solved with CPU 1 core, do it sequentially for 10,000 elements
If you solve it with a GPU that can perform 200 cores simultaneously, you can perform it in parallel by grouping 200 times and repeat it 50 times
1. **Thread**: Units handled by one core in a GPU are called threads.
2. **Thread Block**: bundle of threads. working unit of 200 threads.
3. **Grid**: 50 pieces are grid size.

## command <br>
`nvcc -o test test.cu` <br>
`./test`
