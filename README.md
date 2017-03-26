# GeometricMultigrid
A simple two dimensional geometric multigrid solver for poisson equation in python. It uses numpy.

```c++
             _______ _______ _______ _______ 
            |       :       |       :       |
            |   +   :   +   |   +   :   +   |
            |.......@.......|.......@.......|
            |       :       |       :       |
            |   +   :   +   |   +   :   +   |
            |_______:_______#_______:_______|
            |       :       |       :       |
            |   +   :   +   |   +   :   +   |
            |.......@.......|.......@.......|
            |       :       |       :       |
            |   +   :   +   |   +   :   +   |
            |_______:_______|_______:_______|
```



**Details**:

- *Smoother*: Two color Gauss-Seidel
- *Restriction*: Nearest points average
- *Prolongation*: Bilinear interpolation
- V cycle and FMG



The examples show V cycle and FMG in action. An example script demonstrates the method of deferred correction to obtain 4th order accuracy.  

