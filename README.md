# libguiding
It's a guiding library!  
Our dream has finally come true!
_(at least in a very limited sense…)_

## Usage
To use this library, you need to let it know what types to use for vectors and floats:

```c++
#include <yourstuff>

namespace guiding {

  using Float = float;
//using Float = double;


  // use whatever vector library you want to use!
  // (it only needs to support subscripting)
  template<int Dim>
  using VectorXf<D> = std::array<Float, Dim>;
//using VectorXf<D> = Eigen::Matrix<Float, Dim, 1>;

}

#include <guiding/distributions/btree.h>
#include <guiding/wrappers/kdtree.h>

{your code}
```

You can then compose guiding trees like this:

```c++
// A 2D-BTree embedded in a 3D-KDTree is represented by:
using Muller = KDTreeWrapper<3, BTreeWrapper<2, Spectrum>>;

// we can inform libguiding how to deal with non-scalar types
// by specializing the guiding::target() template:
template<>
guiding::Float guiding::target(const Spectrum &s) {
  // probability density should be proportional to average
  return s.average();
}
```

## Compilation
Just add it as a CMake subdirectory to your project!

## Demo
You can also directly compile `libguiding` and look at some demos
built in the `tst/` directory.
