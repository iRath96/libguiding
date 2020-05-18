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
using Muller =
guiding::Wrapper< // takes care of MIS, building and target function
  MySample,       // data provided to the target function
  KDTree<3,       // spatial cache: 3D KD-Tree
    BTree<2,      // directional cache: 2D B-Tree
      Spectrum    // leaf nodes should also store spectrum data
    >
  >
>;

// you can configure it like this:
auto guiding = Muller({ // wrapper settings
  .uniformProb = 0.1f,
  //.target = myFunction // if you want a custom target function

  .child = { // KD-Tree settings
    .maxDepth       = 12,
    .splitThreshold = 1000.f,
    .splitting      = TreeSplitting::EWeight,
    .filtering      = TreeFilter::EStochastic,

    .child = { // B-Tree settings
      .maxDepth        = 16,
      .splitThreshold  = 0.005f,
      .leafReweighting = true,
      .filtering       = TreeFilter::EBox,

      .child = { // leaf node settings
        .secondMoment = true
      }
    }
  }
});
```

Working with these guiding structures is especially easy:

```c++
VectorXf<3> x = rnd.get3D();
VectorXf<2> d = rnd.get2D();

Float pdf  = guiding.sample(x, d); // takes care of MIS (uniform)
MySample f = integrand(x, d);      // evaluate your integrand
guiding.splat(f, 1/pdf, x, d);     // will re-build the tree automatically
```

## Compilation
Just add it as a CMake subdirectory to your project!

## Demo
You can also directly compile `libguiding` and look at some demos
built in the `tst/` directory.
