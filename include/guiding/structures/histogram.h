#ifndef HUSSAR_GUIDING_HISTOGRAMWRAPPER_H
#define HUSSAR_GUIDING_HISTOGRAMWRAPPER_H

#include <guiding/guiding.h>

#include <array>
#include <vector>
#include <fstream>
#include <cstring>
#include <cassert>

namespace guiding {

template<int D, typename C>
class HistogramWrapper {
public:
    static constexpr auto Dimension = D;

    typedef C Distribution;
    typedef VectorXf<Dimension> Vector;

    HistogramWrapper(int resolution, const Vector &min, const Vector &max)
    : m_resolution(resolution), m_min(min), m_max(max) {
        int count = 1;
        for (int dim = 0; dim < Dimension; ++dim)
            count *= m_resolution;
        
        m_sampling = new Distribution[count];
        m_training = new Distribution[count];
    }

    Distribution &training(const Vector &x) { return get(m_training, x); }
    Distribution &sampling(const Vector &x) { return get(m_sampling, x); }

    ~HistogramWrapper() {
        delete[] m_training;
        delete[] m_sampling;
    }

protected:
    Distribution &get(Distribution *container, const Vector &x) {
        int i = 0;
        for (int dim = 0; dim < Dimension; ++dim) {
            i *= m_resolution;
            
            Float v = (x[dim] - m_min[dim]) / (m_max[dim] - m_min[dim]);
            if (v < 0 || v >= 1)
                return nullptr;
            
            i += int(v * m_resolution);
        }

        return container[i];
    }

    Vector m_min, m_max;
    int m_resolution;

    Distribution *m_sampling, *m_training;
};

}

#endif
