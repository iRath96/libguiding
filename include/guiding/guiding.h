#ifndef HUSSAR_CORE_GUIDING_H
#define HUSSAR_CORE_GUIDING_H

#include <atomic>
#include <fstream>
#include <iostream>
#include <array>
#include <cassert>
#include <mutex>

namespace guiding {

/**
 * Writes an element to disk.
 * Override this if you store complex objects in your distribution that
 * need special serialization procedures.
 */
template<typename T>
void write(std::ostream &os, const T &t) {
    os.write((const char *)&t, sizeof(T));
}

/**
 * Reads an element to disk.
 * Override this if you store complex objects in your distribution that
 * need special deserialization procedures.
 */
template<typename T>
void read(std::istream &is, T &t) {
    is.read((char *)&t, sizeof(T));
}

/**
 * Assign a target to a distribution value (the density will equal to this, but normalized).
 * For instance, if you want to store Spectrums in your guiding distribution, do this;
 * `Float guiding::target(const Spectrum &x) { return x.average(); }`
 */
template<typename T>
Float target(const T &x, T &aux) {
    return (aux = Float(x));
}

template<typename V>
class atomic {
public:
    void operator+=(const V &v) {
        std::unique_lock lock(m_mutex);
        m_value += v;
    }

    void operator=(const V &value) {
        m_value = value;
    }

    void operator=(const atomic<V> &other) {
        m_value = other.m_value;
    }

    const V &value() const { return m_value; }

private:
    V m_value;
    std::mutex m_mutex;
};

template<>
class atomic<Float> : public std::atomic<Float> {
public:
    atomic() : std::atomic<Float>() {}

    atomic(const atomic<Float> &other) : std::atomic<Float>() {
        *this = other;
    }

    void operator=(const Float &value) {
        store(value, std::memory_order_relaxed);
    }

    void operator=(const atomic<Float> &other) {
        *this = other.load();
    }

    void operator+=(const Float &value) {
        auto current = load();
        while (!compare_exchange_weak(current, current + value));
    }
};

template<int D>
Float computeOverlap(const VectorXf<D> &min1, const VectorXf<D> &max1, const VectorXf<D> &min2, const VectorXf<D> &max2) {
    // @todo this ignores the fact that a hypervolume can extend beyond the [0,1) interval
    // using this directly will give you a bias if you are not using leaf reweighting
    // (directions at the corners will have smaller weights)

    Float overlap = 1;
    for (int i = 0; i < D; ++i)
        overlap *= std::max(std::min(max1[i], max2[i]) - std::max(min1[i], min2[i]), 0.0f);
    return overlap;
}

template<typename T>
class Leaf {
public:
    struct Settings {
        bool secondMoment = false;
    };

    typedef T Aux;

    atomic<Aux> aux;
    atomic<Float> weight;
    atomic<Float> density;

    void splat(const Settings &settings, Float density, const Aux &aux, Float weight) {
        if (settings.secondMoment)
            density *= density;
        
        this->aux     += aux     * weight;
        this->density += density * weight;
        this->weight  += weight;
    }

    void build(const Settings &settings) {
        density = density / weight;
        aux     = aux     / weight;

        if (settings.secondMoment)
            density = std::sqrt(density);
    }

    void refine(const Settings &settings) {
        aux     = Aux();
        weight  = 0.f;
        density = 0.f;
    }

    Float pdf(const Settings &settings) const {
        return density;
    }

    Leaf<T> sample(const Settings &settings) const {
        return *this;
    }

    const atomic<Aux> &estimate() const {
        return aux;
    }

    size_t totalNodeCount() const {
        return 1;
    }

    void dump(const std::string &prefix) const {
        std::cout << prefix << "Leaf (density=" << density << ", weight=" << weight << ")" << std::endl;
    }
};

template<typename ...Args>
struct is_empty {
    enum { value = 0 };
};

template<>
struct is_empty<> {
    enum { value = 1 };
};

//
// these meta-programming hacks might look horrendous,
// but they are not quite as horrible as OpenGL!
//

template<typename...>
struct RecurseChild {
    typedef void Type;
};

template<typename Child, typename Head, typename ...Tail>
struct RecurseChild<Child, Head, Tail...> {
    typedef typename RecurseChild<Child, Tail...>::Type::Child Type;
};

template<typename Child>
struct RecurseChild<Child> {
    typedef Child Type;
};

}

#endif
