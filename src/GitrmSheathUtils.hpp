#ifndef GitrmSheathUtils_hpp
#define GitrmSheathUtils_hpp


namespace sheath{
class Vector2 {
public:
    double components_[2];

    KOKKOS_FUNCTION
    Vector2();

    /**
     * @brief Default destructor.
     */
    ~Vector2() = default;

    KOKKOS_FUNCTION
    Vector2(double x1, double x2);

    KOKKOS_FUNCTION
    Vector2(double x[2]);

    KOKKOS_FUNCTION
    double &operator[](int i);
};

/**
 * @brief Default constructor.
 */
KOKKOS_INLINE_FUNCTION
Vector2::Vector2() : components_{0.0, 0.0} {}

/**
 * @brief Construct a vector with three given components.
 *
 * @param[in] x1 x1-component of vector
 * @param[in] x2 x2-component of vector
 */
KOKKOS_INLINE_FUNCTION
Vector2::Vector2(double x1, double x2) : components_{x1, x2} {}

/**
 * @brief Construct a 2-vector from an array.
 *
 * @param[in] x double[2] array.
 */
KOKKOS_INLINE_FUNCTION
Vector2::Vector2(double x[2]) : components_{x[0], x[1]} {}

/**
 * @brief Vector element access overload.
 *
 * @param[in] i Int, component of vector to access.
 * @return i-th component.
 */
KOKKOS_INLINE_FUNCTION
double &Vector2::operator[](int i) { return components_[i]; }

}

#endif
