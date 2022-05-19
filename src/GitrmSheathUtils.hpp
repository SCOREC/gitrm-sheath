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

    KOKKOS_FUNCTION
    Vector2 operator-();

    KOKKOS_FUNCTION
    double operator[](int i) const;

    KOKKOS_FUNCTION
    Vector2 operator+(Vector2 v);

    KOKKOS_FUNCTION
    Vector2 operator-(Vector2 v);

    KOKKOS_FUNCTION
    Vector2 operator*(double scalar) const;

    KOKKOS_FUNCTION
    Vector2 &operator+=(const Vector2 &v);

    KOKKOS_FUNCTION
    void operator=(const double &scalar);

    KOKKOS_FUNCTION
    double dot(Vector2 v);

    KOKKOS_FUNCTION
    double cross(Vector2 v);

    KOKKOS_FUNCTION
    double magnitude();
};

/**
 * @brief Default constructor.
 */
KOKKOS_INLINE_FUNCTION
Vector2::Vector2() : components_{0.0, 0.0} {}

/**
 * @brief Construct a vector with two given components.
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

/**
 * @brief Unitary minus, or negation.
 *
 * @return Vector2 with negated components.
 */
KOKKOS_INLINE_FUNCTION
Vector2 Vector2::operator-() { return Vector2(-components_[0], -components_[1]); }

/**
 * @brief Vector element retrieval.
 *
 * Allows easy read of const Vector2 components.
 *
 * @param[in] i Int, component of vector to read.
 * @return i-th component.
 */
KOKKOS_INLINE_FUNCTION
double Vector2::operator[](int i) const { return components_[i]; }

/**
 * @brief Vector addition.
 *
 * @param[in] v Another Vector2.
 * @return Sum of two vectors.
 */
KOKKOS_INLINE_FUNCTION
Vector2 Vector2::operator+(Vector2 v) {
    return Vector2(components_[0] + v[0],
                components_[1] + v[1]);
}

/**
 * @brief Vector subtraction.
 *
 * @param[in] v Another Vector2.
 * @return Difference of two vectors.
 */
KOKKOS_INLINE_FUNCTION
Vector2 Vector2::operator-(Vector2 v) {
    return Vector2(components_[0] - v[0],
                components_[1] - v[1]);
}

/**
 * @brief Scalar multiplication.
 *
 * @param[in] scalar Float.
 * @return Vector multiplied by scalar.
 */
KOKKOS_INLINE_FUNCTION
Vector2 Vector2::operator*(double scalar) const {
    return Vector2(components_[0]*scalar,
                components_[1]*scalar);
}

/**
 * @brief Vector increment.
 *
 * @param[in] v Another Vector2.
 * @return Sum of two vectors.
 */
KOKKOS_INLINE_FUNCTION
Vector2 &Vector2::operator+=(const Vector2 &v) {
    components_[0] += v.components_[0];
    components_[1] += v.components_[1];
    return *this;
}

/**
 * @brief Vector dot product.
 *
 * @param[in] v Another Vector2.
 * @return Dot product of two vectors.
 */
KOKKOS_INLINE_FUNCTION
double Vector2::dot(Vector2 v) {
    return components_[0]*v[0] +
        components_[1]*v[1];
}

/**
 * @brief Vector cross product z-component
 *
 * @param[in] v Another Vector2.
 * @return Z-component Cross product of two vectors.
 */
KOKKOS_INLINE_FUNCTION
double Vector2::cross(Vector2 v) {
    return (components_[0]*v[1] - components_[1]*v[0]);
}

/**
 * @brief Vector magnitude.
 *
 * @return Vector magnitude.
 */
KOKKOS_INLINE_FUNCTION
double Vector2::magnitude() {
    return sqrt(components_[0]*components_[0] +
                components_[1]*components_[1] );
}

/**
 * @brief Set all components to a scalar.
 */
KOKKOS_INLINE_FUNCTION
void Vector2::operator=(const double &scalar) {
    components_[0] = scalar;
    components_[1] = scalar;
}

}

#endif
