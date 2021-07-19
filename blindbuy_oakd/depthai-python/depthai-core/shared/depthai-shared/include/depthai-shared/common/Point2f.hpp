#pragma once

// std
#include <cstdint>

// libraries
#include "nlohmann/json.hpp"

namespace dai {

/**
 * Point2f structure
 *
 * x and y coordinates that define a 2D point.
 */
struct Point2f {
    Point2f() {}
    Point2f(float x, float y) {
        this->x = x;
        this->y = y;
    }
    float x, y;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Point2f, x, y);
};

}  // namespace dai
