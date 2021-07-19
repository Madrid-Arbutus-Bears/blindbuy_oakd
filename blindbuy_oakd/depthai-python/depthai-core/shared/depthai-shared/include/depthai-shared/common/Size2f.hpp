#pragma once

// std
#include <cstdint>

// libraries
#include "nlohmann/json.hpp"

namespace dai {

struct Size2f {
    Size2f() {}
    Size2f(float width, float height) {
        this->width = width;
        this->height = height;
    }
    float width, height;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Size2f, width, height);
};

}  // namespace dai
