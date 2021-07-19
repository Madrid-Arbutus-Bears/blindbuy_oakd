#pragma once

// std
#include <cstdint>

// libraries
#include "nlohmann/json.hpp"

namespace dai {

struct Timestamp {
    int64_t sec, nsec;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Timestamp, sec, nsec);
};

}  // namespace dai
