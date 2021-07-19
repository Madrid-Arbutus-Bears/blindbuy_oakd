//  To parse this JSON data, first install
//
//      Boost     http://www.boost.org
//      json.hpp  https://github.com/nlohmann/json
//
//  Then include this file, and then do
//
//     SPIOutProperties.hpp data = nlohmann::json::parse(jsonString);

#pragma once

#include <nlohmann/json.hpp>

namespace dai {

/**
 * Properties for SPIOut node
 */
struct SPIOutProperties {
    /// Output stream name
    std::string streamName;
    /// SPI bus to use
    int busId;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(SPIOutProperties, streamName, busId);

}  // namespace dai
