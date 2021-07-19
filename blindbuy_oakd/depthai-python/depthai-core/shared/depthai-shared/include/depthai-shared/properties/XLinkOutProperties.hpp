#pragma once

#include <nlohmann/json.hpp>

namespace dai {

/**
 * Properties for XLinkOut which define stream name
 */
struct XLinkOutProperties {
    /**
     * Set a limit to how many packets will be sent further to host
     */
    float maxFpsLimit = -1;

    /**
     * Name of stream
     */
    std::string streamName;

    /**
     * Whether to transfer data or only object attributes
     */
    bool metadataOnly = false;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(XLinkOutProperties, maxFpsLimit, streamName, metadataOnly);

}  // namespace dai
