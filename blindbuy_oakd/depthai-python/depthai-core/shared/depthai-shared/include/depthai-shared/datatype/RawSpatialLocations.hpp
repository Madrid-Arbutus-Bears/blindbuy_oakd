#pragma once
#include <cstdint>
#include <nlohmann/json.hpp>
#include <vector>

#include "DatatypeEnum.hpp"
#include "RawBuffer.hpp"
#include "RawImgFrame.hpp"
#include "RawSpatialLocationCalculatorConfig.hpp"
#include "depthai-shared/common/Point3f.hpp"
#include "depthai-shared/common/Rect.hpp"

namespace dai {

/**
 * Spatial location information structure
 *
 * Contains configuration data, average depth for the calculated ROI on depth map.
 * Together with spatial coordinates: x,y,z. Origin is the center of ROI.
 * Units are in millimeters.
 */
struct SpatialLocations {
    SpatialLocationCalculatorConfigData config;
    float depthAverage;
    Point3f spatialCoordinates;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(SpatialLocations, config, depthAverage, spatialCoordinates);

struct RawSpatialLocations : public RawBuffer {
    std::vector<SpatialLocations> spatialLocations;

    void serialize(std::vector<std::uint8_t>& metadata, DatatypeEnum& datatype) override {
        nlohmann::json j = *this;
        metadata = nlohmann::json::to_msgpack(j);
        datatype = DatatypeEnum::SpatialLocationCalculatorData;
    };

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(RawSpatialLocations, spatialLocations);
};

}  // namespace dai