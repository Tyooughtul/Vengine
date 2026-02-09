/**
 * @file    api.hpp
 * @brief   API 接口定义
 * @author  Tyooughtul
 */

#pragma once

#include <vector>
#include <string>
#include "../../third_party/json.hpp"
#include "dataset.hpp"

namespace minimilvus {

using json = nlohmann::json;

struct SearchRequest {
    std::vector<float> vector;
    int top_k = 10;
};

struct SearchResultItem {
    idx_t id;
    float distance;
};

struct SearchResponse {
    std::vector<SearchResultItem> results;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(SearchRequest, vector, top_k)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(SearchResultItem, id, distance)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(SearchResponse, results)

inline std::string serialize_search_request(const SearchRequest& request) {
    json j = request;
    return j.dump();
}

inline SearchRequest parse_search_request(const std::string& json_str) {
    json j = json::parse(json_str);
    return j.get<SearchRequest>();
}

inline std::string serialize_search_response(const SearchResponse& response) {
    json j = response;
    return j.dump();
}

inline SearchResponse parse_search_response(const std::string& json_str) {
    json j = json::parse(json_str);
    return j.get<SearchResponse>();
}

}  // namespace minimilvus
