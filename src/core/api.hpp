/**
 * @file    api.hpp
 * @brief   API 接口定义
 * @details 定义搜索请求和响应的数据结构
 * @author  Tyooughtul
 */

#pragma once

#include <vector>
#include <string>
#include "../dataset.hpp"

namespace minimilvus {

/**
 * @brief   搜索请求
 */
struct SearchRequest {
    std::vector<float> vector;  ///< 查询向量
    int top_k = 10;           ///< 返回结果数量
};

/**
 * @brief   搜索结果
 */
struct SearchResultItem {
    idx_t id;        ///< 向量ID
    float distance;  ///< 距离
};

/**
 * @brief   搜索响应
 */
struct SearchResponse {
    std::vector<SearchResultItem> results;  ///< 搜索结果列表
};

/**
 * @brief   将搜索请求序列化为 JSON
 * @param   request    搜索请求
 * @return  JSON 字符串
 */
std::string serialize_search_request(const SearchRequest& request) {
    std::map<std::string, JsonValue> obj;
    
    // 添加 vector
    std::vector<JsonValue> vec_array;
    for (float v : request.vector) {
        vec_array.push_back(JsonValue(v));
    }
    obj["vector"] = JsonValue(vec_array);
    
    // 添加 top_k
    obj["top_k"] = JsonValue(static_cast<double>(request.top_k));
    
    // 序列化
    JsonValue json(obj);
    return json.serialize();
}

/**
 * @brief   从 JSON 解析搜索请求
 * @param   json_str    JSON 字符串
 * @return  搜索请求
 */
SearchRequest parse_search_request(const std::string& json_str) {
    SearchRequest request;
    
    JsonValue json = JsonValue::parse(json_str);
    
    // 提取 vector（简化：假设是对象类型）
    if (json.serialize().find("\"vector\"") != std::string::npos) {
        // 手动解析 vector 数组
        size_t vec_start = json_str.find("\"vector\":");
        if (vec_start != std::string::npos) {
            vec_start = json_str.find('[', vec_start);
            if (vec_start != std::string::npos) {
                size_t vec_end = json_str.find(']', vec_start);
                if (vec_end != std::string::npos) {
                    std::string vec_str = json_str.substr(vec_start + 1, vec_end - vec_start - 1);
                    
                    // 解析数字数组
                    size_t elem_start = 0;
                    while (elem_start < vec_str.size()) {
                        size_t elem_end = vec_str.find(',', elem_start);
                        if (elem_end == std::string::npos) elem_end = vec_str.size();
                        
                        std::string elem = vec_str.substr(elem_start, elem_end - elem_start);
                        elem = JsonValue::trim(elem);
                        if (!elem.empty()) {
                            request.vector.push_back(std::stod(elem));
                        }
                        
                        elem_start = elem_end + 1;
                    }
                }
            }
        }
    }
    
    // 提取 top_k
    size_t topk_start = json_str.find("\"top_k\":");
    if (topk_start != std::string::npos) {
        size_t topk_end = json_str.find_first_of(",}", topk_start);
        if (topk_end != std::string::npos) {
            std::string topk_str = json_str.substr(topk_start + 8, topk_end - topk_start - 8);
            topk_str = JsonValue::trim(topk_str);
            if (!topk_str.empty()) {
                request.top_k = static_cast<int>(std::stod(topk_str));
            }
        }
    }
    
    return request;
}

/**
 * @brief   将搜索响应序列化为 JSON
 * @param   response    搜索响应
 * @return  JSON 字符串
 */
std::string serialize_search_response(const SearchResponse& response) {
    std::map<std::string, JsonValue> obj;
    
    // 添加 results 数组
    std::vector<JsonValue> results_array;
    for (const auto& item : response.results) {
        std::map<std::string, JsonValue> result_obj;
        result_obj["id"] = JsonValue(static_cast<double>(item.id));
        result_obj["distance"] = JsonValue(item.distance);
        results_array.push_back(JsonValue(result_obj));
    }
    obj["results"] = JsonValue(results_array);
    
    // 序列化
    JsonValue json(obj);
    return json.serialize();
}

}  // namespace minimilvus