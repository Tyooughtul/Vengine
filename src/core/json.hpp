/**
 * @file    json.hpp
 * @brief   简单的 JSON 序列化/反序列化
 * @details 只支持基本功能，用于学习目的
 * @author  Tyooughtul
 */

#pragma once

#include <string>
#include <vector>
#include <map>
#include <sstream>
#include <algorithm>

namespace minimilvus {

/**
 * @brief   JSON 值类型
 */
enum class JsonType {
    Null,
    Bool,
    Number,
    String,
    Array,
    Object
};

/**
 * @brief   JSON 值类
 * @details 支持多种 JSON 类型
 */
class JsonValue {
public:
    /**
     * @brief   构造函数（空）
     */
    JsonValue() : type_(JsonType::Null) {}
    
    /**
     * @brief   构造函数（字符串）
     */
    explicit JsonValue(const std::string& value) 
        : type_(JsonType::String), string_value_(value) {}
    
    /**
     * @brief   构造函数（数字）
     */
    explicit JsonValue(double value) 
        : type_(JsonType::Number), number_value_(value) {}
    
    /**
     * @brief   构造函数（布尔）
     */
    explicit JsonValue(bool value) 
        : type_(JsonType::Bool), bool_value_(value) {}
    
    /**
     * @brief   构造函数（数组）
     */
    explicit JsonValue(const std::vector<JsonValue>& value) 
        : type_(JsonType::Array), array_value_(value) {}
    
    /**
     * @brief   构造函数（对象）
     */
    explicit JsonValue(const std::map<std::string, JsonValue>& value) 
        : type_(JsonType::Object), object_value_(value) {}
    
    /**
     * @brief   序列化为 JSON 字符串
     * @return  JSON 字符串
     */
    std::string serialize() const {
        switch (type_) {
            case JsonType::Null:
                return "null";
            case JsonType::Bool:
                return bool_value_ ? "true" : "false";
            case JsonType::Number:
                return serialize_number();
            case JsonType::String:
                return serialize_string();
            case JsonType::Array:
                return serialize_array();
            case JsonType::Object:
                return serialize_object();
            default:
                return "null";
        }
    }
    
    /**
     * @brief   从 JSON 字符串解析
     * @param   json_str    JSON 字符串
     * @return  JsonValue
     * @note    简化版本，只支持基本解析
     */
    static JsonValue parse(const std::string& json_str) {
        // 去除前后空格
        size_t start = json_str.find_first_not_of(" \t\r\n");
        size_t end = json_str.find_last_not_of(" \t\r\n");
        if (start == std::string::npos) {
            return JsonValue();
        }
        
        std::string trimmed = json_str.substr(start, end - start + 1);
        
        // 简化：只解析对象
        if (trimmed.empty() || trimmed[0] != '{') {
            return JsonValue();
        }
        
        std::map<std::string, JsonValue> obj;
        size_t pos = 1;  // 跳过 {
        
        while (pos < trimmed.size() && trimmed[pos] != '}') {
            // 跳过空白
            pos = trimmed.find_first_not_of(" \t\r\n", pos);
            if (pos == std::string::npos || trimmed[pos] == '}') break;
            
            // 解析 key
            size_t key_start = trimmed.find('"', pos);
            if (key_start == std::string::npos) break;
            size_t key_end = trimmed.find('"', key_start + 1);
            if (key_end == std::string::npos) break;
            
            std::string key = trimmed.substr(key_start + 1, key_end - key_start - 1);
            pos = key_end + 1;
            
            // 跳过 :
            pos = trimmed.find(':', pos);
            if (pos == std::string::npos) break;
            pos++;
            
            // 解析 value（简化：只支持字符串和数字）
            pos = trimmed.find_first_not_of(" \t\r\n", pos);
            if (pos == std::string::npos) break;
            
            if (trimmed[pos] == '"') {
                // 字符串值
                size_t value_end = trimmed.find('"', pos + 1);
                if (value_end == std::string::npos) break;
                std::string value = trimmed.substr(pos + 1, value_end - pos - 1);
                obj[key] = JsonValue(value);
                pos = value_end + 1;
            } else if (std::isdigit(trimmed[pos]) || trimmed[pos] == '-') {
                // 数字值
                size_t value_end = trimmed.find_first_of(",}", pos);
                if (value_end == std::string::npos) value_end = trimmed.size();
                std::string value_str = trimmed.substr(pos, value_end - pos);
                double value = std::stod(value_str);
                obj[key] = JsonValue(value);
                pos = value_end;
            } else if (trimmed[pos] == '[') {
                // 数组值（简化：只支持数字数组）
                size_t array_end = trimmed.find(']', pos);
                if (array_end == std::string::npos) break;
                
                std::string array_str = trimmed.substr(pos + 1, array_end - pos - 1);
                std::vector<JsonValue> array;
                
                size_t elem_start = 0;
                while (elem_start < array_str.size()) {
                    size_t elem_end = array_str.find(',', elem_start);
                    if (elem_end == std::string::npos) elem_end = array_str.size();
                    
                    std::string elem = array_str.substr(elem_start, elem_end - elem_start);
                    elem = trim(elem);
                    if (!elem.empty()) {
                        double value = std::stod(elem);
                        array.push_back(JsonValue(value));
                    }
                    
                    elem_start = elem_end + 1;
                }
                
                obj[key] = JsonValue(array);
                pos = array_end + 1;
            }
            
            // 跳过 ,
            pos = trimmed.find(',', pos);
            if (pos == std::string::npos) break;
            pos++;
        }
        
        return JsonValue(obj);
    }
    
private:
    JsonType type_;
    std::string string_value_;
    double number_value_;
    bool bool_value_;
    std::vector<JsonValue> array_value_;
    std::map<std::string, JsonValue> object_value_;
    
    /**
     * @brief   序列化数字
     */
    std::string serialize_number() const {
        std::ostringstream oss;
        oss << number_value_;
        return oss.str();
    }
    
    /**
     * @brief   序列化字符串
     */
    std::string serialize_string() const {
        std::string result = "\"";
        for (char c : string_value_) {
            if (c == '"') {
                result += "\\\"";
            } else if (c == '\\') {
                result += "\\\\";
            } else if (c == '\n') {
                result += "\\n";
            } else if (c == '\r') {
                result += "\\r";
            } else if (c == '\t') {
                result += "\\t";
            } else {
                result += c;
            }
        }
        result += "\"";
        return result;
    }
    
    /**
     * @brief   序列化数组
     */
    std::string serialize_array() const {
        std::string result = "[";
        for (size_t i = 0; i < array_value_.size(); ++i) {
            if (i > 0) result += ",";
            result += array_value_[i].serialize();
        }
        result += "]";
        return result;
    }
    
    /**
     * @brief   序列化对象
     */
    std::string serialize_object() const {
        std::string result = "{";
        bool first = true;
        for (const auto& pair : object_value_) {
            if (!first) result += ",";
            first = false;
            
            result += "\"";
            result += pair.first;
            result += "\":";
            result += pair.second.serialize();
        }
        result += "}";
        return result;
    }
    
    /**
     * @brief   去除字符串前后空格
     */
    static std::string trim(const std::string& str) {
        size_t start = str.find_first_not_of(" \t\r\n");
        size_t end = str.find_last_not_of(" \t\r\n");
        if (start == std::string::npos) {
            return "";
        }
        return str.substr(start, end - start + 1);
    }
};

}  // namespace minimilvus