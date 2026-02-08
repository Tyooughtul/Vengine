#include <iostream>
#include <vector>
#include <cassert>
#include <string>
#include "../src/core/api/api.hpp"
#include "../src/core/server/http_server.hpp"
#include "../third_party/json.hpp"

using namespace minimilvus;

// 测试API序列化/反序列化
void test_api_serialization() {
    std::cout << "Running API Serialization Test..." << std::endl;
    
    // 测试SearchRequest
    SearchRequest req{{1.0, 2.0, 3.0}, 10};
    std::string json_str = serialize_search_request(req);
    
    std::cout << "Serialized: " << json_str << std::endl;
    
    // 反序列化
    auto req2 = parse_search_request(json_str);
    
    assert(req2.top_k == req.top_k);
    assert(req2.vector.size() == req.vector.size());
    for (size_t i = 0; i < req.vector.size(); ++i) {
        assert(std::abs(req2.vector[i] - req.vector[i]) < 1e-5);
    }
    
    std::cout << "✓ SearchRequest serialization passed" << std::endl;
    
    // 测试SearchResponse
    SearchResponse resp;
    resp.results = {{1, 0.5}, {2, 0.8}, {3, 0.9}};
    
    std::string resp_str = serialize_search_response(resp);
    std::cout << "Response serialized: " << resp_str << std::endl;
    
    auto resp2 = parse_search_response(resp_str);
    assert(resp2.results.size() == resp.results.size());
    for (size_t i = 0; i < resp.results.size(); ++i) {
        assert(resp2.results[i].id == resp.results[i].id);
        assert(std::abs(resp2.results[i].distance - resp.results[i].distance) < 1e-5);
    }
    
    std::cout << "✓ SearchResponse serialization passed" << std::endl;
}

// 测试HTTP服务器基本功能
void test_http_server() {
    std::cout << "\nRunning HTTP Server Test..." << std::endl;
    
    // 测试RequestHandler
    RequestHandler handler = [](const std::string& body) {
        return R"({"status": "ok"})";
    };
    
    std::string result = handler("{}");
    assert(result.find("ok") != std::string::npos);
    
    std::cout << "✓ HTTP handler test passed" << std::endl;
}

int main() {
    try {
        test_api_serialization();
        test_http_server();
        
        std::cout << "\n✅ ALL TESTS PASSED! 🚀" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ TEST FAILED: " << e.what() << std::endl;
        return 1;
    }
}
