#include <string>
#include <functional>
#include <thread>
#include <mutex>
#include <atomic>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <arpa/inet.h>

namespace minimilvus {
struct HttpRequest {
    using namespace std;
    string method;
    string path;
    string body;
    string content_type;
};

struct HttpResponse {
    using namespace std;
    int status_code = 200;
    string content_type = "application/json";
    string body;
};

class HttpServer {
    using namespace std;
    using RequestHandler = function<string(const HttpRequest&)>;
    explicit HttpServer(int port):port_(port),running_(false){}






private:
    int port_;
    int server_fd_ = -1;
    std::atomic<bool> running_;

    void handle_client(int client_fd) {
        char buffer[4096];
        // 读到buffer，但是不会填满，最少也会留出来一个空位填充换行符
        ssize_t bytes_read = read(client_fd,buffer,sizeof(buffer)-1);
        if (bytes_read <= 0)return;
        buffer[bytes_read]='\0';
        string request_str(buffer);
        HttpRequest request = parse_request(request_str);

    }

    HttpRequest parse_request(const string& request_str){
        HttpRequest request;
        size_t first_line_end=request_str.find("\r\n");
        if(first_line_end==string::npos)return request;
    }
}


}