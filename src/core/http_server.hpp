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
        string first_line=request_str.substr(0,first_line_end);
        size_t space1=first_line.find(' ');
        if(space1!=string::npos){
            request.method=first_line.substr(0,space1);
            size_t space2=first_line.find(' ',space1+1);
            if(space1!=string::npos){
                request.path=request_str.substr(space1+1,space2-space1-1);
            }
        }
        size_t content_type_pos=request_str.find("Content-Type:");
        if(content_type_pos!=string::npos){
            size_t line_end=request_str.find("\r\n",content_type_pos);
            request.content_type=request_str.substr(content_type_pos+14,line_end-content_type_pos-14);
            size_t start=request.content_type.find_first_not_of(" \t");
            size_t end = request.content_type.find_last_not_of(" \t\r\n");
            if (start != std::string::npos && end != std::string::npos) {
                request.content_type = request.content_type.substr(start, end - start + 1);
            }
        }
        size_t body_pos = request_str.find("\r\n\r\n");
        if (body_pos != std::string::npos) {
            request.body = request_str.substr(body_pos + 4);
        }
        
        return request;
    }

    string build_response(const HttpResponse& response){
        string result;
        result+="HTTP/1.1";
        result+=to_string(response.status_code);
         result += "HTTP/1.1 ";
        result += std::to_string(response.status_code);
        
        if (response.status_code == 200) {
            result += " OK\r\n";
        } else if (response.status_code == 404) {
            result += " Not Found\r\n";
        } else if (response.status_code == 500) {
            result += " Internal Server Error\r\n";
        } else {
            result += "\r\n";
        }
        
        // Content-Type
        result += "Content-Type: ";
        result += response.content_type;
        result += "\r\n";
        
        // Content-Length
        result += "Content-Length: ";
        result += std::to_string(response.body.size());
        result += "\r\n";
        
        // 空行
        result += "\r\n";
        
        // Body
        result += response.body;
        
        return result;
    }
}


}