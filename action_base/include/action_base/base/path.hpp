#ifndef PATH_H
#define PATH_H

#include <string>
#include <unistd.h>
#include <vector>

namespace Action {
namespace Base {

    class Path {
    public:
        Path() {};
        ~Path() {};

        std::string GetRelativePath()
        {
            char* buf;
            buf = get_current_dir_name();
            std::string path = buf;
            free(buf); // 一定记得free,因为是由get_current_dir_name()函数动态分配的内存
            return path;
        }

        std::string GetAbsolutePath()
        {
            const int size = 255;
            char buf[size];
            getcwd(buf, size);
            std::string path = buf;
            free(buf);
            return path;
        }
    };
}
}
#endif