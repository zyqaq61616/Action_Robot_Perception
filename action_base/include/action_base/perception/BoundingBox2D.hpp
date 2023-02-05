#include <vector>
#include <list>
#include <string>

namespace Action
{
    namespace Base
    {

        class BoundingBox2D
        {
            public:
                BoundingBox2D()
                {
                    this->x_min_=0;
                    this->y_min_=0;
                    this->x_max_=0;
                    this->y_max_=0;
                }
                BoundingBox2D(int x_min,int y_max,int x_max,int y_min)
                {
                    this->x_min_=x_min;
                    this->y_max_=y_max;

                    this->x_max_=x_max;
                    this->y_min_=y_min;
                }
            private:
                //Upper left
                int x_min_;
                int y_max_;

                //lower right
                int x_max_;
                int y_min_;
        };
    }
}