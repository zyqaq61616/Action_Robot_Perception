#include "action_node/perception/action_perception_2d_detection.h"
namespace Action {
namespace Perception {
    Perception2dDetNode::Perception2dDetNode(ros::NodeHandle& n)
    {
        this->image_sub = n.subscribe("/camera/color/camera_info", 1, &Perception2dDetNode::GetImage, this);
        this->bbox_2d_pub = n.advertise<action_msg::ActionBoundingBox2DList>("BBox_2d", 100);
    }
    void Perception2dDetNode::GetImage(const sensor_msgs::ImageConstPtr& msg)
    {
    }
    Perception2dDetNode::~Perception2dDetNode()
    {
    }
}
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "action_perception_2d_detection");
    ros::NodeHandle n;
    Action::Perception::Perception2dDetNode node(n);
    ros::spin();
    return 0;
}