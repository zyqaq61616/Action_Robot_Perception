#ifndef _ACTION_PERCEPTION_2D_DETECTION_H
#define _ACTION_PERCEPTION_2D_DETECTION_H
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include "action_msg/ActionBoundingBox2D.h"
#include "action_msg/ActionBoundingBox2DList.h"

namespace Action
{
namespace Perception
{
    class Perception2dDetNode
    {
        public:
            message_filters::Subscriber<sensor_msgs::Image>* color_image_sub;
            ros::Subscriber image_sub;
            ros::Publisher bbox_2d_pub;
            Perception2dDetNode(ros::NodeHandle &n);
            void GetImage(const sensor_msgs::ImageConstPtr& msg);
            ~Perception2dDetNode();
        private:

    };
}//Perception
}//Action

#endif