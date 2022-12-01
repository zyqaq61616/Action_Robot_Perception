#include<iostream>
#include<vector>
#include<action_msg/ActionBoundingBox2D.h>
#include<action_msg/ActionBoundingBox2DList.h>
#include<ros/ros.h>
using namespace std;

int main(int argc, char **argv)
{
    action_msg::ActionBoundingBox2D a;
    ros::init(argc, argv, "test_node");
    ros::NodeHandle n;
    ros::Rate rate(1);
    cout<<a.Class<<endl;
    while (n.ok())
    {
        ROS_INFO("spin once");
        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}