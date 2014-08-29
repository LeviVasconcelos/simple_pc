#ifndef simple_pc_H
#define simple_pc_H
#include <stdlib.h>
#include <string.h>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <rosbag/message_instance.h>
#include <std_msgs/Int32.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CameraInfo.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/simple_filter.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <list>

#include <opencv/cv.h>
#include <opencv/ml.h>
#include <cv_bridge/cv_bridge.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/cloud_viewer.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/core/core.hpp>

#include <string.h>
#include <boost/foreach.hpp>
#include <boost/bind.hpp>
#define foreach BOOST_FOREACH
class RGBDpair {
public:
    RGBDpair(sensor_msgs::Image::ConstPtr d,
             sensor_msgs::Image::ConstPtr rgb):depthImg(d),rgbImg(rgb) {}
    ~RGBDpair();
    sensor_msgs::Image::ConstPtr depthImg,rgbImg;
};

class simple_pc {
public:
    simple_pc(char* filename,std::vector<std::string> topics,int buffer_size);
    ~simple_pc();

    static void sync_XtionCallBack(const sensor_msgs::Image::ConstPtr dImg,
                                   const sensor_msgs::Image::ConstPtr rgbImg,
                                   simple_pc *obj);

    bool fillBuffer();
    void flushImgBuffer(const std_msgs::Int32& i);
    void pointCallBack(pcl::PCLPointCloud2ConstPtr pcd);
    std::list<pcl::PointCloud<pcl::PointXYZ>::Ptr > pcds;
    cv::Mat nextImg();
    pcl::PointCloud<pcl::PointXYZ>::Ptr nextCloud();
    //    void flushBuffer(int i);
    ros::Subscriber sub_;
    ros::Subscriber pointSub_;
    ros::Publisher pointPub_;
    ros::Publisher dethPub_;
    ros::Publisher rgbPub_;
    ros::Publisher depthInfoPub_;
    ros::Publisher rgbInfoPub_;
    ros::NodeHandle nh_;
    std::list<RGBDpair*> imgBuffer;

    int current_filling;
    sensor_msgs::CameraInfo depthInfo;

private:
    rosbag::Bag bagFile;
    rosbag::View bagViewer;
    rosbag::View::iterator currentBagPosition;
    int buffer_size;
    sensor_msgs::CameraInfo currentCameraInfo;
    std::vector<std::string> topics;
    std::list<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> pointsBuffer;
    sensor_msgs::CameraInfo::ConstPtr depthInfoPtr;
    sensor_msgs::CameraInfo::ConstPtr rgbInfoPtr;
    sensor_msgs::CameraInfo rgbInfo;

};

template <class M>
class BagSubscriber : public message_filters::SimpleFilter<M>
{
public:
    void newMessage(const boost::shared_ptr<M const> &msg);
};


#endif // simple_pc_H
