
#include <boost/make_shared.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/point_representation.h>

#include <pcl/io/pcd_io.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>

#include <pcl/features/normal_3d.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/features/range_image_border_extractor.h>
#include <pcl/keypoints/narf_keypoint.h>

#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/transforms.h>

#include <pcl/range_image/range_image.h>


#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/point_cloud_color_handlers.h>

#include <opencv/cv.h>
#include <opencv/ml.h>
#include <opencv/highgui.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/core/core.hpp>

#include <cstdio>

using pcl::visualization::PointCloudColorHandlerGenericField;
using pcl::visualization::PointCloudColorHandlerCustom;

//convenient typedefs
typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointNormal PointNormalT;
typedef pcl::PointCloud<PointNormalT> PointCloudWithNormals;
struct PCD
{
  PointCloud::Ptr cloud;
  std::string f_name;

  PCD() : cloud (new PointCloud) {}
};
struct PCDComparator
{
  bool operator () (const PCD& p1, const PCD& p2)
  {
    return (p1.f_name < p2.f_name);
  }
};
class registrationModule {
public:
    registrationModule();
    int vp_1, vp_2; //viewports ids.

    void addCloud(PointCloud::Ptr cloud);

    void compute();
    void showCloudsRight(const PointCloudWithNormals::Ptr cloud_target , const PointCloudWithNormals::Ptr cloud_source);
    void showCloudsLeft(const PointCloud::Ptr cloud_target, const PointCloud::Ptr cloud_source);
    void pairAlign (bool downsample = false);
    void prepareToAlign();
    void getRangeImage();
    pcl::visualization::PCLVisualizer *p;

    PointCloud::Ptr pcd_tgt;
    PointCloud::Ptr pcd_src;
    PointCloud::Ptr registered_cloud;

    pcl::RangeImage rimg;

    std::vector<PCD, Eigen::aligned_allocator<PCD> > data;
    Eigen::Matrix4f final_transform;


    //Eigen::Matrix4f final_transform;
};







// Define a new point representation for < x, y, z, curvature >
class MyPointRepresentation : public pcl::PointRepresentation <PointNormalT>
{
  using pcl::PointRepresentation<PointNormalT>::nr_dimensions_;
public:
  MyPointRepresentation ()
  {
    // Define the number of dimensions
    nr_dimensions_ = 4;
  }

  // Override the copyToFloatArray method to define our feature vector
  virtual void copyToFloatArray (const PointNormalT &p, float * out) const
  {
    // < x, y, z, curvature >
    out[0] = p.x;
    out[1] = p.y;
    out[2] = p.z;
    out[3] = p.curvature;
  }
};
