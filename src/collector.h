#ifndef COLLECTOR_H
#define COLLECTOR_H
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/point_cloud_color_handlers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/io/pcd_io.h>
#include <opencv/cv.h>
#include <opencv/ml.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/core/core.hpp>
#include <sensor_msgs/CameraInfo.h>
#include <math.h>
#include <list>
#include <vector>
#include <iterator>
#include <sstream>
#include <fstream>
#include "registrationModule.h"

void visualize_normals(cv::Mat normals);
void loadNormals(char* filename,cv::Mat &m);
void calculate_std_using_median(cv::Mat source,float & median,float& std);
void normalMean_median(cv::Rect roi,cv::Mat normals,cv::Mat depth, cv::Point3f &result,cv::Mat &coloredDepth);
void entropy_calc(cv::Mat src, float* probs, float & ent, bool & valid);
void entropy_calc2(float* src,int size, float* probs, float & ent, bool & valid);
void pre_process_cloud(pcl::PointCloud<pcl::PointXYZ>::Ptr &pcd,bool downsample);

class descriptor {
public:
    descriptor() {
        features = NULL;
        quality = 0;
    }
    ~descriptor() { delete features; }
    float* features;
    float quality; //utilizar para non-maximal supression
};

class kvd_descriptor {
public:
    kvd_descriptor(char* filename);
    ~kvd_descriptor();
    std::vector<cv::Mat> posDotNormals;
    std::vector<cv::Mat> negDotNormals;
    std::vector<cv::Mat> posCurvatures;
    std::vector<cv::Mat> negCurvatures;
    std::vector<cv::Mat> pos_descriptors;
    std::vector<cv::Mat> neg_descriptors;
    std::vector<cv::Mat> pos_ED;
    std::vector<cv::Mat> neg_ED;
    cv::Mat kvd_pos;
    cv::Mat kvd_neg;

    void build_train_test(float train_percentage,cv::Mat &train,cv::Mat &train_targets,cv::Mat &test,cv::Mat &test_targets);
    void visualize(int size,int i);
    int rows;
    int cols;

    void compute_descriptors(float treshold,float ed_threshold,int pos0,int posF);
    cv::Mat compute(float treshold,float dist_treshold,cv::Mat N,cv::Mat K,cv::Mat ED);
    void readFromFile(char* filename);
};






















class collector {
public:
    collector(char* wName,cv::Mat mt,sensor_msgs::CameraInfo ci);
    ~collector();

    std::vector<pcl::PointXYZRGB>* bresenhamApplyTemp(int xC,int yC,int Radius,void (*f)(int,int,int,int,std::vector<pcl::PointXYZRGB>*,collector*));

    void windowCallBack(int event, int x, int y, int flags, void* userdata);
    void wcb_regionsSelector(int event,int x,int y, int flags, void* userdata);
    void window_cb_clickTest(int event, int x, int y, int flags, void* userdata);
    void compute_normals();
    std::vector<float> *bresenhamApply(int x,int y,int radius,void (*f)(int,int,int,int,std::vector<float>*,collector*));

    float bresenhamApply2(int xC,int yC,int Radius,int& idx,float* feat,float (*f)(int,int,int,int,float* ,int&,collector*));
    descriptor* compute_features(int x,int y);
    void findStalactite2();
    bool isMaximum(int i,int j,descriptor*** features);
    void showcloud(cv::Point3f *p = NULL,int size = 0);
    void getNormalMeans(int vis = 0);

    char* wName;
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcd;
    void setImg(cv::Mat img);
    void show();
    void getPointsDescriptors();
    void writeDescriptors(std::string prefix);
    void writeRois(std::string prefix);
    void writeNormals(std::string prefix);
    void loadTree(std::string filename);
    void extract_descriptor_information_from_region(int positive);
    void curvatureFilter(std::list<cv::Point> &pList);
    cv::Mat kvd(cv::Mat dotNormals,float treshold);

    int convertIdx(int x,int y);
    int inverseIdxX(int x);
    int inverseIdxY(int x);

    kvd_descriptor kvd_util;
    cv::Mat img;
    cv::Mat cloud;
    cv::Mat deph;
    cv::Mat coloredImg;
    cv::Mat normals;
    cv::Mat dN_matrix;
    cv::Mat K_matrix;
    CvDTree tree;
    cv::Point2d p_aux;
    cv::Rect r_aux;
    int counter_aux;
    int mouseControl;
    int ask_for_normals;
    sensor_msgs::CameraInfo camera_info;
    registrationModule reg;



    cv::Point2d center1;
    cv::Point2d center2;
    pcl::PointCloud<pcl::Normal>::Ptr pcl_normals;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud_gray;
    std::vector<pcl::PointXYZRGB>* pointList; //temp... delete later.
    std::vector<cv::Rect> roi_queue;
    std::vector<cv::Point3f> normals_queue;

private:

    int totalPosPoints;
    int totalNegPoints;
    std::list<cv::Point> positivePoints;
    std::list<cv::Point> negativePoints;
    std::vector<cv::Mat> posDotProducts; //vetor de matrizes para armazenar dot products das normais
    std::vector<cv::Mat> negDotProducts; //vetor de matrizes para armazenar dot products das normais
    std::vector<cv::Mat> posK;
    std::vector<cv::Mat> negK;
    std::vector<cv::Mat> posED;
    std::vector<cv::Mat> negED;

    void create_cloud( );


    //DEBUG
    void drawCircle(int xC,int yC,int Radius);
    void Draw(int x,int y,int xC,int yC);


};

#endif


