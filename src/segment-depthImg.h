#ifndef S_DEPTH_IMG
#define S_DEPTH_IMG
#include "disjoint-set.h"
#include "segment-graph.h"
#include <opencv/cv.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/core/core.hpp>
#include <opencv/highgui.h>
#include <sensor_msgs/CameraInfo.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/integral_image_normal.h>
#include <math.h>

void cb_link(int event, int x, int y, int flags, void* userdata);
class depth_img {
public:
    depth_img(cv::Mat d,float* cInfo);
    ~depth_img();

    cv::Mat dImg;
    cv::Mat normals;
    cv::Mat cloud;
    cv::Mat coloredImg;
    pcl::PointCloud<pcl::Normal>::Ptr pcl_normals;
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud;

    int num_components;
    int target_component;

    void create_cloud();
    void compute_normals();
    void showColorImg() { cv::imshow(wName,coloredImg); cv::setMouseCallback(wName,cb_link,this); std::cout << "from SCI: " << this->dImg.size() << std::endl;}
    void showDImg() {
        double min,max;
        cv::minMaxIdx(dImg,&min,&max);
        cv::Mat color;
        dImg.convertTo(color,CV_8UC1,255 / (max-min), -min);
        cv::applyColorMap(color,color,cv::COLORMAP_AUTUMN);
        cv::imshow("wName",color);
    }
    void windowCallBack(int event, int x, int y, int flags, void* userdata);
    void test() {
        std::cout << "from test: " << dImg.size() << std::endl;
    }

    void applyFilter();
    void segment(float c,int min_size);
    void segment_distance(float c,int min_size,float dist_th);
    universe *u;
    char* wName;
    float *camera_params;
    edge* edges;
};
void cb_link(int event, int x, int y, int flags, void* userdata) {
    if(event != cv::EVENT_LBUTTONDOWN)
        return;
    depth_img *obj = (depth_img*)userdata;
    std::cout << "from the linker: " << obj->dImg.size() << std::endl;
    obj->windowCallBack(event,x,y,flags,NULL);
}
void depth_img::windowCallBack(int event, int x, int y, int flags, void *userdata) {
    if(event != cv::EVENT_LBUTTONDOWN)
        return;
    this->test();
    int component = u->find(y*dImg.cols + x);
    target_component = component;

    std::cout << "(x,y)" << x << " " << y << " cols: " << dImg.cols << " rows: " << dImg.rows << std::endl;
    std::cout << "component number: " << component << std::endl;
    std::cout << "component size: " << u->size(component) <<  std::endl;
    cv::Vec3b preto(255,255,255);
    cv::Mat coloredImg2;
    coloredImg2.create(480,640,CV_8UC3);
    for(int y = 0; y < 480;y++) {
        for(int x = 0; x < 640; x++) {
            if(u->find(y*640 + x) == component) {
                coloredImg2.at<cv::Vec3b>(y,x) = preto;
            }
        }
    }
    cv::imshow(wName,coloredImg2);
    //this->showColorImg();
}

void getComponentMask(depth_img seg_img,int component,cv::Mat &output_mask) {
    output_mask.create(seg_img.dImg.size(),CV_8UC1);
    output_mask = cv::Mat::ones(seg_img.dImg.size(),CV_8UC1);
    output_mask = output_mask * 0;
    int height = seg_img.dImg.rows;
    int width = seg_img.dImg.cols;

    std::cout << "height " << height << "width " << width << std::endl;
    std::cout << component << std::endl;

    for(int y = 0; y < height;y++) {
        for(int x = 0; x < width;x++) {
            int nodeComponent = seg_img.u->find(y*width + x);
            if (nodeComponent == component) {
                output_mask.at<uchar>(y,x) = 255;

            }
        }
    }
    cv::Mat distImg;
    distImg.create(seg_img.dImg.size(),CV_32FC1);
    cv::distanceTransform(output_mask,distImg,CV_DIST_L2,5);
    cv::GaussianBlur(distImg,distImg,cv::Size(21,21),0,0);
    cv::Laplacian(distImg,distImg,CV_32FC1,3,1,0);
    double minL,maxL,min,max;
    cv::minMaxLoc(distImg,&minL,&maxL);
    cv::Mat coloredImg;

    distImg = distImg/minL;
    cv::minMaxIdx(distImg,&min,&max);

    distImg.convertTo(coloredImg,CV_8UC1,255 / (max-min), -min);
    cv::threshold(coloredImg,coloredImg,20,255,CV_8UC1);
    //cv::applyColorMap(coloredImg,coloredImg,cv::COLORMAP_AUTUMN);
    cv::imshow("distImg",coloredImg);


}

void depth_img::applyFilter() {
    cv::Mat dst;
    dst.create(dImg.size(),CV_32FC1);
    for(int i = 0; i < dImg.rows; i++) {
        for(int j = 0; j < dImg.cols; j++) {
            dImg.at<float>(i,j) = ((isnan(dImg.at<float>(i,j))) ? 0:dImg.at<float>(i,j));
        }
    }
    cv::bilateralFilter(dImg,dst,5,30.,80.);
    dImg = dst;
}


float diff(int x,int y, int x2,int y2, depth_img* obj) {
    float c1,c2;
    double normalizer;
    c1 = 1;c2 = 1;
    cv::Point3f n1,n2,p1,p2,pdiff;
    n1 = obj->normals.at<cv::Point3f>(y,x);
    n2 = obj->normals.at<cv::Point3f>(y2,x2);

    float dotNormals = n1.dot(n2);
    dotNormals = (dotNormals < 0) ? -dotNormals:dotNormals;
    p1 = obj->cloud.at<cv::Point3f>(y,x);
    p2 = obj->cloud.at<cv::Point3f>(y2,x2);
    pdiff = p2 - p1;
    float euclidianDistance = pdiff.dot(pdiff);
    if(isnan(euclidianDistance)){
        euclidianDistance = 1000000.; //set it to +inf
    }

    if(isnan(dotNormals)) { //if nan, drop the dotNormals term.
        c1 = 1;
        dotNormals = 100;

    }
    float normalTerm = c1*(1.-dotNormals);
    float EDTerm = c2 * euclidianDistance;
    //printf("Normal term: %lf \nEDTerm: %lf \n",normalTerm,EDTerm);
    return   (0*normalTerm + EDTerm);
}
float diff_by_curvature(int x,int y, int x2,int y2, depth_img* obj) {
    float k1,k2;
    float THRESHOLD_DISTANCE = 0.3;
    pcl::Normal n1,n2;
    cv::Point3f p1,p2,pdiff;
    n1 = obj->pcl_normals->at(x,y);
    n2 = obj->pcl_normals->at(x2,y2);

    p1 = obj->cloud.at<cv::Point3f>(y,x);
    p2 = obj->cloud.at<cv::Point3f>(y2,x2);
    pdiff = p2 - p1;
    float euclidianDistance = pdiff.dot(pdiff);
    if(euclidianDistance > THRESHOLD_DISTANCE)
        //return 10*euclidianDistance;
    return fabs(n1.curvature-n2.curvature);
}

depth_img::depth_img(cv::Mat d, float* cInfo) {
    dImg = d;
    camera_params = cInfo;
    num_components = 0;
    target_component = 0;
    wName = "Segmented";
    std::cout << "from the constructor: " << dImg.size() << std::endl;

}

depth_img::~depth_img() {

}

cv::Vec3b random_color() {
    int r = rand() % 255;
    int g = rand() % 255;
    int b = rand() % 255;
    return cv::Vec3b(b,g,r);
}

void depth_img::segment_distance(float c,int min_size,float dist_th) {
    int width = dImg.cols;
    int height = dImg.rows;
    edges = new edge[width*height*4];
    int num = 0;
    compute_normals();

    for (int y = 100; y < height; y++) { //loop nas linhas
        for (int x = 100; x < width; x++) { //loop nas colunas
            if(dImg.at<float>(y,x) > dist_th)
                continue;
            if (x < width-1) {
                if(dImg.at<float>(y,x+1) <= dist_th) {
                    edges[num].a = y * width + x;
                    edges[num].b = y * width + (x+1);
                    edges[num].w = diff_by_curvature(x, y, x+1, y,this);
                    //edges[num].w = diff(x, y, x+1, y,this);
                    num++;
                }
            }

            if (y < height-1) {
                if(dImg.at<float>(y+1,x) <= dist_th){
                    edges[num].a = y * width + x;
                    edges[num].b = (y+1) * width + x;
                    edges[num].w = diff_by_curvature(x, y, x, y+1,this);
                    num++;
                }
            }

            if ((x < width-1) && (y < height-1)) {
                if(dImg.at<float>(y+1,x+1) <= dist_th){
                    edges[num].a = y * width + x;
                    edges[num].b = (y+1) * width + (x+1);
                    edges[num].w = diff_by_curvature(x, y, x+1, y+1,this);
                    num++;
                }
            }

            if ((x < width-1) && (y > 0)) {
                if(dImg.at<float>(y-1,x+1) <= dist_th){
                    edges[num].a = y * width + x;
                    edges[num].b = (y-1) * width + (x+1);
                    edges[num].w = diff_by_curvature(x, y, x+1, y-1,this);
                    num++;
                }
            }
        }
    }
    u = segment_graph(width * height,num,edges,c);

    for (int i = 0; i < num; i++) {
        int a = u->find(edges[i].a);
        int b = u->find(edges[i].b);
        if ((a != b) && ((u->size(a) < min_size) || (u->size(b) < min_size)))
            u->join(a, b);
    }

    this->num_components = u->num_sets();
    cv::Vec3b colors[width*height];
    for(int i = 0; i < width * height; i++) {
        colors[i] = random_color();
    }
    coloredImg.create(dImg.size(), CV_8UC3);
    for(int y = 0; y < height;y++)
        for(int x = 0; x < width;x++) {
            int component = u->find(y*width + x);
            coloredImg.at<cv::Vec3b>(y,x) = colors[component];
        }
    this->showColorImg();
    std::cout << "From segment_distance: " << dImg.size() <<" " << dImg.cols <<" " << dImg.rows << std::endl;
}

void depth_img::segment(float c,int min_size) {
    int width = dImg.cols;
    int height = dImg.rows;
    edges = new edge[width*height*4];
    int num = 0;
    compute_normals();

    for (int y = 0; y < height; y++) { //loop nas linhas
        for (int x = 0; x < width; x++) { //loop nas colunas
            if (x < width-1) {
                edges[num].a = y * width + x;
                edges[num].b = y * width + (x+1);
                edges[num].w = diff(x, y, x+1, y,this);
                num++;
            }

            if (y < height-1) {
                edges[num].a = y * width + x;
                edges[num].b = (y+1) * width + x;
                edges[num].w = diff(x, y, x, y+1,this);
                num++;
            }

            if ((x < width-1) && (y < height-1)) {
                edges[num].a = y * width + x;
                edges[num].b = (y+1) * width + (x+1);
                edges[num].w = diff(x, y, x+1, y+1,this);
                num++;
            }

            if ((x < width-1) && (y > 0)) {
                edges[num].a = y * width + x;
                edges[num].b = (y-1) * width + (x+1);
                edges[num].w = diff(x, y, x+1, y-1,this);
                num++;
            }
        }
    }
    u = segment_graph(width * height,num,edges,c);

    for (int i = 0; i < num; i++) {
        int a = u->find(edges[i].a);
        int b = u->find(edges[i].b);
        if ((a != b) && ((u->size(a) < min_size) || (u->size(b) < min_size)))
            u->join(a, b);
    }

    this->num_components = u->num_sets();
    cv::Vec3b colors[width*height];
    for(int i = 0; i < width * height; i++) {
        colors[i] = random_color();
    }
    coloredImg.create(dImg.size(), CV_8UC3);
    for(int y = 0; y < height;y++)
        for(int x = 0; x < width;x++) {
            int component = u->find(y*width + x);
            coloredImg.at<cv::Vec3b>(y,x) = colors[component];
        }
    this->showColorImg();

}

void depth_img::create_cloud()
{

    /* Projection/camera matrix
#     [fx'  0  cx' Tx]
# P = [ 0  fy' cy' Ty]
#     [ 0   0   1   0]
      */
    float fx = camera_params[0];
    float fy = camera_params[5];
    float cx = camera_params[2];
    float cy = camera_params[6];

    const float inv_fx = 1.f/fx;
    const float inv_fy = 1.f/fy;

    cloud.create( dImg.size(), CV_32FC3 );
    std::cout << fx << ' ' << fy << ' ' << cx << ' ' << ' ' << cy << std::endl;
    for( int y = 0; y < cloud.rows; y++ )
    {
        cv::Point3f* cloud_ptr = (cv::Point3f*)cloud.ptr(y);

        //const float* depth_ptr = (float*)img.ptr(y);
        //const ufloat32_t* depth_pt = (ufloat16_t*)img.ptr(y);
        for( int x = 0; x < cloud.cols; x++ )
        {

            float d = dImg.at<float>(y,x);


            cloud_ptr[x].x = (x - cx) * d * inv_fx;
            cloud_ptr[x].y = (y - cy) * d * inv_fy;
            cloud_ptr[x].z = d;

        }
    }
}

void depth_img::compute_normals()
{
    this->create_cloud();

    pcl_cloud.reset(new pcl::PointCloud<pcl::PointXYZ> );

    pcl_cloud->clear();
    pcl_cloud->width     = dImg.cols;
    pcl_cloud->height    = dImg.rows;
    pcl_cloud->points.resize( pcl_cloud->width * pcl_cloud->height);

    for(int y = 0; y < dImg.rows; ++y)
        for(int x = 0; x < dImg.cols; ++x)
        {
            pcl_cloud->at(x,y).x = this->cloud.at<cv::Point3f>(y,x).x;
            pcl_cloud->at(x,y).y = this->cloud.at<cv::Point3f>(y,x).y;
            pcl_cloud->at(x,y).z = this->cloud.at<cv::Point3f>(y,x).z;
        }

    pcl_normals.reset(new pcl::PointCloud<pcl::Normal>);
    pcl_normals->clear();
    pcl_normals->width  = pcl_cloud->width;
    pcl_normals->height = pcl_cloud->height;
    pcl_normals->points.resize(pcl_cloud->width * pcl_cloud->height);

    pcl::IntegralImageNormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    //ne.setNormalEstimationMethod (ne.AVERAGE_3D_GRADIENT);
    ne.setMaxDepthChangeFactor(0.05f);
    ne.setNormalSmoothingSize(5.0f);
    ne.setKSearch(300);
    ne.setNormalEstimationMethod(ne.COVARIANCE_MATRIX);
    ne.setInputCloud(pcl_cloud);
    ne.compute(*pcl_normals);

    normals.create( dImg.size(), CV_32FC3 );

    for(int y = 0; y < pcl_normals->height; ++y)
        for(int x = 0; x < pcl_normals->width; ++x)
        {
            normals.at<cv::Point3f>(y,x).x = pcl_normals->at(x,y).normal_x;
            normals.at<cv::Point3f>(y,x).y = pcl_normals->at(x,y).normal_y;
            normals.at<cv::Point3f>(y,x).z = pcl_normals->at(x,y).normal_z;
        }

}

#endif
