#include "registrationModule.h"


registrationModule::registrationModule() {

}

void showDepthImg(cv::Mat img) {
    double min,max;
    cv::Mat coloredImg;
    min = 50;
    max = 0;
    std::cout << "comecnado..." << std::endl;
    for(int i = 0; i < img.rows; i++) {
        for(int j = 0; j < img.cols; j++) {
            float value = img.at<float>(i,j);
            //std::cout << value << " (" << i << "," << j << ")" << std::endl;
            if(isinf(value))
                continue;
            if(isnan(value))
                continue;
            if(value < min)
                min = value;
            if(value > max)
                max = value;
        }
    }
    std::cout << "min and max: " << min << " " << max << std::endl;
    img.convertTo(coloredImg,CV_8UC1,255 / (max-min), -min);
    cv::applyColorMap(coloredImg,coloredImg,cv::COLORMAP_AUTUMN);
    cv::Rect r(50,50,60,60);

    //std::cout << coloredImg(r) << std::endl;
    cv::imshow("wName",coloredImg);

}
void pclRangetoOpencv(const pcl::RangeImage pcl_range,cv::Mat &m) {
    m.create(pcl_range.height,pcl_range.width,CV_32FC1);
    for(int x = 0; x < pcl_range.width; x++) {
        for(int y = 0; y < pcl_range.height; y++) {
            m.at<float>(y,x) = pcl_range.at(x,y).range;
        }
    }
    std::cout << "msize: " << m.size() << std::endl;
    showDepthImg(m);
}


void registrationModule::getRangeImage() {
    // We now want to create a range image from the above point cloud, with a 1deg angular resolution
    float angularResolution = (float) (  0.092f * (M_PI/180.0f));  //   1.0 degree in radians
    float maxAngleWidth     = (float) (360.0f * (M_PI/180.0f));  // 360.0 degree in radians
    float maxAngleHeight    = (float) (180.0f * (M_PI/180.0f));  // 180.0 degree in radians
    Eigen::Affine3f sensorPose = (Eigen::Affine3f)Eigen::Translation3f(0.0f, 0.0f, 0.0f);
    pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::CAMERA_FRAME;
    float noiseLevel=0.00;
    float minRange = 0.0f;
    int borderSize = 0;

    rimg.createFromPointCloud(*registered_cloud, angularResolution, maxAngleWidth, maxAngleHeight,
                              sensorPose, coordinate_frame, noiseLevel, minRange, borderSize);

    cv::Mat m;
    pclRangetoOpencv(rimg,m);

    float support_size = 1.0f;
    rimg.setUnseenToMaxRange();
    pcl::RangeImageBorderExtractor range_image_border_extractor;
    pcl::NarfKeypoint narf_keypoint_detector;
    narf_keypoint_detector.setRangeImageBorderExtractor (&range_image_border_extractor);
    narf_keypoint_detector.setRangeImage (&rimg);
    narf_keypoint_detector.getParameters ().support_size = support_size;


    pcl::PointCloud<int> keypoint_indices;
    //narf_keypoint_detector.compute (keypoint_indices);
    std::cout << "Found "<<keypoint_indices.points.size ()<<" key points.\n";

    /*
    pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_ptr (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>& keypoints = *keypoints_ptr;
    keypoints.points.resize (keypoint_indices.points.size ());
    for (size_t i=0; i<keypoint_indices.points.size (); ++i)
        keypoints.points[i].getVector3fMap () = rimg.points[keypoint_indices.points[i]].getVector3fMap ();
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> keypoints_color_handler (keypoints_ptr, 0, 0, 255);
    p->addPointCloud<pcl::PointXYZ> (keypoints_ptr, keypoints_color_handler, "keypoints",vp_1);
    p->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "keypoints");
    p->spin();
    */
}

void registrationModule::addCloud(PointCloud::Ptr cloud) {
    PCD m;
    int sz = this->data.size();
    char s[5000];
    snprintf(s,5000,"%d",sz);
    m.f_name = s;
    pcl::copyPointCloud(*cloud,*m.cloud);
    std::vector<int> idxs;
    pcl::removeNaNFromPointCloud(*m.cloud,*m.cloud,idxs);
    data.push_back(m);
}

////////////////////////////////////////////////////////////////////////////////
/** \brief Display source and target on the first viewport of the visualizer
 *
 */
void registrationModule::showCloudsLeft(const PointCloud::Ptr cloud_target, const PointCloud::Ptr cloud_source)
{
    p->removePointCloud ("vp1_target");
    p->removePointCloud ("vp1_source");

    PointCloudColorHandlerCustom<PointT> tgt_h (cloud_target, 0, 255, 0);
    PointCloudColorHandlerCustom<PointT> src_h (cloud_source, 100, 100, 100);
    p->addPointCloud (cloud_target, tgt_h, "vp1_target", vp_1);
    //p->addPointCloud (cloud_source, src_h, "vp1_source", vp_1);

    PCL_INFO ("Press q to begin the registration.\n");
    p-> spin();
}


////////////////////////////////////////////////////////////////////////////////
/** \brief Display source and target on the second viewport of the visualizer
 *
 */
void registrationModule::showCloudsRight(const PointCloudWithNormals::Ptr cloud_target, const PointCloudWithNormals::Ptr cloud_source)
{
    p->removePointCloud ("source");
    p->removePointCloud ("target");


    PointCloudColorHandlerGenericField<PointNormalT> tgt_color_handler (cloud_target, "curvature");
    if (!tgt_color_handler.isCapable ())
        PCL_WARN ("Cannot create curvature color handler!");

    PointCloudColorHandlerGenericField<PointNormalT> src_color_handler (cloud_source, "curvature");
    if (!src_color_handler.isCapable ())
        PCL_WARN ("Cannot create curvature color handler!");


    p->addPointCloud (cloud_target, tgt_color_handler, "target", vp_2);
    p->addPointCloud (cloud_source, src_color_handler, "source", vp_2);

    p->spinOnce();
}
void registrationModule::prepareToAlign() {
    if(data.empty()) {
        PCL_WARN("Data is empty!\n");
        return;
    }
    // Create a PCLVisualizer object
    p = new pcl::visualization::PCLVisualizer ("Pairwise Incremental Registration example");
    p->createViewPort (0.0, 0, 0.5, 1.0, vp_1);
    p->createViewPort (0.5, 0, 1.0, 1.0, vp_2);
    p->setBackgroundColor(100,100,100,vp_1);
    p->setBackgroundColor(200,200,200,vp_2);
    PointCloud::Ptr result (new PointCloud);
    Eigen::Matrix4f GlobalTransform = Eigen::Matrix4f::Identity ();
    for(int i = 1; i < data.size(); i++) {
        pcd_src = data[i-1].cloud;
        pcd_tgt = data[i].cloud;
        //showCloudsLeft(pcd_src, pcd_tgt);
        registered_cloud.reset(new PointCloud);
        PCL_INFO("Begining Alignment(%d,%d)...\n",i,i-1);
        pairAlign(1); // /param 1 = yes to downsampling.
        pcl::transformPointCloud (*registered_cloud, *result, GlobalTransform);

        //update the global transform
        GlobalTransform = GlobalTransform * this->final_transform;
    }
    pcl::io::savePCDFileASCII("registered_cloud.pcd",*registered_cloud);
    showCloudsLeft(registered_cloud,result);
    getRangeImage();
}

void registrationModule::pairAlign (bool downsample)
{
    //
    // Downsample for consistency and speed
    // \note enable this for large datasets
    const PointCloud::Ptr cloud_src = pcd_src;
    const PointCloud::Ptr cloud_tgt = pcd_tgt;
    PointCloud::Ptr output = registered_cloud;
    PointCloud::Ptr src (new PointCloud);
    PointCloud::Ptr tgt (new PointCloud);
    pcl::VoxelGrid<PointT> grid;
    if (downsample)
    {
        grid.setLeafSize (0.05, 0.05, 0.05);
        grid.setInputCloud (cloud_src);
        grid.filter (*src);

        grid.setInputCloud (cloud_tgt);
        grid.filter (*tgt);
    }
    else
    {
        src = cloud_src;
        tgt = cloud_tgt;
    }


    // Compute surface normals and curvature
    PointCloudWithNormals::Ptr points_with_normals_src (new PointCloudWithNormals);
    PointCloudWithNormals::Ptr points_with_normals_tgt (new PointCloudWithNormals);

    pcl::NormalEstimation<PointT, PointNormalT> norm_est;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
    norm_est.setSearchMethod (tree);

    //I don't know why, but the last elements are still NaN, even after removeNaNFromPointCloud methos was called, so we're simply taking it off by hands...
    src->erase(src->end()-1); tgt->erase(tgt->end()-1);


    //Uncomment this if the clouds are not huge... (also set downscale = 0).
    //
    /*    pcl::IntegralImageNormalEstimation<PointT, PointNormalT> norm_est;
    norm_est.setNormalEstimationMethod(norm_est.COVARIANCE_MATRIX);
    norm_est.setMaxDepthChangeFactor(0.2f);
    */
    norm_est.setKSearch (30);

    norm_est.setInputCloud (src);
    norm_est.compute (*points_with_normals_src);
    pcl::copyPointCloud (*src, *points_with_normals_src);
    std::vector<int> idxsrc;
    pcl::removeNaNNormalsFromPointCloud(*points_with_normals_src,*points_with_normals_src,idxsrc);

    norm_est.setInputCloud (tgt);
    norm_est.compute (*points_with_normals_tgt);
    pcl::copyPointCloud (*tgt, *points_with_normals_tgt);
    std::vector<int> idxtgt;
    pcl::removeNaNNormalsFromPointCloud(*points_with_normals_tgt,*points_with_normals_tgt,idxtgt);

    //
    // Instantiate our custom point representation (defined above) ...
    MyPointRepresentation point_representation;
    // ... and weight the 'curvature' dimension so that it is balanced against x, y, and z
    float alpha[4] = {1.0, 1.0, 1.0, 1.0};
    point_representation.setRescaleValues (alpha);

    //
    // Align
    pcl::IterativeClosestPointNonLinear<PointNormalT, PointNormalT> reg;

    reg.setTransformationEpsilon (1e-6);
    // Set the maximum distance between two correspondences (src<->tgt) to 10cm
    // Note: adjust this based on the size of your datasets
    reg.setMaxCorrespondenceDistance(0.05);
    // Set the point representation
    reg.setPointRepresentation (boost::make_shared<const MyPointRepresentation> (point_representation));

    reg.setInputSource (points_with_normals_src);
    reg.setInputTarget (points_with_normals_tgt);

    //
    // Run the same optimization in a loop and visualize the results
    Eigen::Matrix4f Ti = Eigen::Matrix4f::Identity (), prev, targetToSource;
    PointCloudWithNormals::Ptr reg_result = points_with_normals_src;
    reg.setMaximumIterations(200);
    reg.align (*reg_result);

    Ti = reg.getFinalTransformation ();

    // visualize current state
    //showCloudsRight(points_with_normals_tgt, points_with_normals_src);

    //
    // Get the transformation from target to source
    targetToSource = Ti.inverse();
    //
    // Transform target back in source frame

    pcl::transformPointCloud (*cloud_tgt, *output, targetToSource);

    //add the source to the transformed target
    *output += *cloud_src;
    final_transform = targetToSource;

}
