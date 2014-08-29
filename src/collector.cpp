#include "collector.h"
#include <ios>
#include <cmath>
#include <cstdio>
#define DISTANCE_THRESHOLD 0.25


void kvd_descriptor::build_train_test(float train_percentage,cv::Mat &train,cv::Mat &train_targets,cv::Mat &test,cv::Mat &test_targets) {
    int train_Psize = (int)(kvd_pos.rows * train_percentage);
    int train_Nsize = (int)(kvd_neg.rows * train_percentage);
    kvd_pos.convertTo(kvd_pos,CV_32FC1);
    kvd_neg.convertTo(kvd_neg,CV_32FC1);
    train = kvd_pos.rowRange(0,train_Psize);
    train.push_back(kvd_neg.rowRange(0,train_Nsize));

    test = kvd_pos.rowRange(train_Psize+1,kvd_pos.rows);
    int test_pos = test.rows;
    test.push_back(kvd_neg.rowRange(train_Nsize+1,kvd_neg.rows));

    train_targets = cv::Mat::zeros(train_Psize + train_Nsize,1,CV_32FC1);
    train_targets.rowRange(0,train_Psize) += 1;

    test_targets = cv::Mat::zeros(test.rows,1,CV_32FC1);
    test_targets.rowRange(0,test_pos) += 1;

}

kvd_descriptor::kvd_descriptor(char* filename = NULL){

    for(int i =0; i < 15;i++) {
        cv::Mat a,b,c,d,e,f;
        this->posDotNormals.push_back(a);
        this->negDotNormals.push_back(b);
        this->posCurvatures.push_back(c);
        this->negCurvatures.push_back(d);
        this->pos_ED.push_back(e);
        this->neg_ED.push_back(f);
    }
    if(filename)
        this->readFromFile(filename);
}

kvd_descriptor::~kvd_descriptor() {

}
void kvd_descriptor::visualize(int size, int i) {
    cv::Mat positive,negative;
    cv::Mat pDesc,nDesc;
    if(i > 0) {
        pDesc = pos_descriptors.at(i);
        nDesc = neg_descriptors.at(i);
    }
    else {
        pDesc = kvd_pos;
        nDesc = kvd_neg;
    }
    cv::Size s(kvd_pos.cols,size);
    positive.create(s, CV_8UC3);
    negative.create(s, CV_8UC3);
    cv::Vec3b one(255,0,0); cv::Vec3b two(0,255,0);
    cv::Vec3b three(0,0,255); cv::Vec3b four(0,0,0);
    cv::Vec3b five(255,255,255);
    std::vector<cv::Vec3b> colors;
    colors.push_back(one); colors.push_back(two); colors.push_back(three); colors.push_back(four); colors.push_back(five);
    std::cout << "checking: " << pDesc.rows << ',' << pDesc.cols << std::endl;
    srand(time(NULL));
    for(int z = 0; z < size; z++) {
        int y = rand() % kvd_pos.rows;
        for(int x = 0; x < negative.cols; x++) {
            int p_cIdx = (int)pDesc.at<uchar>(y,x); int n_cIdx = (int)nDesc.at<uchar>(y,x);
            //std::cout << p_cIdx << ',' << n_cIdx << std::endl;
            positive.at<cv::Vec3b>(z,x) = colors[p_cIdx];
            negative.at<cv::Vec3b>(z,x) = colors[n_cIdx];
        }
    }
    cv::imshow("positive",positive);
    cv::imshow("negative",negative);
    cv::waitKey();

}

void kvd_descriptor::compute_descriptors(float treshold,float ed_threshold,int pos0,int posF) {
    if(pos0 < 0) {
        pos0 = 0;
        posF = 15;
    }
    for(int i = pos0; i < posF;i++) {
        cv::Mat p = compute(treshold,ed_threshold,posDotNormals.at(i),posCurvatures.at(i),pos_ED.at(i));
        cv::Mat n = compute(treshold,ed_threshold,negDotNormals.at(i),negCurvatures.at(i),neg_ED.at(i));
        pos_descriptors.push_back(p);
        neg_descriptors.push_back(n);
    }
    cv::hconcat(pos_descriptors,kvd_pos);
    cv::hconcat(neg_descriptors,kvd_neg);
    /*if(pos0 < 0) {
        cv::hconcat(pos_descriptors,kvd_pos);
        cv::hconcat(neg_descriptors,kvd_neg);
    }
    else {
        kvd_pos = pos_descriptors.at(pos);
        kvd_neg = neg_descriptors.at(pos);
    }*/
}

cv::Mat kvd_descriptor::compute(float treshold,float dist_treshold,cv::Mat N,cv::Mat K,cv::Mat ED) {
    cv::Mat kvdDescriptor;
    kvdDescriptor.create(N.size(), CV_8U);
    float ED_distance = dist_treshold;
    for(int y = 0; y < N.rows;y++)
        for(int x = 0; x < N.cols;x++) {
            if(ED.at<float>(y,x) > ED_distance) {
                kvdDescriptor.at<uchar>(y,x) = 3;
                //kvdDescriptor.at<uchar>(y,x) = N.at<float>(y,x);
                continue;
            }
            if(isnan(N.at<float>(y,x))) { //mark as a non-existent information to be taken off when in learning phase
                kvdDescriptor.at<uchar>(y,x) = 4;
                continue;
            }

            if(N.at<float>(y,x) > treshold) {
                kvdDescriptor.at<uchar>(y,x) = 0;
                continue;
            }
            kvdDescriptor.at<uchar>(y,x) = ((K.at<float>(y,x) > 0) ? 1:2);
        }
    return kvdDescriptor;
}

void kvd_descriptor::readFromFile(char* filename) {
    cv::FileStorage fs;
    fs.open(filename,cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        std::cout << "Failed to open " << filename << std::endl;
        return ;
    }
    for(int i = 0 ; i < 15 ;i++) {
        std::stringstream pNormals,nNormals,pK,nK,posED,negED;
        pNormals << "positiveDotNormalsR" << (i+1);
        nNormals << "negativeDotNormalsR" << (i+1);
        pK << "K_positivePointsR" << (i+1);
        nK << "K_negativePointsR" << (i+1);
        posED << "ED_positivePointsR" << (i+1);
        negED << "ED_negativePointsR" << (i+1);
        cv::Mat pN,nN,pK_,nK_,pED,nED;
        fs[pNormals.str()] >> pN;
        fs[nNormals.str()] >> nN;
        fs[pK.str()] >> pK_;
        fs[nK.str()] >> nK_;
        fs[posED.str()] >> pED;
        fs[negED.str()] >> nED;
        this->posDotNormals.at(i).push_back(pN);
        this->negDotNormals.at(i).push_back(nN);
        this->posCurvatures.at(i).push_back(pK_);
        this->negCurvatures.at(i).push_back(nK_);
        this->pos_ED.at(i).push_back(pED);
        this->neg_ED.at(i).push_back(nED);
    }
}







/******************************************************* UTILS *********************************************************/



//calculates the standart deviation of a float column vector in respect of its median (not mean).
void calculate_std_using_median(cv::Mat source,float & median,float& std) {
    cv::Mat ordered;
    cv::sort(source,ordered,CV_SORT_EVERY_COLUMN);
    median = ordered.at<float>(floor(ordered.rows/2));
    std = 0.;
    for(int i = 0; i < ordered.rows; i++) {
        float value = ordered.at<float>(i);
        if(isnan(value)) {
            printf("In function: calculate_std_using_median(cv::Mat src) -> the cv::Mat received contains Nan, fix it.\n");
            return;
        }
        std += (value - median)*(value - median);
    }
    std /= (float)ordered.rows;
}


void normalMean_median(cv::Rect roi,cv::Mat normals,cv::Mat depth, cv::Point3f &result,cv::Mat &coloredDepth) {
    cv::Mat temp = depth(roi);
    std::vector<float> v;
    for(int i = 0; i < temp.rows; i++)
        for(int j = 0; j < temp.cols; j++) {
            float f = temp.at<float>(i,j);
            if(!isnan(f))
                v.push_back(f);
        }
    cv::Mat lineVect(v);
    float std,median;
    calculate_std_using_median(lineVect,median,std);
    int counter = 0;
    for(int i = 0; i < temp.rows; i++)
        for(int j = 0; j < temp.cols; j++) {
            float dst = temp.at<float>(i,j);
            if(pow((dst - median),2) <= 1.*std) {
                coloredDepth.at<cv::Vec3b>(i + roi.y,j + roi.x) = cv::Vec3b(255,0,0);
                if(!isnan(normals.at<cv::Point3f>(i+roi.y,j +roi.x).x)) {
                    coloredDepth.at<cv::Vec3b>(i + roi.y,j + roi.x) = cv::Vec3b(0,255,0);
                    result.x += normals.at<cv::Point3f>(i + roi.y,j + roi.x).x;
                    result.y += normals.at<cv::Point3f>(i + roi.y,j + roi.x).y;
                    result.z += normals.at<cv::Point3f>(i + roi.y,j + roi.x).z;
                    counter++;
                }
            }
        }
    if(counter == 0)
        return;
    result.x /= counter;
    result.y /= counter;
    result.z /= counter;
}

void entropy_calc(cv::Mat src, float* probs, float & ent, bool & valid) {
    ent = 0.;
    valid = 1;
    for(int i = 0; i < src.cols; i++) {
        int idx = (int)src.at<float>(0,i);
        if(idx == 4) {
            valid = 0;
            break;
        }
        ent += probs[idx] * log2(probs[idx]);
    }
    ent *= -1;
}
void entropy_calc2(float* src,int size, float* probs, float & ent, bool & valid) {
    ent = 0.;
    valid = 1;
    for(int i = 0; i < size; i++) {
        int idx = (int)src[i];
        if(idx == 4) {
            valid = 0;
            break;
        }
        ent += probs[idx] * log2(probs[idx]);
    }
    ent *= -1;
}














/****************************************************   CLASS COLLECTOR      ********************************************************************************/


void collector::extract_descriptor_information_from_region(int positive) {
    if(roi_queue.empty()) {
        return;
    }
    for(int k = 0; k < roi_queue.size(); k++) { //for each region of interest k...
        cv::Rect roi = roi_queue[k];
        //we need to clear all those nan's
        cv::Mat temp = img(roi);
        std::vector<float> v;
        for(int i = 0; i < temp.rows; i++)
            for(int j = 0; j < temp.cols; j++) {
                float f = temp.at<float>(i,j);
                if(!isnan(f))
                    v.push_back(f);
            }
        cv::Mat lineVect(v);
        float std,median;
        calculate_std_using_median(lineVect,median,std);

        for(int i = roi.y; i < roi.y + roi.height; i++) //loop over the IMAGE rows
            for(int j = roi.x; j < roi.x + roi.width; j++) { //loop over the IMAGE collumns
                float value = img.at<float>(i,j);
                float std_value = pow(value - median,2);
                if(std_value <= 1.*std) { //ensure that the point is not an outlier:
                    if (positive)
                        this->positivePoints.push_back(cv::Point(j,i));

                    else
                        this->negativePoints.push_back(cv::Point(j,i));
                }
            }
    }
}


void collector::getNormalMeans(int vis) {
    if (roi_queue.empty())
        return;

    double min,max;
    cv::minMaxIdx(img,&min,&max);
    cv::Mat coloredDepth;
    coloredDepth.create(img.size(), CV_8UC1);

    img.convertTo(coloredDepth,CV_8UC1,255 / (max-min), -min);
    cv::applyColorMap(coloredDepth,coloredDepth,cv::COLORMAP_AUTUMN);
    cv::Rect roi;
    //this->compute_normals();

    cv::Point3f temp_normal;
    std::cout << "number of rois: " << roi_queue.size() <<std::endl;
    cv::Point3f normalVectors[roi_queue.size()];
    for(int i = 0; i < roi_queue.size(); i++) {
        temp_normal.x = 0; temp_normal.y = 0; temp_normal.z = 0;
        normalMean_median(roi_queue[i],this->normals,img,temp_normal,coloredDepth);
        normalVectors[i] = temp_normal;
        std::cout << temp_normal << std::endl;
        normals_queue.push_back(temp_normal);
    }
    //Visualization
    if(vis){
        showcloud(normalVectors,roi_queue.size());
        cv::imshow("affected regions",coloredDepth);

    }
}




int collector::convertIdx(int x,int y) {
    return (x*(this->K_matrix.cols)) + y;
}
int collector::inverseIdxX(int x) {
    return  ((int)(x/this->K_matrix.cols)) ;
}
int collector::inverseIdxY(int x) {
    return (int)(x % this->K_matrix.cols);
}

void cloudCoordinatesTest(int x,int y,int xC,int yC,std::vector<pcl::PointXYZRGB> *list,collector* obj) {
    pcl::PointXYZRGB p1,p2,p3,p4,p5,p6,p7,p8;
    cv::Point3f pt;
    pt = obj->cloud.at<cv::Point3f>(xC +x,yC + y);
    p1.x = pt.x; p1.y = pt.y; p1.z = pt.z;

    pt = obj->cloud.at<cv::Point3f>(xC + x,yC - y);
    p2.x = pt.x; p2.y = pt.y; p2.z = pt.z;

    pt = obj->cloud.at<cv::Point3f>(xC - x,yC + y);
    p3.x = pt.x; p3.y = pt.y; p3.z = pt.z;

    pt = obj->cloud.at<cv::Point3f>(xC -x,yC - y);
    p4.x = pt.x; p4.y = pt.y; p4.z = pt.z;

    pt = obj->cloud.at<cv::Point3f>(xC + y,yC + x);
    p5.x = pt.x; p5.y = pt.y; p5.z = pt.z;

    pt = obj->cloud.at<cv::Point3f>(xC - y,yC + x);
    p6.x = pt.x; p6.y = pt.y; p6.z = pt.z;

    pt = obj->cloud.at<cv::Point3f>(xC + y,yC - x);
    p7.x = pt.x; p7.y = pt.y; p7.z = pt.z;

    pt = obj->cloud.at<cv::Point3f>(xC - y,yC - x);
    p8.x = pt.x; p8.y = pt.y; p8.z = pt.z;

    list->push_back(p1); list->push_back(p2); list->push_back(p3); list->push_back(p4);
    list->push_back(p5); list->push_back(p6); list->push_back(p7); list->push_back(p8);
}

void computeNormalDots(int x,int y,int xC,int yC,std::vector<float> *list,collector* obj)
{
    list->push_back(obj->normals.at<cv::Point3f>(xC,yC).dot(obj->normals.at<cv::Point3f>(xC + x,yC + y)));
    list->push_back(obj->normals.at<cv::Point3f>(xC,yC).dot(obj->normals.at<cv::Point3f>(xC + x,yC - y)));
    list->push_back(obj->normals.at<cv::Point3f>(xC,yC).dot(obj->normals.at<cv::Point3f>(xC - x,yC + y)));
    list->push_back(obj->normals.at<cv::Point3f>(xC,yC).dot(obj->normals.at<cv::Point3f>(xC - x,yC - y)));
    list->push_back(obj->normals.at<cv::Point3f>(xC,yC).dot(obj->normals.at<cv::Point3f>(xC + y,yC + x)));
    list->push_back(obj->normals.at<cv::Point3f>(xC,yC).dot(obj->normals.at<cv::Point3f>(xC - y,yC + x)));
    list->push_back(obj->normals.at<cv::Point3f>(xC,yC).dot(obj->normals.at<cv::Point3f>(xC + y,yC - x)));
    list->push_back(obj->normals.at<cv::Point3f>(xC,yC).dot(obj->normals.at<cv::Point3f>(xC - y,yC - x)));

}

void computeK(int x,int y,int xC,int yC,std::vector<float> *list,collector* obj)
{
    cv::Point3f *center_normal = &obj->normals.at<cv::Point3f>(xC,yC);
    cv::Point3f *center_pos = &obj->cloud.at<cv::Point3f>(xC,yC);
    list->push_back((*center_normal - obj->normals.at<cv::Point3f>(xC + x,yC + y)).dot(*center_pos - obj->cloud.at<cv::Point3f>(xC + x,yC + y)));
    list->push_back((*center_normal - obj->normals.at<cv::Point3f>(xC + x,yC - y)).dot(*center_pos - obj->cloud.at<cv::Point3f>(xC + x,yC - y)));
    list->push_back((*center_normal - obj->normals.at<cv::Point3f>(xC - x,yC + y)).dot(*center_pos - obj->cloud.at<cv::Point3f>(xC - x,yC + y)));
    list->push_back((*center_normal - obj->normals.at<cv::Point3f>(xC - x,yC - y)).dot(*center_pos - obj->cloud.at<cv::Point3f>(xC - x,yC - y)));
    list->push_back((*center_normal - obj->normals.at<cv::Point3f>(xC + y,yC + x)).dot(*center_pos - obj->cloud.at<cv::Point3f>(xC + y,yC + x)));
    list->push_back((*center_normal - obj->normals.at<cv::Point3f>(xC - y,yC + x)).dot(*center_pos - obj->cloud.at<cv::Point3f>(xC - y,yC + x)));
    list->push_back((*center_normal - obj->normals.at<cv::Point3f>(xC + y,yC - x)).dot(*center_pos - obj->cloud.at<cv::Point3f>(xC + y,yC - x)));
    list->push_back((*center_normal - obj->normals.at<cv::Point3f>(xC - y,yC - x)).dot(*center_pos - obj->cloud.at<cv::Point3f>(xC - y,yC - x)));
}
void computeED(int x,int y,int xC,int yC,std::vector<float> *list,collector* obj)
{
    cv::Point3f *center_pos = &obj->cloud.at<cv::Point3f>(xC,yC);
    list->push_back((*center_pos - obj->cloud.at<cv::Point3f>(xC + x,yC + y)).dot(*center_pos - obj->cloud.at<cv::Point3f>(xC + x,yC + y)));
    list->push_back((*center_pos - obj->cloud.at<cv::Point3f>(xC + x,yC - y)).dot(*center_pos - obj->cloud.at<cv::Point3f>(xC + x,yC - y)));
    list->push_back((*center_pos - obj->cloud.at<cv::Point3f>(xC - x,yC + y)).dot(*center_pos - obj->cloud.at<cv::Point3f>(xC - x,yC + y)));
    list->push_back((*center_pos - obj->cloud.at<cv::Point3f>(xC - x,yC - y)).dot(*center_pos - obj->cloud.at<cv::Point3f>(xC - x,yC - y)));
    list->push_back((*center_pos - obj->cloud.at<cv::Point3f>(xC + y,yC + x)).dot(*center_pos - obj->cloud.at<cv::Point3f>(xC + y,yC + x)));
    list->push_back((*center_pos - obj->cloud.at<cv::Point3f>(xC - y,yC + x)).dot(*center_pos - obj->cloud.at<cv::Point3f>(xC - y,yC + x)));
    list->push_back((*center_pos - obj->cloud.at<cv::Point3f>(xC + y,yC - x)).dot(*center_pos - obj->cloud.at<cv::Point3f>(xC + y,yC - x)));
    list->push_back((*center_pos - obj->cloud.at<cv::Point3f>(xC - y,yC - x)).dot(*center_pos - obj->cloud.at<cv::Point3f>(xC - y,yC - x)));
}
std::vector<pcl::PointXYZRGB>* collector::bresenhamApplyTemp(int xC,int yC,int Radius,void (*f)(int,int,int,int,std::vector<pcl::PointXYZRGB>*,collector*)) {
    int P;
    int x,y;
    std::vector<pcl::PointXYZRGB> *dotProducts = new std::vector<pcl::PointXYZRGB>;
    P = 1 - Radius;
    x = 0;
    y = Radius;
    f(x,y,xC,yC,dotProducts,this);
    while (x<=y)
    {
        x++;
        if (P<0)
        {
            P += 2 * x + 1;
        }
        else
        {
            P += 2 * (x - y) + 1;
            y--;
        }
        f(x,y,xC,yC,dotProducts,this);
    }
    return dotProducts;
}
std::vector<float>* collector::bresenhamApply(int xC,int yC,int Radius,void (*f)(int,int,int,int,std::vector<float>*,collector*)) {
    int P;
    int x,y;
    std::vector<float> *dotProducts = new std::vector<float>;
    P = 1 - Radius;
    x = 0;
    y = Radius;
    f(x,y,xC,yC,dotProducts,this);
    while (x<=y)
    {
        x++;
        if (P<0)
        {
            P += 2 * x + 1;
        }
        else
        {
            P += 2 * (x - y) + 1;
            y--;
        }
        f(x,y,xC,yC,dotProducts,this);
    }
    return dotProducts;
}

float computeFeature(int x,int y,int xC,int yC, float* feat,int& idx, collector*obj) {
    float normals[8];
    float curvatures[8];
    float distance[8];
    const float treshold = 0.99;
    float r = 0;
    //computa dot product dos pontos no circulo com o ponto central.
    normals[0] = obj->normals.at<cv::Point3f>(xC,yC).dot(obj->normals.at<cv::Point3f>(xC + x,yC + y));
    normals[1] = obj->normals.at<cv::Point3f>(xC,yC).dot(obj->normals.at<cv::Point3f>(xC + x,yC - y));
    normals[2] = obj->normals.at<cv::Point3f>(xC,yC).dot(obj->normals.at<cv::Point3f>(xC - x,yC + y));
    normals[3] = obj->normals.at<cv::Point3f>(xC,yC).dot(obj->normals.at<cv::Point3f>(xC - x,yC - y));
    normals[4] = obj->normals.at<cv::Point3f>(xC,yC).dot(obj->normals.at<cv::Point3f>(xC + y,yC + x));
    normals[5] = obj->normals.at<cv::Point3f>(xC,yC).dot(obj->normals.at<cv::Point3f>(xC - y,yC + x));
    normals[6] = obj->normals.at<cv::Point3f>(xC,yC).dot(obj->normals.at<cv::Point3f>(xC + y,yC - x));
    normals[7] = obj->normals.at<cv::Point3f>(xC,yC).dot(obj->normals.at<cv::Point3f>(xC - y,yC - x));
    cv::Point3f center_normal = obj->normals.at<cv::Point3f>(xC,yC);
    cv::Point3f center_pos = obj->cloud.at<cv::Point3f>(xC,yC);
    //computa curvatura dos pontos do circulo com o ponto central.
    curvatures[0] = (center_normal - obj->normals.at<cv::Point3f>(xC + x,yC + y)).dot(center_pos - obj->cloud.at<cv::Point3f>(xC + x,yC + y));
    curvatures[1] = (center_normal - obj->normals.at<cv::Point3f>(xC + x,yC - y)).dot(center_pos - obj->cloud.at<cv::Point3f>(xC + x,yC - y));
    curvatures[2] = (center_normal - obj->normals.at<cv::Point3f>(xC - x,yC + y)).dot(center_pos - obj->cloud.at<cv::Point3f>(xC - x,yC + y));
    curvatures[3] = (center_normal - obj->normals.at<cv::Point3f>(xC - x,yC - y)).dot(center_pos - obj->cloud.at<cv::Point3f>(xC - x,yC - y));
    curvatures[4] = (center_normal - obj->normals.at<cv::Point3f>(xC + y,yC + x)).dot(center_pos - obj->cloud.at<cv::Point3f>(xC + y,yC + x));
    curvatures[5] = (center_normal - obj->normals.at<cv::Point3f>(xC - y,yC + x)).dot(center_pos - obj->cloud.at<cv::Point3f>(xC - y,yC + x));
    curvatures[6] = (center_normal - obj->normals.at<cv::Point3f>(xC + y,yC - x)).dot(center_pos - obj->cloud.at<cv::Point3f>(xC + y,yC - x));
    curvatures[7] = (center_normal - obj->normals.at<cv::Point3f>(xC - y,yC - x)).dot(center_pos - obj->cloud.at<cv::Point3f>(xC - y,yC - x));
    //computa distancias
    distance[0] = (center_pos - obj->cloud.at<cv::Point3f>(xC + x,yC + y)).dot(center_pos - obj->cloud.at<cv::Point3f>(xC + x,yC + y));
    distance[1] = (center_pos - obj->cloud.at<cv::Point3f>(xC + x,yC - y)).dot(center_pos - obj->cloud.at<cv::Point3f>(xC + x,yC - y));
    distance[2] = (center_pos - obj->cloud.at<cv::Point3f>(xC - x,yC + y)).dot(center_pos - obj->cloud.at<cv::Point3f>(xC - x,yC + y));
    distance[3] = (center_pos - obj->cloud.at<cv::Point3f>(xC - x,yC - y)).dot(center_pos - obj->cloud.at<cv::Point3f>(xC - x,yC - y));
    distance[4] = (center_pos - obj->cloud.at<cv::Point3f>(xC + y,yC + x)).dot(center_pos - obj->cloud.at<cv::Point3f>(xC + y,yC + x));
    distance[5] = (center_pos - obj->cloud.at<cv::Point3f>(xC - y,yC + x)).dot(center_pos - obj->cloud.at<cv::Point3f>(xC - y,yC + x));
    distance[6] = (center_pos - obj->cloud.at<cv::Point3f>(xC + y,yC - x)).dot(center_pos - obj->cloud.at<cv::Point3f>(xC + y,yC - x));
    distance[7] = (center_pos - obj->cloud.at<cv::Point3f>(xC - y,yC - x)).dot(center_pos - obj->cloud.at<cv::Point3f>(xC - y,yC - x));
    //Calcula descritor...
    /*
      * Descritor utilizado:
      *  4 para nan (utilizar apenas para informar falta de informação pra árvore na hora do treinamento).
      *  3 se distancia euclidiana for maior que um dado threshold
      *  0 caracteristica planar
      *  1 curvatura positiva
      *  2 curvatura negativa
      */
    for(int i = 0; i < 8; i++,idx++) {
        if(distance[i] > DISTANCE_THRESHOLD) {
            feat[idx] = 3;
            continue;
        }
        if(isnan(normals[i])) {
            feat[idx] = 4;
            continue;
        }


        if(normals[i] > treshold) {
            feat[idx] = 0;
            continue;
        }
        feat[idx] = ((curvatures[i] > 0) ? 1:2);
    }
    //Calculo da "qualidade" do descritor
    /*
    for(int i = 0; i < 8; i++) {
        if(!isnan(normals[i]) && normals[i] < treshold)
            r += ((feat[idx] == 1) ? (1.-normals[i]):(0.05*(1.-normals[i])));
    }*/
    return r;
}

float collector::bresenhamApply2(int xC,int yC,int Radius,int& idx,float* feat,float (*f)(int,int,int,int,float* ,int&,collector*)) {
    int P;
    int x,y;
    P = 1 - Radius;
    x = 0;
    y = Radius;
    float r = f(x,y,xC,yC,feat,idx,this);
    while (x<=y)
    {
        x++;
        if (P<0)
        {
            P += 2 * x + 1;
        }
        else
        {
            P += 2 * (x - y) + 1;
            y--;
        }
        r += f(x,y,xC,yC,feat,idx,this);
    }
    return r;
}

//152 eh o tamanho do feature...
descriptor* collector::compute_features(int x,int y) {
    const int f_size = 904;
    descriptor* d = new descriptor();
    float* feat = new float[f_size];
    int idx = 0;
    d->quality = 0;
    d->features = feat;
    for(int r = 1; r <= 15; r++) {
        d->quality += bresenhamApply2(x,y,r,idx,feat,computeFeature);
    }
    return d;
}

bool collector::isMaximum(int iC,int jC,descriptor*** descriptors) {
    int size = 40;
    float d = img.at<float>(iC,jC);
    if(isnan(img.at<float>(iC,jC)) || d <= FLT_EPSILON)
        return false;
    size = (size < 10) ? 10:size;
    size = (int)floor(size*1/img.at<float>(iC,jC));
    int i,jStart,j,endI,endJ,halfSize;
    halfSize = (int)floor(size/2);
    i = ((iC - halfSize < 5) ? 5:(iC - halfSize));
    jStart = ((jC - halfSize < 5) ? 5:(jC - halfSize));
    endI = ((iC + halfSize > (this->img.rows-5)) ? (this->img.rows-5):(iC + halfSize));
    endJ = ((jC + halfSize > (this->img.cols-5)) ? (this->img.cols-5):(jC + halfSize));
    float qualityIJ = descriptors[iC][jC]->quality;
    //std::cout << "qualidade: " << qualityIJ << std::endl;
    for(; i < endI;i++) {
        j = jStart;
        for(; j < endJ;j++) {
            if(i == iC && j == jC)
                continue;
            if(descriptors[i][j] == NULL)
                continue;
            if(descriptors[i][j]->quality > qualityIJ){
                return false;

            }
        }
    }
    return true;
}


void collector::findStalactite2() {

    int rows = (img.rows);
    int cols = (img.cols);
    const int f_size = 152;
    std::list<cv::Point2d> keyPoints;
    CvDTreeNode* resultNode;
    //float* v;
    descriptor*** features = new descriptor**[rows];
    for(int i = 0; i < rows; i++) {
        features[i] = new descriptor*[cols];
    }
    //Compute features and stores it to further apply non-max Sup.
    std::cout << normals.rows << " " << normals.cols << std::endl;
    for(int i = 5; i < (normals.rows-5); i++)
        for(int j = 5; j < (normals.cols-5); j++) {
            features[i][j] = compute_features(i,j);
            cv::Mat sample(1,f_size,CV_32F,features[i][j]->features);
            cv::Mat mMask(1,f_size,CV_8U);

            for(int i = 0; i < mMask.cols;i++) {
                mMask.at<uchar>(0,i) = (((int)sample.at<float>(0,i)) == 4);
            }

            resultNode = tree.predict(sample, mMask, false);
            //resultNode = tree.predict(sample, cv::Mat(),false);
            int result = fabs(resultNode->value - 1) <= FLT_EPSILON;
            if(result) {
                keyPoints.push_back(cv::Point2d(i,j));
                //cv::circle(coloredImg,cv::Point(j,i),5,cv::Scalar(0,255,0));
            }
            else
                features[i][j]->quality = -1;
        }

    //Non-Maximal supression:
    std::list<cv::Point2d>::iterator it = keyPoints.begin();
    for(;it != keyPoints.end();it++) {
        int i = (*it).y;
        int j = (*it).x;
        if(isMaximum(j,i,features)){
            cv::circle(coloredImg,cv::Point(i,j),5,cv::Scalar(0,0,0));
        }
    }
    cv::imshow(wName,coloredImg);

    std::cout << "We finished dude! Check it out" << std::endl;
}

void cb(int event, int x, int y, int flags, void* userdata) {
    collector *obj = (collector*)userdata;
    //obj->windowCallBack(event,x,y,flags,NULL);
    //obj->wcb_regionsSelector(event,x,y,flags,NULL);
    obj->window_cb_clickTest(event,x,y,flags,NULL);
}

collector::collector(char* wName,cv::Mat mt,sensor_msgs::CameraInfo ci):wName(wName) {
    this->setImg(mt);
    this->totalPosPoints = 0;
    this->totalNegPoints = 0;
    this->counter_aux = 0;
    this->p_aux.x = -1; this->p_aux.y = -1;
    this->mouseControl = 0;
    cv::namedWindow(wName,1);
    cv::setMouseCallback(wName, cb, this);
    this->camera_info = ci;
    for(int i = 0; i < 15; i++) {
        cv::Mat mat1; cv::Mat mat3; cv::Mat mat5;
        cv::Mat mat2; cv::Mat mat4; cv::Mat mat6;
        this->posDotProducts.push_back(mat2);
        this->negDotProducts.push_back(mat1);
        this->posK.push_back(mat3);
        this->negK.push_back(mat4);
        this->posED.push_back(mat5);
        this->negED.push_back(mat6);
    }
    this->show();
}

collector::~collector() {
    cv::destroyWindow(wName);
}

void collector::loadTree(std::string filename) {
    tree.load(filename.c_str());
}


void collector::show() {
    double min,max;
    cv::minMaxIdx(img,&min,&max);
    img.convertTo(coloredImg,CV_8UC1,255 / (max-min), -min);
    cv::applyColorMap(coloredImg,coloredImg,cv::COLORMAP_AUTUMN);
    cv::imshow(wName,coloredImg);

    //cv::imshow(wName,img);
}


void collector::wcb_regionsSelector(int event,int x,int y, int flags, void* userdata) {
    if (event == cv::EVENT_LBUTTONDOWN) {
        if(ask_for_normals) {
            this->compute_normals();
            ask_for_normals = 0;
        }
        this->p_aux.x = x;
        this->p_aux.y = y;
        cv::Point p(x,y);
        this->pointList = bresenhamApplyTemp(p.y,p.x,3,cloudCoordinatesTest);
        mouseControl = 1;
        return;
    }
    if (event == cv::EVENT_MOUSEMOVE) {
        if (p_aux.x != -1 && p_aux.y != -1) {
            int mousex,mousey;
            mousex = x; mousey = y;
            if(x < 0)
                mousex = 0;
            if(x > this->img.cols)
                mousex = img.cols;
            if(y < 0)
                mousey = 0;
            if (y > this->img.rows)
                mousey = this->img.rows;
            cv::Mat temp = coloredImg.clone();
            if(mouseControl == 1)
                cv::rectangle(temp,p_aux,cv::Point2d(mousex,mousey),cv::Scalar(255,0,0));
            if(mouseControl == 2)
                cv::line(temp,p_aux,cv::Point2d(mousex,mousey),cv::Scalar(0,255,0));
            cv::imshow(wName,temp);
            return;
        }
    }
    if(event == cv::EVENT_LBUTTONUP) {
        int mousex,mousey;
        mousex = x; mousey = y;
        if(x < 0)
            mousex = 0;
        if(x > this->img.cols)
            mousex = img.cols;
        if(y < 0)
            mousey = 0;
        if (y > this->img.rows)
            mousey = this->img.rows;
        cv::Rect roi(p_aux,cv::Point2d(mousex,mousey));
        cv::rectangle(coloredImg,roi,cv::Scalar(255,0,0));
        r_aux = roi;
        this->roi_queue.push_back(roi);
        p_aux.x = -1; p_aux.y = -1;
        mouseControl = 0;
        cv::imshow(wName,coloredImg);
        return;
    }
    if (event == cv::EVENT_RBUTTONDOWN) {
        this->p_aux.x = x;
        this->p_aux.y = y;
        this->mouseControl = 2;
        return;
    }
    if (event == cv::EVENT_RBUTTONUP) {
        int mousex,mousey;
        mousex = x; mousey = y;
        if(x < 0)
            mousex = 0;
        if(x > this->img.cols)
            mousex = img.cols;
        if(y < 0)
            mousey = 0;
        if (y > this->img.rows)
            mousey = this->img.rows;
        float size = 0;
        std::cout << "Getting cloud.." << std::endl;
        cv::Point3f p1 = this->cloud.at<cv::Point3f>(p_aux.y,p_aux.x);
        cv::Point3f p2 = this->cloud.at<cv::Point3f>(mousey,mousex);
        std::cout << "nope..." << std::endl;
        cv::Point3f diff = p1 - p2;
        cv::line(coloredImg,p_aux,cv::Point2d(mousex,mousey),cv::Scalar(0,255,0));
        cv::circle(coloredImg,p_aux,3,cv::Scalar(0,0,0));
        cv::circle(coloredImg,cv::Point2d(mousex,mousey),3,cv::Scalar(0,0,0));
        size = diff.dot(diff);
        size = sqrt(size);
        std::cout << "p1_img(" << p_aux.x << "," << p_aux.y << ") "
                  << "p2_img(" << mousex << "," << mousey << ") " << std::endl
                  << "P1(" << p1.x << "," << p1.y << "," << p1.z << ") "
                  << "P2(" << p2.x << "," << p2.y << "," << p2.z << ")" << std::endl;
        std::cout << "Tamanho: " << size << std::endl;
        center1 = p_aux;
        center2.x = mousex;
        center2.y = mousey;
        p_aux.x = -1; p_aux.y = -1;
        mouseControl = 0;
        cv::imshow(wName,coloredImg);

    }
}

void collector::windowCallBack(int event, int x, int y, int flags, void* userdata) {

    if (event == cv::EVENT_LBUTTONDOWN) {
        cv::Point p(x,y);
        cv::Vec3b blue(255,0,0);
        cv::Vec3b green(0,255,0);
        cv::Vec3b black(0,0,0);
        cv::Vec3b pColor = coloredImg.at<cv::Vec3b>(p);
        if(x - 5 < 0 || x + 5 > img.cols || y - 5 < 0 || y + 5 > img.rows)
            return;
        if(pColor != blue && pColor != green) {
            positivePoints.push_back(p);
            this->coloredImg.at<cv::Vec3b>(p) = blue;
            cv::circle(this->coloredImg,p,10,cv::Scalar(255,0,0));
            cv::imshow(wName,coloredImg);
            return;
        }
        if(pColor == blue) {
            positivePoints.remove(p);
            this->coloredImg.at<cv::Vec3b>(p) = black;
            cv::circle(this->coloredImg,p,10,cv::Scalar(0,0,0));

            cv::imshow(wName,coloredImg);
            return;
        }
    }
    if (event == cv::EVENT_RBUTTONDOWN) {
        cv::Point p(x,y);
        cv::Vec3b blue(255,0,0);
        cv::Vec3b green(0,255,0);
        cv::Vec3b black(0,0,0);
        cv::Vec3b pColor = coloredImg.at<cv::Vec3b>(x,y);
        if(x - 5 < 0 || x + 5 > img.cols || y - 5 < 0 || y + 5 > img.rows)
            return;
        if(pColor != blue && pColor != green) {
            negativePoints.push_back(p);
            this->coloredImg.at<cv::Vec3b>(p) = green;
            cv::circle(this->coloredImg,p,10,cv::Scalar(0,255,0));
            cv::imshow(wName,coloredImg);
            return;
        }
        if(pColor == green) {
            negativePoints.remove(p);
            this->coloredImg.at<cv::Vec3b>(p) = black;
            cv::circle(this->coloredImg,p,10,cv::Scalar(0,0,0));
            cv::imshow(wName,coloredImg);
            return;
        }
    }
}

void collector::window_cb_clickTest(int event, int x, int y, int flags, void* userdata) {
    /*
465.597 u
429.881 std
      */
    //float g[4] = { 0.286408, 0.41653, 0.282107, 0.0149545} ;

    float g2[4] = { 0.324565, 0.329844, 0.345592, 0};
    float g[4] = { 0.262362, 0.4106, 0.314239, 0.0127997};

    float g3[4] = {0.359466, 0.321703, 0.309114, 0.00971709};
    if(event == cv::EVENT_LBUTTONDOWN) {
        if(ask_for_normals) {
            this->compute_normals();
            std::cout << "normals computed" << std::endl;

            ask_for_normals = 0;
        }
        descriptor *d = compute_features(y,x);
        float en; bool valid;
        entropy_calc2(d->features,904,g3,en,valid);
        float instance_std = pow(fabs(en - 475),2);
        std::cout << "instance: " << instance_std << " " << valid << std::endl;
        std::cout << "normal: " << normals.at<cv::Point3f>(y,x).x << std::endl;
        pcl::Normal pn1 = pcl_normals->at(x,y);
        std::cout << "match: " << pn1.normal_x <<  "curvature: " << pn1.curvature << std::endl;
        cv::Scalar color(0,255,0);
        if(instance_std <=  90 || !valid)
            color = cv::Scalar(0,0,255);
        cv::circle(coloredImg,cv::Point(x,y),5,color);
        cv::imshow(wName,coloredImg);

    }
}

void collector::curvatureFilter(std::list<cv::Point> &pList) {
    compute_normals();
    float t = 0.2;
    for(int x = 0; x < pcl_normals->width; x++) {
        for(int y = 0; y < pcl_normals->height; y++) {
            float k = pcl_normals->at(x,y).curvature;
            if(k > t && k < 0.3) {
                pList.push_back(cv::Point(x,y));
                cv::circle(coloredImg,cv::Point(x,y),5,cv::Scalar(0,255,0));
            }
        }
    }

}

void collector::setImg(cv::Mat img) {
    this->img = img;
    this->deph = img;
    //    this->ate_cloud();
    //    this->compute_normals();
}



void collector::getPointsDescriptors() {
    cv::Point p;
    std::vector<float>* out_N;
    std::vector<float>* out_K;
    std::vector<float>* out_ED;


    if(positivePoints.size() == 0 && negativePoints.size() == 0)
        return;
    this->compute_normals();
    if(positivePoints.size() > 0) {
        this->totalPosPoints += positivePoints.size();
        while(this->positivePoints.size()) {
            p = positivePoints.front();
            for(int r = 1; r <= 15; r++) {
                out_N = this->bresenhamApply(p.y,p.x,r,computeNormalDots); //Here we need to change the coordenates to match opencv's coords.
                out_K = this->bresenhamApply(p.y,p.x,r,computeK);
                out_ED = this->bresenhamApply(p.y,p.x,r,computeED);
                cv::Mat pointDescriptorDotNormals(*out_N);
                cv::Mat pointK(*out_K);
                cv::Mat pointED(*out_ED);
                (posDotProducts.at(r-1)).push_back(pointDescriptorDotNormals);
                (posK.at(r-1)).push_back(pointK);
                (posED.at(r-1)).push_back(pointED);
            }
            positivePoints.pop_front();
        }
    }

    if(negativePoints.size() > 0) {
        this->totalNegPoints += negativePoints.size();
        while(this->negativePoints.size()) {
            p = negativePoints.front();
            for(int r = 1; r <= 15; r++) {
                out_N = this->bresenhamApply(p.y,p.x,r,computeNormalDots); //Here we need to change the coordenates to match opencv's coords.
                out_K = this->bresenhamApply(p.y,p.x,r,computeK);
                out_ED = this->bresenhamApply(p.y,p.x,r,computeED);

                cv::Mat pointDescriptorDotNormals(*out_N);
                cv::Mat pointK(*out_K);
                cv::Mat pointED(*out_ED);

                negDotProducts.at((r-1)).push_back(pointDescriptorDotNormals);
                (negK.at(r-1)).push_back(pointK);
                (negED.at(r-1)).push_back(pointED);

            }
            negativePoints.pop_front();
        }
    }
}
void collector::writeRois(std::string prefix) {
    cv::FileStorage fs;
    int qtd = this->roi_queue.size();
    for(int i = 0; i < qtd; i++, counter_aux++) {
        char filename[20];
        char filename_yml[20];
        snprintf(filename,20,"%s%d.png",prefix.c_str(),counter_aux);
        snprintf(filename_yml,20,"%s%d.yml",prefix.c_str(),counter_aux);
        cv::Rect temp = roi_queue.at(i);
        cv::Mat cropColorImg;
        cv::Mat cropDepthImg = img(temp);
        cropColorImg.create(cropDepthImg.size(),CV_8UC1);
        fs.open(filename_yml,cv::FileStorage::WRITE);
        fs << "Mat" << cropDepthImg;
        fs.release();

        double min,max;
        cv::minMaxIdx(cropDepthImg,&min,&max);
        std::cout << "min and max: " << min << " " << max << std::endl;
        cropDepthImg.convertTo(cropColorImg,CV_8UC1,255 / (max-min), -min);
        cv::applyColorMap(cropColorImg,cropColorImg,cv::COLORMAP_AUTUMN);
        cv::imwrite(filename,cropColorImg);

    }
    this->roi_queue.clear();
}

void collector::writeNormals(std::string prefix) {
    if(normals_queue.empty())
        return;
    cv::FileStorage fs;
    fs.open(prefix,cv::FileStorage::WRITE);
    cv::Mat m(normals_queue);
    fs << "normals" << m;
    fs.release();
}

void collector::writeDescriptors(std::string prefix) {
    cv::FileStorage fs;
    fs.open(prefix,cv::FileStorage::WRITE);
    for(int i = 0; i < 15; i++) { /*Writing positive examples..*/
        std::stringstream pos_nodeName;
        std::stringstream neg_nodeName;
        std::stringstream pos_k;
        std::stringstream neg_k;
        std::stringstream posED;
        std::stringstream negED;

        pos_nodeName << "positiveDotNormalsR" << (i+1);
        neg_nodeName << "negativeDotNormalsR" << (i+1);
        pos_k << "K_positivePointsR" << (i+1);
        neg_k << "K_negativePointsR" << (i+1);
        posED << "ED_positivePointsR" << (i+1);
        negED << "ED_negativePointsR" << (i+1);

        fs << pos_nodeName.str().c_str() << this->posDotProducts.at(i).reshape(0,this->totalPosPoints);
        fs << neg_nodeName.str().c_str() << this->negDotProducts.at(i).reshape(0,this->totalNegPoints);
        fs << pos_k.str().c_str() << this->posK.at(i).reshape(0,this->totalPosPoints);
        fs << neg_k.str().c_str() << this->negK.at(i).reshape(0,this->totalNegPoints);
        fs << posED.str().c_str() << this->posED.at(i).reshape(0,this->totalPosPoints);
        fs << negED.str().c_str() << this->negED.at(i).reshape(0,this->totalNegPoints);
    }
    fs.release();
}

void collector::showcloud(cv::Point3f *norms,int size) {
    this->compute_normals();

    pcl::visualization::PCLVisualizer viewer("Nuvem");
    viewer.setBackgroundColor (100, 100, 100);

    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(pcl_cloud);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_normals (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::concatenateFields(*pcl_cloud, *pcl_normals, *cloud_normals);

    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZRGBNormal> handler_curvature (cloud_normals, "curvature");
    viewer.addPointCloud(cloud_normals, handler_curvature, "cloud_curvature");
    //viewer.addPointCloud<pcl::PointXYZRGB>(pcl_cloud, rgb,"cloud");
    //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2,"cloud");
    //viewer.addPointCloudNormals<pcl::PointXYZRGB,pcl::Normal>(pcl_cloud,pcl_normals,10, 0.02, "normals");
    viewer.addCoordinateSystem (1.0);
    viewer.initCameraParameters();
    for(int i = 0;  i < size;i++) {
        printf("Normal[%d]: %f %f %f\n",i,norms[i].x,norms[i].y,norms[i].z);
        char s[30];
        cv::Rect roi = roi_queue.at(i);
        int x = roi.x + floor(roi.width/2);
        int y = roi.y + floor(roi.height/2);
        pcl::PointXYZRGB p1 = pcl_cloud->at(x,y);
        pcl::PointXYZRGB p2;
        p2.x = p1.x + norms[i].x;
        p2.y = p1.y + norms[i].y;
        p2.z = p1.z + norms[i].z;
        snprintf(s,30,"l%d",i);
        viewer.addLine(p1,p2,s);
    }

    while(!viewer.wasStopped()) {
        viewer.spinOnce();
    }
}

void collector::compute_normals()
{
    this->create_cloud();
    //dN_matrix = cv::Mat(img.rows*img.cols,img.rows*img.cols,CV_32FC1);
    //K_matrix = cv::Mat(img.rows*img.cols,img.rows*img.cols,CV_32FC1);
    //dN_matrix += 2.1;
    //K_matrix += 2.1;

    pcl_cloud_gray.reset(new pcl::PointCloud<pcl::PointXYZ>);
    pcl_cloud_gray->clear();
    pcl_cloud_gray->width     = img.cols;
    pcl_cloud_gray->height    = img.rows;
    pcl_cloud_gray->points.resize( pcl_cloud_gray->width * pcl_cloud_gray->height);

    pcl_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);


    pcl_cloud->clear();
    pcl_cloud->width     = img.cols;
    pcl_cloud->height    = img.rows;
    pcl_cloud->points.resize( pcl_cloud->width * pcl_cloud->height);

    for(int y = 0; y < img.rows; y++)
        for(int x = 0; x < img.cols; x++)
        {
            pcl_cloud->at(x,y).x = this->cloud.at<cv::Point3f>(y,x).x;
            pcl_cloud->at(x,y).y = this->cloud.at<cv::Point3f>(y,x).y;
            pcl_cloud->at(x,y).z = this->cloud.at<cv::Point3f>(y,x).z;

            pcl_cloud_gray->at(x,y).x = this->cloud.at<cv::Point3f>(y,x).x;
            pcl_cloud_gray->at(x,y).y = this->cloud.at<cv::Point3f>(y,x).y;
            pcl_cloud_gray->at(x,y).z = this->cloud.at<cv::Point3f>(y,x).z;
        }

    pcl_normals.reset(new pcl::PointCloud<pcl::Normal>);
    pcl_normals->clear();
    pcl_normals->width  = pcl_cloud->width;
    pcl_normals->height = pcl_cloud->height;
    pcl_normals->points.resize(pcl_cloud->width * pcl_cloud->height);
    std::vector<int> idx;
    //pcl::removeNaNFromPointCloud(*pcl_cloud,*pcl_cloud,idx);
    /*
    pcl::IntegralImageNormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud( pcl_cloud );

    ne.setNormalSmoothingSize( 5 );
    ne.setNormalEstimationMethod(ne.COVARIANCE_MATRIX);
    ne.compute( *pcl_normals );
    */
    pcl::IntegralImageNormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
    ne.setNormalEstimationMethod(ne.COVARIANCE_MATRIX);

    //ne.setMaxDepthChangeFactor(0.2f);
    ne.setKSearch(100);
    //ne.setRadiusSearch(1.0f);
    //ne.setNormalSmoothingSize(20.0f);
    ne.setInputCloud(pcl_cloud);
    ne.compute(*pcl_normals);
    std::vector<int> idx_normals;
    normals.create( pcl_normals->height, pcl_normals->width, CV_32FC3 );

    for(int y = 0; y < pcl_normals->height; ++y)
        for(int x = 0; x < pcl_normals->width; ++x)
        {
            normals.at<cv::Point3f>(y,x).x = pcl_normals->at(x,y).normal_x;
            normals.at<cv::Point3f>(y,x).y = pcl_normals->at(x,y).normal_y;
            normals.at<cv::Point3f>(y,x).z = pcl_normals->at(x,y).normal_z;
        }
    //pcl::removeNaNNormalsFromPointCloud(*pcl_normals,*pcl_normals,idx_normals);

    /*
    // visualize normals
    pcl::visualization::PCLVisualizer viewer("Normals");
    viewer.setBackgroundColor (0.0, 0.0, 0.5);
    viewer.addPointCloudNormals<pcl::PointXYZ,pcl::Normal>(pcl_cloud, pcl_normals);

    while (!viewer.wasStopped ())
    {
        viewer.spinOnce ();
    }*/

}
/*
 * This funciton, together with the function draw, will build a list of dotproducts(central_point,circle_point) for each circle point.
 */


void collector::create_cloud()
{

    /* Projection/camera matrix
#     [fx'  0  cx' Tx]
# P = [ 0  fy' cy' Ty]
#     [ 0   0   1   0]
      */
    float fx = (float)camera_info.P[0];
    float fy = (float)camera_info.P[5];
    float cx = (float)camera_info.P[2];
    float cy = (float)camera_info.P[6];
    char bsp = ' ';
    std::cout << "Parametros: " << fx <<  bsp << fy << bsp << cx << bsp << cy << std::endl;
    const float inv_fx = 1.f/fx;
    const float inv_fy = 1.f/fy;

    cloud.create( img.size(), CV_32FC3 );

    for( int y = 0; y < cloud.rows; y++ )
    {
        cv::Point3f* cloud_ptr = (cv::Point3f*)cloud.ptr(y);
        //const float* depth_ptr = (float*)img.ptr(y);
        //const ufloat32_t* depth_pt = (ufloat16_t*)img.ptr(y);
        for( int x = 0; x < cloud.cols; x++ )
        {
            float d = img.at<float>(y,x);
            if(isnan(d)) {
                //d = 20000;
            }

            cloud_ptr[x].x = (x - cx) * d /fx;
            cloud_ptr[x].y = (y - cy) * d /fy;
            cloud_ptr[x].z = d;
        }
    }
}



void collector::drawCircle(int xC,int yC,int Radius) {
    int P;
    int x,y;
    P = 1 - Radius;
    x = 0;
    y = Radius;
    Draw(x,y,xC,yC);
    while (x<=y)
    {
        x++;
        if (P<0)
        {
            P += 2 * x + 1;
        }
        else
        {
            P += 2 * (x - y) + 1;
            y--;
        }
        Draw(x,y,xC,yC);
    }
}

void collector::Draw(int x,int y,int xC,int yC)
{
    cv::Vec3b preto(0,0,0);
    coloredImg.at<cv::Vec3b>(xC + x,yC + y) = preto;
    coloredImg.at<cv::Vec3b>(xC + x,yC - y) = preto;
    coloredImg.at<cv::Vec3b>(xC - x,yC + y) = preto;
    coloredImg.at<cv::Vec3b>(xC - x,yC - y) = preto;
    coloredImg.at<cv::Vec3b>(xC + y,yC + x) = preto;
    coloredImg.at<cv::Vec3b>(xC - y,yC + x) = preto;
    coloredImg.at<cv::Vec3b>(xC + y,yC - x) = preto;
    coloredImg.at<cv::Vec3b>(xC - y,yC - x) = preto;
}

cv::Mat collector::kvd(cv::Mat dotNormals,float t) {
    for(int i = 0; i < dotNormals.rows;i++) {
        for(int j = 0; j < dotNormals.cols; j++) {
        }
    }

}

void loadNormals(char* filename,cv::Mat &m) {
    cv::FileStorage fs;
    fs.open(filename,cv::FileStorage::READ);
    fs["normals"] >> m;
    fs.release();
}

void visualize_normals(cv::Mat normals) {
    pcl::visualization::PCLVisualizer viewer("normals");
    viewer.addCoordinateSystem(1.0);
    viewer.initCameraParameters();
    for(int i = 0; i < normals.rows; i++) {
        char s[5000];
        snprintf(s,5000,"%d",i);
        cv::Point3f p3f = normals.at<cv::Point3f>(i);
        pcl::PointXYZ p_pcl(p3f.x,p3f.y,p3f.z);
        viewer.addLine(pcl::PointXYZ(0,0,0),p_pcl,s);
    }
    while(!viewer.wasStopped()) {
        viewer.spinOnce();
    }
}


void pre_process_cloud(pcl::PointCloud<pcl::PointXYZ>::Ptr &pcd,bool downsample) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr src;
    pcl::VoxelGrid<pcl::PointXYZ> grid;
    src = pcd;

    //Downsampling...

    std::vector<int> idx;
    src->is_dense = false;
    pcl::removeNaNFromPointCloud(*src,*src,idx);
    if(downsample) {
        src.reset(new pcl::PointCloud<pcl::PointXYZ>);
        grid.setLeafSize (0.1, 0.1, 0.1);
        grid.setInputCloud (pcd);
        grid.filter(*src);
    }


    //Normal Calculation...

    pcl::PointCloud<pcl::PointNormal>::Ptr normals(new pcl::PointCloud<pcl::PointNormal>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    pcl::NormalEstimation<pcl::PointXYZ,pcl::PointNormal> ne;
    ne.setInputCloud(src);
    ne.setSearchMethod(tree);
    ne.setKSearch(30);
    ne.compute(*normals);

    pcl::concatenateFields(*src,*normals,*normals);

    pcl::io::savePCDFileASCII("normals2.pcd",*normals);


    //Visualization
    pcl::visualization::PCLVisualizer *v = new pcl::visualization::PCLVisualizer ("Processed cloud");
    v->setBackgroundColor(0,0,0);
    v->addCoordinateSystem(1.0);



    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointNormal> handler_k(normals,"curvature");
    if (!handler_k.isCapable ())
        PCL_WARN ("Cannot create curvature color handler!");

    v->addPointCloud(normals,handler_k,"cloud");
    v->spin();

}
//pcl::io::savePCDFileASCII("teste3.pcd",*src);

//    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_normals(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
//    pcl::concatenateFields(*src,*normals,*cloud_normals);
//    for(int i = 0; i < normals->size(); i++)
//        std::cout << cloud_normals->at(i).curvature << " ";
//    std::cout << std::endl;
