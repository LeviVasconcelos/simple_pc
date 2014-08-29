#include "simple_pc.h"
#include "collector.h"
#include "disjoint-set.h"
#include "segment-graph.h"
#include "segment-depthImg.h"
#include "pQueue.h"
#include <cstdlib>
const char* TOPIC_SUBSCRIBED = "/points";
const char* TOPIC_PUBLISHED_DEPTH_IMG = "/image_rect";
const char* TOPIC_PUBLISHED_RGB_IMG = "/rgb/image_rect_color";
const char* TOPIC_PUBLISHED_DEPTH_INFO = "/camera_info";
const char* TOPIC_PUBLISHED_RGB_INFO = "/rgb/camera_info";
const char* TOPIC_PUBLISHED_POINT_CLOUD = "/point_cloud";


std::string DEPTH_CAMERA_RECT_TOPIC = "/camera/depth/image_rect";
std::string DEPTH_CAMERA_INFO_TOPIC = "/camera/depth/camera_info";
std::string RGB_CAMERA_RECT_TOPIC = "/camera/rgb/image_rect";
std::string RGB_CAMERA_INFO_TOPIC = "/camera/rgb/camera_info";

const std::string t = "/points";

RGBDpair::~RGBDpair() {}

simple_pc::simple_pc(char* filename,std::vector<std::string> topics,int buffer_size = 20):
    bagFile(filename,rosbag::bagmode::Read),bagViewer(bagFile,rosbag::TopicQuery(topics)),topics(topics),buffer_size(buffer_size)
{
    this->currentBagPosition = bagViewer.begin();
    //sub_ = nh_.subscribe(TOPIC_SUBSCRIBED,100,&simple_pc::flushImgBuffer,this);
    pointPub_ = nh_.advertise<sensor_msgs::PointCloud2>(TOPIC_PUBLISHED_POINT_CLOUD,buffer_size);
    dethPub_ = nh_.advertise<sensor_msgs::Image>(TOPIC_PUBLISHED_DEPTH_IMG,buffer_size);
    rgbPub_ = nh_.advertise<sensor_msgs::Image>(TOPIC_PUBLISHED_RGB_IMG,buffer_size);
    rgbInfoPub_ = nh_.advertise<sensor_msgs::CameraInfo>(TOPIC_PUBLISHED_RGB_INFO,1);
    depthInfoPub_ = nh_.advertise<sensor_msgs::CameraInfo>(TOPIC_PUBLISHED_DEPTH_INFO,1);
    //pointSub_ = nh_.subscribe("/points",100,&simple_pc::pointCallBack,this);
    currentBagPosition = bagViewer.begin();
    ros::spinOnce();
    sleep(1);
    this->fillBuffer();
}

bool simple_pc::fillBuffer() {
    ROS_INFO("Enchendo Buffer Servidor...\n");
    BagSubscriber<sensor_msgs::Image> rgb_img_sub,depth_img_sub;
    BagSubscriber<sensor_msgs::CameraInfo> rgb_info_sub, depth_info_sub;
    /*
     * Cria callback para sincronizar depth image e intensity image
     */
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> mySyncPolice;
    message_filters::Synchronizer<mySyncPolice> sync(mySyncPolice(10), depth_img_sub, rgb_img_sub);
    sync.registerCallback(boost::bind(&simple_pc::sync_XtionCallBack, _1, _2, this));


    current_filling = 0;
    for(; this->currentBagPosition != bagViewer.end(); this->currentBagPosition++) {
        rosbag::MessageInstance m = *(this->currentBagPosition);
        if(current_filling >= buffer_size) {
            //TODO: Se precisar, implementar continuação de onde parou no arquivo bag
            break;
        }

        if(m.getTopic() == DEPTH_CAMERA_RECT_TOPIC) {
            sensor_msgs::Image::ConstPtr img_ptr = m.instantiate<sensor_msgs::Image>();
            if(!img_ptr)
            {
                ROS_ERROR("ERROR: null pointer in simple_pc.fillBag(), couldn't instantiate %s \n",DEPTH_CAMERA_RECT_TOPIC.c_str());
                return false;
            }
            depth_img_sub.newMessage(img_ptr);
        }
        if(m.getTopic() == DEPTH_CAMERA_INFO_TOPIC) {
            if(this->depthInfoPtr)
                continue;
            sensor_msgs::CameraInfo::ConstPtr info_ptr = m.instantiate<sensor_msgs::CameraInfo>();
            if(!info_ptr) {
                ROS_ERROR("ERROR: null pointer in simple_pc.fillBag(), couldn't instantiate %s \n",DEPTH_CAMERA_INFO_TOPIC.c_str());
                return false;
            }
            this->depthInfo = *info_ptr;
            this->depthInfoPtr = info_ptr;
        }
        if(m.getTopic() == RGB_CAMERA_RECT_TOPIC) {
            sensor_msgs::Image::ConstPtr img_ptr = m.instantiate<sensor_msgs::Image>();
            if(!img_ptr) {
                ROS_ERROR("ERROR: null pointer in simple_pc.fillBag(), couldn't instantiate %s \n",RGB_CAMERA_RECT_TOPIC.c_str());
                return false;
            }
            rgb_img_sub.newMessage(img_ptr);
        }
        if(m.getTopic() == RGB_CAMERA_INFO_TOPIC) {
            if(this->rgbInfoPtr)
                continue;
            sensor_msgs::CameraInfo::ConstPtr info_ptr = m.instantiate<sensor_msgs::CameraInfo>();
            if(!info_ptr) {
                ROS_ERROR("ERROR: null pointer in simple_pc.fillBag, couldn't instantiate %s\n",RGB_CAMERA_INFO_TOPIC.c_str());
            }
            this->rgbInfo = *info_ptr;
            this->rgbInfoPtr = info_ptr;
        }
    }
    return true;
}

void  simple_pc::sync_XtionCallBack(const sensor_msgs::Image::ConstPtr dImg,
                                    const sensor_msgs::Image::ConstPtr rgbImg,
                                    simple_pc *obj)
{
    RGBDpair *p = new RGBDpair(dImg,rgbImg);
    obj->imgBuffer.push_back(p);
    obj->current_filling++;
    /*
    obj->depthInfo.header.stamp = dImg->header.stamp;
    obj->dethPub_.publish(*dImg);
    obj->depthInfoPub_.publish(obj->depthInfo);
    obj->current_filling++;
            */
    //ROS_INFO("Pblishing... %s,%s",obj->dethPub_.getTopic(),obj->depthInfoPub_.getTopic());
}

void simple_pc::pointCallBack(pcl::PCLPointCloud2ConstPtr pcd) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromPCLPointCloud2(*pcd,*cloud);
    this->pcds.push_back(cloud);
}

void simple_pc::flushImgBuffer(const std_msgs::Int32& i) {
    ROS_INFO("Fui invocado flushImgBuffer msg: %d\n",i.data);
    if (i.data != 1)
        return;
    if (current_filling == 0) {
        ROS_ERROR("ERROR: Empty Buffer, at simple_pc::flushBuffer\n");
        return;
    }
    ROS_INFO("Enviando buffer do servidor...\n");
    int c = 0;
    RGBDpair * past;
    foreach(RGBDpair* it,this->imgBuffer) {
        if(c++ > 0) {
            this->imgBuffer.pop_front();
            delete past;
        }
        ROS_INFO("Enviando: %d\n",c);
        this->depthInfo.header.stamp = (it)->depthImg->header.stamp;
        this->rgbInfo.header.stamp = it->rgbImg->header.stamp;
        this->dethPub_.publish(it->depthImg);
        this->rgbPub_.publish(it->rgbImg);
        this->depthInfoPub_.publish(this->depthInfo);
        this->rgbInfoPub_.publish(this->rgbInfo);
        sleep(1);
        past = it;
    }
    this->fillBuffer();
}

cv::Mat simple_pc::nextImg() {
    cv::Mat r;
    try{
        std::list<RGBDpair*>::iterator it = this->imgBuffer.begin();
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy((*it)->depthImg,sensor_msgs::image_encodings::TYPE_32FC1);
        //cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy((*it)->rgbImg,sensor_msgs::image_encodings::BGR8);
        //cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy((*it)->rgbImg,sensor_msgs::image_encodings::BGR8);
        r = cv_ptr->image;
        this->imgBuffer.pop_front();
        delete *it;
        this->current_filling--;
    }
    catch(cv_bridge::Exception &e) {
        ROS_ERROR("Couldnt convert... %s",e.what());
    }
    return r;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr simple_pc::nextCloud() {
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcd = (pcds.front());
    pcds.pop_front();
    return pcd;
}

template <class M> void BagSubscriber<M>::newMessage(const boost::shared_ptr<M const> &msg)
{
    signalMessage(msg);
}



















CvDTree* train_tree(cv::Mat train,cv::Mat trainTarg) {
    CvDTree* tree = new CvDTree;
    cv::Mat var_type = cv::Mat(train.cols + 1, 1, CV_8U );
    var_type.setTo(cv::Scalar(CV_VAR_CATEGORICAL) ); // all inputs are categorical
    //var_type.setTo(cv::Scalar(CV_VAR_NUMERICAL));
    //var_type.at<uchar>(train.cols,0) = CV_VAR_CATEGORICAL;

    //Mask for missing data
    cv::Mat missing;
    missing.create(train.size(),CV_8U);
    for(int i = 0; i < train.rows;i++)
        for(int j = 0; j < train.cols;j++) {
            missing.at<uchar>(i,j) = (((int)train.at<float>(i,j)) == 4);
        }

    // define the parameters for training the decision tree

    float priors[] = {5,1};  // weights of each classification for classes
    // (all equal as equal samples of each character)


    CvDTreeParams params = CvDTreeParams(40, // max depth
                                         30, // min sample count
                                         0, // regression accuracy: N/A here
                                         true, // compute surrogate split
                                         15, // max number of categories (use sub-optimal algorithm for larger numbers)
                                         30, // the number of cross-validation folds
                                         true, // use 1SE rule => smaller tree
                                         true, // throw away the pruned tree branches
                                         priors // the array of priors
                                         );


    // train decision tree classifier (using training data)
    tree->train(train,CV_ROW_SAMPLE,trainTarg, cv::Mat(), cv::Mat(), var_type, missing,params);
    return tree;
}






void trainTestTree() {

    kvd_descriptor kvd("pontos_positivos_pontas_eslactites.yml");
    kvd.readFromFile("pontos_negativos.yml");

    /*
    kvd_descriptor kvd("previousData/base_se8.yml");
    kvd.readFromFile("previousData/base_se11.yml");
    kvd.readFromFile("previousData/base_se12.yml");
    */
    kvd.compute_descriptors(0.90,0.2,0,15);
    kvd.visualize(700,-1);
    cv::Mat train,train_targ,test,test_targ;
    kvd.build_train_test(0.7,train,train_targ,test,test_targ);
    std::cout << "Rows,cols: " << train.rows << "," << train.cols << std::endl;
    std::cout << "Rows,cols: " << test.rows << "," << test.cols << std::endl;
    float p2 = 0;

    CvDTree* tree = train_tree(train,train_targ);
    std::cout << "Arvore treinada!!" << std::endl;

    CvDTreeNode* resultNode;
    int NUMBER_OF_CLASSES = 2;
    cv::Mat test_sample;
    cv::Mat missing_mask;
    int correct_class = 0;
    int wrong_class = 0;
    int false_positives [NUMBER_OF_CLASSES];
    char class_labels[NUMBER_OF_CLASSES];

    // zero the false positive counters in a simple loop

    for (int i = 0; i < NUMBER_OF_CLASSES; i++)
    {
        false_positives[i] = 0;
        class_labels[i] = (char) 65 + i; // ASCII 65 = A
    }

    //Creating test Mask...
    cv::Mat missing_test;
    missing_test.create(test.size(), CV_8U);
    for(int i = 0; i < test.rows; i++) {
        for(int j = 0; j < test.cols; j++) {
            missing_test.at<uchar>(i,j) = (((int)test.at<float>(i,j)) == 4);
        }
    }

    //    printf( "\nUsing testing database: %s\n\n", argv[2]);

    for (int tsample = 0; tsample < test.rows; tsample++)
    {

        // extract a row from the testing matrix

        test_sample = test.row(tsample);
        missing_mask = missing_test.row(tsample);
        // run decision tree prediction

        resultNode = tree->predict(test_sample, missing_mask, false);

        //printf("Testing Sample %i -> class result (character %c)\n", tsample,
             //  class_labels[((int) (resultNode->value)) - 1]);

        // if the prediction and the (true) testing classification are the same
        // (N.B. openCV uses a floating point decision tree implementation!)

        if (fabs(resultNode->value - test_targ.at<float>(tsample, 0))
                >= FLT_EPSILON)
        {
            // if they differ more than floating point error => wrong class

            wrong_class++;

            false_positives[((int) (resultNode->value)) - 1]++;

        }
        else
        {

            // otherwise correct

            correct_class++;
        }
    }

    printf( "\nResults on the testing database: \n"
            "\tCorrect classification: %d (%g%%)\n"
            "\tWrong classifications: %d (%g%%)\n",
            correct_class, (double) correct_class*100/test.rows,
            wrong_class, (double) wrong_class*100/test.rows);

    for (int i = 0; i < NUMBER_OF_CLASSES; i++)
    {
        printf( "\tClass (character %c) false postives 	%d (%g%%)\n", class_labels[i],
                false_positives[i],
                (double) false_positives[i]*100/test.rows);
    }
    tree->save("dtree.tr");

}


typedef struct filaInteiro {
    int numb;
    struct filaInteiro *next;
}filaInteiro;

int compare_fi(filaInteiro *ff1,filaInteiro *ff2) {
    return (ff1->numb == ff2->numb);
}
void printFi(filaInteiro *f) {
    std::cout << "(" << f->numb << ") ";
}
void probability_calc(cv::Mat src,float value, float &prob) {
    int total,found;
    found = 0;
    int missing_Data = 0;
    total = src.rows * src.cols;
    for(int i = 0; i < src.rows; i++) {
        for(int j = 0; j < src.cols; j++) {
            float v = src.at<float>(i,j);
            //std::cout << v <<" " << value << " " << fabs(v - value) << std::endl;
            if(fabs(v - value) <= 0.5){
                //std::cout << "got it" << std::endl;
                found++;
            }
            if(fabs(v - 4.) <= 0.5) {
                missing_Data++;
            }
        }
    }
    std::cout << "missing_Data: " << missing_Data << "/" << total << std::endl;
    total -= missing_Data;
    prob = (float)found/(float)total;
}
void data_analyzer() {
    //kvd_descriptor kvd("pontos_positivos_pontas_eslactites.yml");
    //kvd.readFromFile("pontos_negativos.yml");
    //kvd_descriptor kvd("positive_complete_stalactites.yml");
    //kvd.readFromFile("negative_general.yml");
    kvd_descriptor kvd("pontos_positivos_pontas_filtradas.yml");
    kvd.readFromFile("pontos_negtivos_filtrados.yml");
    kvd.compute_descriptors(0.99,0.25,0,15);
    kvd.kvd_pos.convertTo(kvd.kvd_pos,CV_32FC1);
    kvd.kvd_neg.convertTo(kvd.kvd_neg,CV_32FC1);
    //kvd.visualize(700,-1);
    float p[4];
    float n[4];
    float g[4];
    cv::Mat G;
    cv::vconcat(kvd.kvd_pos,kvd.kvd_neg,G);
    probability_calc(G,0.,g[0]);
    probability_calc(G,1.,g[1]);
    probability_calc(G,2.,g[2]);
    probability_calc(G,3.,g[3]);

    probability_calc(kvd.kvd_pos,0.,p[0]);
    probability_calc(kvd.kvd_pos,1.,p[1]);
    probability_calc(kvd.kvd_pos,2.,p[2]);
    probability_calc(kvd.kvd_pos,3.,p[3]);

    probability_calc(kvd.kvd_neg,0.,n[0]);
    probability_calc(kvd.kvd_neg,1.,n[1]);
    probability_calc(kvd.kvd_neg,2.,n[2]);
    probability_calc(kvd.kvd_neg,3.,n[3]);

    std::cout <<"Probs: " << p[0] << " " << p[1] << " " << p[2] << " " << p[3] << " " << p[1]+p[2]+p[3]+p[0] << std::endl;
    std::cout <<"Probs: " << n[0] << " " << n[1] << " " << n[2] << " " << n[3] << " " << n[1]+n[2]+n[3]+n[0] << std::endl;
    std::cout <<"Probs: " << g[0] << " " << g[1] << " " << g[2] << " " << g[3] << " " << g[1]+g[2]+g[3]+g[0] << std::endl;
    float en = 0;


    int numb = 7000;
    srand(time(NULL));
    std::cout << "number: " << numb << std::endl;
    float mean = 0.;
    float std = 0;
    int counter = 0;
    for(int i = 0; i < numb; i++) {
        int row = rand() % kvd.kvd_pos.rows;
        bool valid;
        entropy_calc(kvd.kvd_pos.row(row),g,en,valid);
        float temp_std = pow(fabs(en - 474),2);
        mean += (valid) ? en:0;
        std += (valid) ? temp_std:0;
        counter += (valid) ? 1:0;
    }
    std::cout << mean/counter << std::endl;
    std::cout << std/counter << std::endl << std::endl;
    mean = 0.;
    std = 0.;
    counter = 0;
    for(int i = 0; i < numb; i++) {
        int row = rand() % kvd.kvd_neg.rows;
        bool valid;

        entropy_calc(kvd.kvd_neg.row(row),g,en,valid);
        float temp_std = pow(fabs(en - 474),2);

        mean += (valid) ? en:0;
        std += (valid) ? temp_std:0;

        counter += (valid) ? 1:0;
    }
        std::cout << mean/counter << std::endl;
        std::cout << std/counter << std::endl << std::endl;

}

int main(int argc,char **argv) {

    ros::init(argc,argv,"simple_pc");
    std::vector<std::string> topics;
    topics.push_back("/camera/depth/image_rect");
    topics.push_back( "/camera/depth/camera_info");
    topics.push_back("/camera/rgb/image_rect");
    topics.push_back("/camera/rgb/camera_info");

    //data_analyzer();

    //std::cout << "Treinando arvore..." << std::endl;
    //trainTestTree();
    simple_pc *s = new simple_pc(argv[1],topics,20);
    cv::Mat img = s->nextImg();
    collector myCol("teste",img,s->depthInfo);
    //myCol.loadTree("dtree.tr");
    std::stringstream pathname;
    pathname << argv[1];
    std::string bagName;
    while(!pathname.eof()) {
        std::getline(pathname,bagName,'/');
    }
    pathname.clear();
    pathname << bagName;
    std::getline(pathname,bagName,'.');
    pathname.clear();
    std::cout << pathname.str() << std::endl;

    //Reading CameraInfo
    float* cInfo = new float[12];
    cv::FileStorage fs;
    fs.open("cinfo.yml",cv::FileStorage::READ);
    cv::FileNode f = fs["cinfo"];
    cv::FileNodeIterator it = f.begin(), it_end = f.end();
    for(int idx = 0; it != it_end; ++it,idx++) {
        cInfo[idx] = *it;
    }
    fs.release();
    cv::Mat n;
    loadNormals("se10_normals.yml",n);
    visualize_normals(n);

    char keyPressed; int imgCounter = 0;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
    for(int i = 0; i < 350; i++) {
        //for(int j = 0; j < 100; j++,ros::spinOnce());
        //cloud = s->nextCloud();
        if(s->current_filling < 3) {
            s->fillBuffer();
            //for(int j = 0; j < 100; j++,ros::spinOnce());
            std::cout << "done!" << std::endl;
            i = 0;
        }
        img = s->nextImg();
        myCol.setImg(img);
        //myCol.pcd = cloud;
        myCol.show();
        //cv::imshow("rgbImg",img);
        do{
            keyPressed = (char)cv::waitKey(0);
            if(keyPressed == 'x') {
                std::cout << "getting descriptors..." << std::endl;
                myCol.getPointsDescriptors();
                std::cout << "in" << std::endl;
                myCol.writeDescriptors(bagName+".yml");
                std::cout << "out" << std::endl;

                myCol.writeNormals(bagName+"_normals.yml");
                return 0;
            }
            if(keyPressed == 'f') {
                //for(; s->current_filling > 3; s->nextImg());
                //break;
                //myCol.findStalactite(3);
                std::cout << "Computing normals and shit..." << std::endl;
                myCol.compute_normals();
                std::cout << "finished, now the features..." << std::endl;
                myCol.findStalactite2();
                cv::imshow(myCol.wName,myCol.coloredImg);
            }
            if(keyPressed == 'l') { //load cloud and normals...

            }
            if(keyPressed == 'w') {
                cv::FileStorage fs;
                std::stringstream fname;
                fname << imgCounter++ << ".yml";
                fs.open(fname.str(),cv::FileStorage::WRITE);
                fs << "img" << myCol.img;
                fs.release();
            }
            if(keyPressed == 'g') {
                depth_img dimg(myCol.img,cInfo);
                dimg.applyFilter();
                dimg.segment_distance(2,1,4.0);
                dimg.showColorImg();
                std::cout << "Universe numSet: " << dimg.u->num_sets() << std::endl;
                filaInteiro *head = NULL;
                filaInteiro *newNode = NULL;
                filaInteiro *temp = NULL;
                int result = 0;
                std::cout << "not here.." << std::endl;
                for(int yi = 0; yi < dimg.dImg.rows; yi++) {
                    for(int xi = 0; xi < dimg.dImg.cols; xi++) {
                        int targ_comp = dimg.u->find(yi*dimg.dImg.cols + xi);
                        int size_comp = dimg.u->size(targ_comp);
                        newNode = (filaInteiro*)malloc(sizeof(filaInteiro));
                        newNode->numb = targ_comp;
                        newNode->next = NULL;
                        temp = head; result = 0;

                        if(size_comp > 400) {
                            while(temp != NULL) {
                                if(temp->numb == newNode->numb) {
                                    result = 1;
                                    break;
                                }
                                temp = temp->next;
                            }
                            if(result) {
                                free(newNode);
                                continue;
                            }
                            PQ_ADD_ELEMENT(head,newNode,PQ_FIFO);
                            cv::Mat outMask;
                            outMask.create(dimg.dImg.size(),CV_32FC1);
                            getComponentMask(dimg,targ_comp,outMask);
                            cv::waitKey();
                        }
                        else{
                            free(newNode);
                            continue;
                        }
                    }
                }

            }
            if(keyPressed == 'c') {
                cv::FileStorage fs("cinfo.yml",cv::FileStorage::WRITE);
                fs << "cinfo" << "[:";
                for(int i = 0; i < 12; i++)
                    fs << myCol.camera_info.P[i];
                fs << "]";
            }
            if(keyPressed == 's') {
                myCol.writeRois("positive");
            }
            if(keyPressed == 'p') {
                myCol.showcloud();
            }
            if(keyPressed == 'n') {
                myCol.getNormalMeans(1);
                myCol.roi_queue.clear();
            }
            if(keyPressed == ' ') {
                myCol.ask_for_normals = 1;
                myCol.extract_descriptor_information_from_region(0);
                if(!myCol.roi_queue.empty()){
                    myCol.getNormalMeans();
                    myCol.roi_queue.clear();
                }
            }
            if(keyPressed == 'k') {
                std::list<cv::Point> pList;
                myCol.curvatureFilter(pList);
                cv::imshow(myCol.wName,myCol.coloredImg);
            }
            if(keyPressed == 'a') {
                myCol.compute_normals();
                myCol.reg.addCloud(myCol.pcl_cloud_gray);
            }
            if(keyPressed == 'r') {
                myCol.reg.prepareToAlign();
                pre_process_cloud(myCol.reg.registered_cloud,1);

            }
        }while(keyPressed != ' ');
        //myCol.getPointsDescriptors();
    }

    ros::spin();





    /******************************** LOAD DATA AND TRAIN A DECISION TREE*******************/

    /*
    //Mounting the train and test data base
    kvd_descriptor kvd("previousData/base_se8.yml");
    kvd.readFromFile("previousData/base_se11.yml");
    kvd.readFromFile("previousData/base_se12.yml");

    kvd.compute_descriptors(0.95,0.3,-1); //computando com raio 3.
    //kvd.visualize(-1);
    cv::Mat train,train_targ,test,test_targ;
    kvd.build_train_test(0.7,train,train_targ,test,test_targ);

    cv::Mat var_type = cv::Mat(train.cols + 1, 1, CV_8U );
    //var_type.setTo(cv::Scalar(CV_VAR_CATEGORICAL) ); // all inputs are categorical
    var_type.setTo(cv::Scalar(CV_VAR_NUMERICAL));
    var_type.at<uchar>(train.cols,0) = CV_VAR_CATEGORICAL;

    CvDTreeNode* resultNode; // node returned from a prediction
    // define the parameters for training the decision tree

    float priors[] = {5,1};  // weights of each classification for classes
    // (all equal as equal samples of each character)

    CvDTreeParams params = CvDTreeParams(25, // max depth
                                         5, // min sample count
                                         0, // regression accuracy: N/A here
                                         true, // compute surrogate split, no missing data
                                         15, // max number of categories (use sub-optimal algorithm for larger numbers)
                                         15, // the number of cross-validation folds
                                         true, // use 1SE rule => smaller tree
                                         true, // throw away the pruned tree branches
                                         priors // the array of priors
                                        );


    // train decision tree classifier (using training data)

    CvDTree* dtree = new CvDTree;

    dtree->train(train, CV_ROW_SAMPLE, train_targ,
                 cv::Mat(), cv::Mat(), var_type, cv::Mat(), params);
    std::cout << "treinado!" << std::endl;
    // perform classifier testing and report results


    int NUMBER_OF_CLASSES = 2;
    cv::Mat test_sample;
    int correct_class = 0;
    int wrong_class = 0;
    int false_positives [NUMBER_OF_CLASSES];
    char class_labels[NUMBER_OF_CLASSES];

    // zero the false positive counters in a simple loop

    for (int i = 0; i < NUMBER_OF_CLASSES; i++)
    {
        false_positives[i] = 0;
        class_labels[i] = (char) 65 + i; // ASCII 65 = A
    }

    printf( "\nUsing testing database: %s\n\n", argv[2]);

    for (int tsample = 0; tsample < test.rows; tsample++)
    {

        // extract a row from the testing matrix

        test_sample = test.row(tsample);

        // run decision tree prediction

        resultNode = dtree->predict(test_sample, cv::Mat(), false);

        printf("Testing Sample %i -> class result (character %c)\n", tsample,
               class_labels[((int) (resultNode->value)) - 1]);

        // if the prediction and the (true) testing classification are the same
        // (N.B. openCV uses a floating point decision tree implementation!)

        if (fabs(resultNode->value - test_targ.at<float>(tsample, 0))
                >= FLT_EPSILON)
        {
            // if they differ more than floating point error => wrong class

            wrong_class++;

            false_positives[((int) (resultNode->value)) - 1]++;

        }
        else
        {

            // otherwise correct

            correct_class++;
        }
    }

    printf( "\nResults on the testing database: \n"
            "\tCorrect classification: %d (%g%%)\n"
            "\tWrong classifications: %d (%g%%)\n",
            correct_class, (double) correct_class*100/test.rows,
            wrong_class, (double) wrong_class*100/test.rows);

    for (int i = 0; i < NUMBER_OF_CLASSES; i++)
    {
        printf( "\tClass (character %c) false postives 	%d (%g%%)\n", class_labels[i],
                false_positives[i],
                (double) false_positives[i]*100/test.rows);
    }
    dtree->save("dtree.tr");
    // all matrix memory free by destructors


    // all OK : main returns 0
*/
    return 0;

}
