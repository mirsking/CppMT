#include "CMT.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include "utils.h"

namespace cmt {
cv::Scalar colorTab[] =
{
    cv::Scalar(255,0,0),
    cv::Scalar(0,255,0),
    cv::Scalar(255, 100, 100)
};
int CMT::display_mirsking(cv::Mat img, std::vector<int>& labels)
{
    using namespace cv;
    Mat im;
    img.copyTo(im);
    //Visualize the output
    //It is ok to draw on im itself, as CMT only uses the grayscale image
    for(size_t i = 0; i < points_active.size(); i++)
    {
        int label_idx = labels[i];
        circle(im, points_active[i], 2, colorTab[label_idx]);
    }

    Point2f vertices[4];
    bb_rot.points(vertices);
    for (int i = 0; i < 4; i++)
    {
        line(im, vertices[i], vertices[(i+1)%4], Scalar(255,0,0));
    }

    std::string win_name = "mirsking_test_win";
    imshow(win_name, im);

    return waitKey(5);
}

void CMT::initialize(const Mat im_gray, const Rect rect)
{
    FILE_LOG(logDEBUG) << "CMT::initialize() call";

    //Remember initial size
    size_initial = rect.size();

    //Remember initial image
    im_prev = im_gray;

    //Compute center of rect
    Point2f center = Point2f(rect.x + rect.width/2.0, rect.y + rect.height/2.0);

    //Initialize rotated bounding box
    bb_rot = RotatedRect(center, size_initial, 0.0);

    //Initialize detector and descriptor
#if CV_MAJOR_VERSION > 2
    detector = cv::FastFeatureDetector::create();
    descriptor = cv::BRISK::create();
#else
    detector = FeatureDetector::create(str_detector);
    descriptor = DescriptorExtractor::create(str_descriptor);
#endif

    //Get initial keypoints in whole image and compute their descriptors
    vector<KeyPoint> keypoints;
    detector->detect(im_gray, keypoints);

    //Divide keypoints into foreground and background keypoints according to selection
    vector<KeyPoint> keypoints_fg;
    vector<KeyPoint> keypoints_bg;

    for (size_t i = 0; i < keypoints.size(); i++)
    {
        KeyPoint k = keypoints[i];
        Point2f pt = k.pt;

        if (pt.x > rect.x && pt.y > rect.y && pt.x < rect.br().x && pt.y < rect.br().y)
        {
            keypoints_fg.push_back(k);
        }

        else
        {
            keypoints_bg.push_back(k);
        }

    }

    //Create foreground classes
    vector<int> classes_fg;
    classes_fg.reserve(keypoints_fg.size());
    for (size_t i = 0; i < keypoints_fg.size(); i++)
    {
        classes_fg.push_back(i);
    }

    //Compute foreground/background features
    Mat descs_fg;
    Mat descs_bg;
    descriptor->compute(im_gray, keypoints_fg, descs_fg);
    descriptor->compute(im_gray, keypoints_bg, descs_bg);

    //Only now is the right time to convert keypoints to points, as compute() might remove some keypoints
    vector<Point2f> points_fg;
    vector<Point2f> points_bg;

    for (size_t i = 0; i < keypoints_fg.size(); i++)
    {
        points_fg.push_back(keypoints_fg[i].pt);
    }

    FILE_LOG(logDEBUG) << points_fg.size() << " foreground points.";

    for (size_t i = 0; i < keypoints_bg.size(); i++)
    {
        points_bg.push_back(keypoints_bg[i].pt);
    }

    //Create normalized points
    vector<Point2f> points_normalized;
    for (size_t i = 0; i < points_fg.size(); i++)
    {
        points_normalized.push_back(points_fg[i] - center);
    }

    //Initialize matcher
    matcher.initialize(points_normalized, descs_fg, classes_fg, descs_bg, center);

    //Initialize consensus
    consensus.initialize(points_normalized);

    //Create initial set of active keypoints
    for (size_t i = 0; i < keypoints_fg.size(); i++)
    {
        points_active.push_back(keypoints_fg[i].pt);
        classes_active = classes_fg;
    }

    FILE_LOG(logDEBUG) << "CMT::initialize() return";
}

void CMT::processFrame(Mat im_gray) {

    FILE_LOG(logDEBUG) << "CMT::processFrame() call";

    //Track keypoints
    vector<Point2f> points_tracked;
    vector<unsigned char> status;
    tracker.track(im_prev, im_gray, points_active, points_tracked, status);

    FILE_LOG(logDEBUG) << points_tracked.size() << " tracked points.";

    //keep only successful classes
    vector<int> classes_tracked;
    for (size_t i = 0; i < classes_active.size(); i++)
    {
        if (status[i])
        {
            classes_tracked.push_back(classes_active[i]);
        }

    }

    //Detect keypoints, compute descriptors
    vector<KeyPoint> keypoints;
    detector->detect(im_gray, keypoints);

    FILE_LOG(logDEBUG) << keypoints.size() << " keypoints found.";

    Mat descriptors;
    descriptor->compute(im_gray, keypoints, descriptors);

    //Match keypoints globally
    vector<Point2f> points_matched_global;
    vector<int> classes_matched_global;
    matcher.matchGlobal(keypoints, descriptors, points_matched_global, classes_matched_global);

    FILE_LOG(logDEBUG) << points_matched_global.size() << " points matched globally.";

    //Fuse tracked and globally matched points
    vector<Point2f> points_fused;
    vector<int> classes_fused;
    fusion.preferFirst(points_tracked, classes_tracked, points_matched_global, classes_matched_global,
            points_fused, classes_fused);

    FILE_LOG(logDEBUG) << points_fused.size() << " points fused.";

    //Estimate scale and rotation from the fused points
    float scale;
    float rotation;
    consensus.estimateScaleRotation(points_fused, classes_fused, scale, rotation);

    FILE_LOG(logDEBUG) << "scale " << scale << ", " << "rotation " << rotation;

    //Find inliers and the center of their votes
    Point2f center;
    vector<Point2f> points_inlier;
    vector<int> classes_inlier;
    consensus.findConsensus(points_fused, classes_fused, scale, rotation,
            center, points_inlier, classes_inlier);

    FILE_LOG(logDEBUG) << points_inlier.size() << " inlier points.";
    FILE_LOG(logDEBUG) << "center " << center;

    //Match keypoints locally
    vector<Point2f> points_matched_local;
    vector<int> classes_matched_local;
    matcher.matchLocal(keypoints, descriptors, center, scale, rotation, points_matched_local, classes_matched_local);

    FILE_LOG(logDEBUG) << points_matched_local.size() << " points matched locally.";

    //Clear active points
    points_active.clear();
    classes_active.clear();

    //Fuse locally matched points and inliers
    fusion.preferFirst(points_matched_local, classes_matched_local, points_inlier, classes_inlier, points_active, classes_active);
//    points_active = points_fused;
//    classes_active = classes_fused;

    FILE_LOG(logDEBUG) << points_active.size() << " final fused points.";

    //TODO: Use theta to suppress result
    bb_rot = RotatedRect(center,  size_initial * scale, rotation/CV_PI * 180);

    //TODO: mirsking: here to cluster active points
    std::vector<int> labels;
    postCluster(points_active, classes_active, labels);
    display_mirsking(im_gray, labels);

    //Remember current image
    im_prev = im_gray;

    FILE_LOG(logDEBUG) << "CMT::processFrame() return";
}

void CMT::postCluster(vector<Point2f> &points_active, vector<int>& classes_active, std::vector<int> &labels)
{
    FILE_LOG(logDEBUG) << "CMT::postCluster() call";

    //step 1. try use kmeans to cluster the data
    FILE_LOG(logDEBUG) << "CMT::postCluster() : clustering";
    //TODO: parameter1: cluster_count
    const int cluster_count = 2;
    cv::Mat cluster_centers;
    cv::kmeans(cv::Mat(points_active), cluster_count, labels,
               cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 0.1),
               3, cv::KMEANS_PP_CENTERS,
               cluster_centers);
    //step 2. calulate the rectangle
    FILE_LOG(logDEBUG) << "CMT::postCluster() : calulating overlap";
    vector< vector<Point2f> > points_clusters(cluster_count);
    vector< vector<int> > classes_clusters(cluster_count);
    for(size_t i=0; i<labels.size(); i++)
    {
        points_clusters[labels[i]].push_back(points_active[i]);
        classes_clusters[labels[i]].push_back(classes_active[i]);
    }
    vector<float> cluster_overlap(cluster_count);

    for(size_t i=0; i<cluster_count; i++)
    {
        vector<Point2f> &points = points_clusters[i];
        cv::RotatedRect rect = cv::minAreaRect(points);
        cluster_overlap[i] = calcRectOverlap(rect.boundingRect(), bb_rot.boundingRect());
        //std::cout << cluster_overlap[i] << std::endl;
    }
    float cluster_sigma = calcSigma(cluster_overlap);
    //TODO: parameter 2: sigma threshold
    const float cluster_threshold = 0.01;
    if(cluster_sigma > cluster_threshold)
    {
        std::cout << "error points detected" << std::endl;
        // find the max overlap and use their points;
        float max_overlap = 0.0;
        int max_overlap_index = -1;
        for(size_t i = 0; i< cluster_count; i++)
        {
            if(cluster_overlap[i] > max_overlap)
            {
                max_overlap = cluster_overlap[i];
                max_overlap_index = i;
            }
        }
        if(max_overlap_index == -1)
        {
            FILE_LOG(logWARNING) << "CMT::postCluster() : all rectangle has no overlap !";
        }
        else
        {
            points_active = points_clusters[max_overlap_index];
            classes_active = classes_clusters[max_overlap_index];
            bb_rot = cv::minAreaRect(points_active);
        }

    }




    FILE_LOG(logDEBUG) << "CMT::postCluster() return";
}

} /* namespace CMT */
