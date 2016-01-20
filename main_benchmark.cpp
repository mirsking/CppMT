#include "CMT.h"
#include "gui.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <chrono>

#ifdef __GNUC__
#include <getopt.h>
#else
#include "getopt/getopt.h"
#endif

using cmt::CMT;
using cv::imread;
using cv::namedWindow;
using cv::Scalar;
using cv::VideoCapture;
using cv::waitKey;
using std::cerr;
using std::istream;
using std::ifstream;
using std::stringstream;
using std::ofstream;
using std::cout;
using std::min_element;
using std::max_element;
using std::endl;
using ::atof;

static string WIN_NAME = "CMT";

string write_rotated_rect(RotatedRect rect)
{
    Point2f verts[4];
    rect.points(verts);
    stringstream coords;

    coords << rect.center.x << " " << rect.center.y << " ";
    coords << rect.size.width << " " << rect.size.height << " ";

    return coords.str();
}
int display(Mat im, CMT & cmt)
{
    //Visualize the output
    //It is ok to draw on im itself, as CMT only uses the grayscale image
    for(size_t i = 0; i < cmt.points_active.size(); i++)
    {
        circle(im, cmt.points_active[i], 2, Scalar(255,0,0));
    }

    Point2f vertices[4];
    cmt.bb_rot.points(vertices);
    for (int i = 0; i < 4; i++)
    {
        line(im, vertices[i], vertices[(i+1)%4], Scalar(255,0,0));
    }

    imshow(WIN_NAME, im);

    return waitKey(5);
}

std::chrono::milliseconds getCurrentMs()
{
	using namespace std::chrono;
    milliseconds ms = duration_cast< milliseconds >(
		system_clock::now().time_since_epoch()
	);
	return ms;
}

//tracker.exe [context] [save] [show] [random] [seq_name] [inp_dir] [out_dir] [startframe] [endframe] [strname_len] [img_format] [x] [y] [w] [h]

int main(int argc, char **argv)
{
    //Create a CMT object
    CMT cmt;

    //Initialization bounding box
    Rect rect;

    //Set up logging
    FILELog::ReportingLevel() = logINFO;
    Output2FILE::Stream() = stdout; //Log to stdout
	
	if (argc != 16)
	{
		cerr << "Usage Error, please read the readme file!" << endl;
		return -1;
	}

	int context = atoi(argv[1]);
	bool save_result = !(atoi(argv[2]) == 0);
	bool show_result = !(atoi(argv[3]) == 0);
	bool rand_generation = !(atoi(argv[4]) == 0);
	std::string seq_name = argv[5];
	std::string inp_dir = argv[6];
	std::string out_dir = argv[7];
	int start_frame = atoi(argv[8]);
	int end_frame = atoi(argv[9]);
	int strname_len = atoi(argv[10]);
	std::string img_format = argv[11];
	rect.x = atof(argv[12]);
	rect.y = atof(argv[13]); 
	rect.width = atof(argv[14]);
	rect.height = atof(argv[15]); 

	if (show_result)
	{
		//Create window
		namedWindow(WIN_NAME);
	}

    VideoCapture cap;

	char buffer[100];
	sprintf(buffer, "%s%%0%dd.%s", inp_dir.c_str(), strname_len, img_format.c_str());
	std::string input_path = buffer;
	cout << "Input source is: " << endl <<  input_path << std::endl;
    cap.open(input_path);

    //If it doesn't work, stop
    if(!cap.isOpened())
    {
        cerr << "Unable to open video capture." << endl;
        return -1;
    }


    //Get initial image
    Mat im0;
	int cur_frame_index = 0;
	while (cur_frame_index != start_frame)
	{
	    cap >> im0;
		cur_frame_index++;
	}

    FILE_LOG(logINFO) << "Using " << rect.x << "," << rect.y << "," << rect.width << "," << rect.height
        << " as initial bounding box.";

    //Convert im0 to grayscale
    Mat im0_gray;
    if (im0.channels() > 1) {
        cvtColor(im0, im0_gray, CV_BGR2GRAY);
    } else {
        im0_gray = im0;
    }

    //Initialize CMT
    cmt.initialize(im0_gray, rect);

    //Open output file.
    ofstream output_file;
	std::string output_path = out_dir + seq_name + "_CMT.txt";
	std::string fps_output_path = out_dir + seq_name + "_CMT_FPS.txt";
	std::chrono::duration<float> fps_ms(60000);

    if (save_result)
    {
        output_file.open(output_path.c_str());
        output_file << write_rotated_rect(cmt.bb_rot) << endl;
    }

    //Main loop
	for (cur_frame_index+=1; cur_frame_index <= end_frame; cur_frame_index++ )
    {
		auto start_time = std::chrono::high_resolution_clock::now();

        Mat im;

        cap >> im; //Else use next image in stream
        if (im.empty()) break; //Exit at end of video stream

        Mat im_gray;
        if (im.channels() > 1) {
            cvtColor(im, im_gray, CV_BGR2GRAY);
        } else {
            im_gray = im;
        }

        //Let CMT process the frame
        cmt.processFrame(im_gray);

		auto end_time = std::chrono::high_resolution_clock::now();

        //Output.
        if (save_result)
        {
			std::chrono::duration<float> delta_time = end_time - start_time;
			if (delta_time < fps_ms)
				fps_ms = delta_time;
            output_file << write_rotated_rect(cmt.bb_rot) << endl;
        }
        else
        {
            //TODO: Provide meaningful output
            FILE_LOG(logINFO) << "#" << cur_frame_index << " active: " << cmt.points_active.size();
        }

		if (show_result)
		{
			//Display image and then quit if requested.
			char key = display(im, cmt);
			if (key == 'q') break;
		}
    }

    //Close output file.
	if (save_result)
	{
		output_file.close();
		output_file.open(fps_output_path);
		std::chrono::milliseconds ms = std::chrono::duration_cast<std::chrono::milliseconds>(fps_ms);
		output_file << float(1.0 / ms.count() * 1000) << endl;
		output_file.close();
	}

	return 0;
}
