#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"

#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

// This is the datatype that the PrepareFiles function will return
// PrepareFiles loads csv file, training images, and haar cascade file
struct modelandcascade {
	Ptr<FaceRecognizer> themodel;
	CascadeClassifier thecc;
	int w, h;
};

modelandcascade PrepareFiles(string& imageslist, string& haarfile){
	modelandcascade datatoreturn;
	ifstream readcsv;
	readcsv.open(imageslist);

	vector<Mat> images; // holds all the training images
	vector<int> labels; //holds the labels (1 for person#1, etc) corresponding to each image in images

	string line, path, classlabel;

	while (getline(readcsv, line)) { // reads each line of "readcsv" into "line"
		stringstream liness(line);
		getline(liness, path, ';');  // first half of each line: file path
		getline(liness, classlabel); // second half of each line: label
		if (!path.empty() && !classlabel.empty()) {
			images.push_back(imread(path, 0));
			labels.push_back(atoi(classlabel.c_str()));
		}
	}
	datatoreturn.w = images[0].cols;
	datatoreturn.h = images[0].rows;

	datatoreturn.themodel = createFisherFaceRecognizer();
	datatoreturn.themodel->train(images, labels);
	datatoreturn.thecc.load(haarfile);
	return datatoreturn;
}

int Parttwo(modelandcascade& returned){
	/*This is the function that does the actual face-recognition. It takes as input
	the model, the cascade classifier, and the height and width of the training-images.*/
	Mat frame; // will hold the current frame
	Mat original; // will hold a copy of the current frame
	Mat gray;  // will hold a grayscale copy of the current frame
	vector< Rect_<int> > faces; // will hold the faces in each frame
	Mat resized; // all images must be same size; this will hold resized img
	int currentp; //holds the current prediction
	string box_text; // holds the label for the box drawn around the current face

	// Now try to open the webcam
	VideoCapture cap(0);
	if (!cap.isOpened()) {
		cerr << "Webcam 0 cannot be opened." << endl;
		return -1;
	}

	for (;;){
		cap >> frame;
		original = frame.clone();
		cvtColor(original, gray, CV_BGR2GRAY);
		returned.thecc.detectMultiScale(gray, faces);

		//Now loop over all of the detected faces to identify each one.
		for (int i = 0; i < faces.size(); i++) {
			cv::resize(gray(faces[i]), resized, Size(returned.w, returned.h), 1.0, 1.0, INTER_CUBIC);
			currentp = returned.themodel->predict(resized);

			rectangle(original, faces[i], CV_RGB(0, 255, 0), 1);
			box_text = format("Prediction = %d", currentp);
			int pos_x = std::max(faces[i].tl().x - 10, 0);
			int pos_y = std::max(faces[i].tl().y - 10, 0);

			putText(original, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 2.0);
		}
		imshow("face_recognizer", original);

		// Display the frame with face-boxes:
		char key = (char)waitKey(20);
	}
	return 0;
}

int main(int argc, const char *argv[]) {
	//Get filenames for training images and Haar cascades
	string imageslist, haarfile;
	cout << "Enter name of the file with list of training images, e.g. f.csv: ";
	cin >> imageslist;
	cout << "Enter name of the file with the Haar Cascades, e.g. a.xml: ";
	cin >> haarfile;

	// Part 1: Prepare the training images and Haar cascades
	modelandcascade mac = PrepareFiles(imageslist, haarfile);

	// Part 2: using training images and cascades, do face recognition
	Parttwo(mac);
	return 0;
}