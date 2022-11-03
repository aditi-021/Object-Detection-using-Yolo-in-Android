package com.example.javacp_objectdetection_1;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.SurfaceView;
import android.view.View;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect;
import org.opencv.core.MatOfRect2d;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;
import org.opencv.dnn.Dnn;
import org.opencv.utils.Converters;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static String TAG = "MainActivity";

    JavaCameraView javaCameraView;
    int counter = 0;
    boolean startYOLO = false;
    boolean firstTimeYOLO = false;
    Net tinyYOLO;

    private final int PERMISSIONS_READ_CAMERA=1;


    // onRequestPermissionResult
    @Override
    public void onRequestPermissionsResult(
            int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {

        // Ensure that this result is for the camera permission request
        if (requestCode == PERMISSIONS_READ_CAMERA) {
            // Check if the request was granted or denied
            if (grantResults.length > 0
                    && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                // The request was granted -> tell the camera view
                javaCameraView.setCameraPermissionGranted();
            } else {
                // The request was denied -> tell the user and exit the application
                Toast.makeText(this, "Camera permission required.",
                        Toast.LENGTH_LONG).show();
                this.finish();
            }
        } else {
            super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        }
    }

    BaseLoaderCallback baseLoaderCallback = new BaseLoaderCallback(MainActivity.this) {

        @Override
        public void onManagerConnected(int status) {
            Log.d(TAG, "callbacksuccess");
            switch(status){
                case BaseLoaderCallback.SUCCESS:{
                    Log.d(TAG,"case success");
                    javaCameraView.enableView();
                    break;
                }
                default:
                {
                    Log.d(TAG, "case default");
                    super.onManagerConnected(status);
                    break;
                }
            }

        }
    };
    static{
        if (OpenCVLoader.initDebug()){
            Log.d(TAG, "OpenCV is initialised");
        }
        else{
            Log.d(TAG, "OpenCV is not initialised");
        }
    }

    public void YOLO(View Button){
        if (startYOLO == false){

            startYOLO = true;

            if (firstTimeYOLO == false){
                firstTimeYOLO = true;
//                String tinyYOLOcfg = Environment.getExternalStorageDirectory() + "/dnns/yolov3-tiny.cfg";
//                String tinyYOLOWeights= Environment.getExternalStorageDirectory() + "/dnns/yolov3-tiny.weights";
//
//
                String tinyYoloCfg = getPath("yolo3-tiny.cfg",this);
                String tinyYoloWeights = getPath("yolov3-tiny.weights",this);

                tinyYOLO = Dnn.readNetFromDarknet(tinyYoloCfg, tinyYoloWeights);



            }
        }
        else{
            startYOLO = false;
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        Log.d(TAG, "onCreate");
        setContentView(R.layout.activity_main);

        javaCameraView = (JavaCameraView)findViewById(R.id.CameraView);
        javaCameraView.setVisibility(SurfaceView.VISIBLE);
        javaCameraView.setCvCameraViewListener(MainActivity.this);


        //Here, this activity is the current activity
        if(ContextCompat.checkSelfPermission(this,Manifest.permission.CAMERA)!=PackageManager.PERMISSION_GRANTED){


            if (ActivityCompat.shouldShowRequestPermissionRationale(this,Manifest.permission.CAMERA)){

            }
            else{
              ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, PERMISSIONS_READ_CAMERA);


            }

        }
        else{
            Log.d(TAG, "Permission granted");
            javaCameraView.setCameraPermissionGranted();
        }


    }




    @Override
    public void onCameraViewStarted(int width, int height) {
        Log.d(TAG, "onCameraViewStarted");

        if (startYOLO == true) {

//            String tinyYoloCfg = Environment.getExternalStorageDirectory() + "/dnns/yolov3-tiny.cfg";
//            String tinyYoloWeights = Environment.getExternalStorageDirectory() + "/dnns/yolov3-tiny.weights";
//
//

            String tinyYoloCfg = getPath("yolo3-tiny.cfg",this);
            String tinyYoloWeights = getPath("yolov3-tiny.weights",this);

            tinyYOLO = Dnn.readNetFromDarknet(tinyYoloCfg, tinyYoloWeights);
        }


    }

    @Override
    public void onCameraViewStopped() {
        Log.d(TAG, "onCameraViewStopped");


    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        Mat frame = inputFrame.rgba();

        if (startYOLO == true){

            Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2RGB);

            Mat imageBlob = Dnn.blobFromImage(frame, 0.00392, new Size(416,416), new Scalar(0,0,0), false, false);

            tinyYOLO.setInput(imageBlob);

            java.util.List<Mat> result = new java.util.ArrayList<Mat>(2);

            List<String> outBlobNames = new java.util.ArrayList<>();
            outBlobNames.add(0, "yolo_16");
            outBlobNames.add(1, "yolo_23");

            tinyYOLO.forward(result,outBlobNames);

            float confThreshold = 0.3f;

            List<Integer> clsIds = new ArrayList<>();
            List<Float> confs = new ArrayList<>();
            List<Rect> rects = new ArrayList<>();

            for (int i =0; i< result.size(); ++i){

                Mat level = result.get(i);

                for (int j = 0; j < level.rows(); ++j){
                    Mat row = level.row(j);
                    Mat scores = row.colRange(5, level.cols());

                    Core.MinMaxLocResult mm = Core.minMaxLoc(scores);

                    float confidence = (float)mm.maxVal;

                    Point classIdPoint = mm.maxLoc;

                    if(confidence > confThreshold){
                        int centerX = (int)(row.get(0,0)[0] * frame.cols());
                        int centerY = (int)(row.get(0,1)[0] * frame.rows());
                        int width   = (int)(row.get(0,2)[0] * frame.cols());
                        int height  = (int)(row.get(0,3)[0] * frame.rows());


                        int left    = centerX - width  / 2;
                        int top     = centerY - height / 2;

                        clsIds.add((int)classIdPoint.x);
                        confs.add((float)confidence);

                        rects.add(new Rect(left, top, width, height));
                    }

                }
            }
            int ArrayLength = confs.size();

            if (ArrayLength>=1) {
                // Apply non-maximum suppression procedure.
                float nmsThresh = 0.2f;

                MatOfFloat confidences = new MatOfFloat(Converters.vector_float_to_Mat(confs));


                Rect2d[] boxesArray = rects.toArray(new Rect2d[0]);

                MatOfRect2d boxes = new MatOfRect2d(boxesArray);

                MatOfInt indices = new MatOfInt();

                Dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThresh, indices);

                //Draw result boxes:
                int[] ind = indices.toArray();
                for (int i = 0; i < ind.length; ++i) {

                    int idx = ind[i];
                    Rect2d box = boxesArray[idx];

                    int idGuy = clsIds.get(idx);

                    float conf = confs.get(idx);


                    List<String> cocoNames = Arrays.asList("a person", "a bicycle", "a motorbike", "an airplane", "a bus", "a train", "a truck", "a boat", "a traffic light", "a fire hydrant", "a stop sign", "a parking meter", "a car", "a bench", "a bird", "a cat", "a dog", "a horse", "a sheep", "a cow", "an elephant", "a bear", "a zebra", "a giraffe", "a backpack", "an umbrella", "a handbag", "a tie", "a suitcase", "a frisbee", "skis", "a snowboard", "a sports ball", "a kite", "a baseball bat", "a baseball glove", "a skateboard", "a surfboard", "a tennis racket", "a bottle", "a wine glass", "a cup", "a fork", "a knife", "a spoon", "a bowl", "a banana", "an apple", "a sandwich", "an orange", "broccoli", "a carrot", "a hot dog", "a pizza", "a doughnut", "a cake", "a chair", "a sofa", "a potted plant", "a bed", "a dining table", "a toilet", "a TV monitor", "a laptop", "a computer mouse", "a remote control", "a keyboard", "a cell phone", "a microwave", "an oven", "a toaster", "a sink", "a refrigerator", "a book", "a clock", "a vase", "a pair of scissors", "a teddy bear", "a hair drier", "a toothbrush");



                    int intConf = (int) (conf * 100);

                    int font = Imgproc.FONT_HERSHEY_SIMPLEX;
                    Imgproc.putText(frame,cocoNames.get(idGuy) + " " + intConf + "%",box.tl(),font, 2, new Scalar(255,255,0),2);



                    Imgproc.rectangle(frame, box.tl(), box.br(), new Scalar(255, 0, 0), 2);





                }
            }








            }



        return frame;

    }

    @Override
    protected void onResume() {
        super.onResume();
        Log.d(TAG, "onResume");
        if (OpenCVLoader.initDebug())
        {
            Log.d(TAG, "OpenCV is intialised again");
            baseLoaderCallback.onManagerConnected((BaseLoaderCallback.SUCCESS));
        }
        else
        {
            Log.d(TAG, "OpenCV is not working");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, baseLoaderCallback);
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        Log.d(TAG, "onPause");
        if (javaCameraView != null)
        {
            javaCameraView.disableView();
        }

    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        Log.d(TAG, "onDestroy");
        if (javaCameraView != null)
        {
            javaCameraView.disableView();
        }
    }

    private static String getPath(String file, Context context){
        AssetManager assetManager = context.getAssets();
        BufferedInputStream inputStream = null;
        try{
            //Read data from assets.
            inputStream = new BufferedInputStream(assetManager.open(file));
            byte[] data = new byte[inputStream.available()];
            inputStream.read(data);
            inputStream.close();
            //Create copy file storage.
            File outFile = new File(context.getFilesDir(),file);
            FileOutputStream os = new FileOutputStream(outFile);
            os.write(data);
            os.close();
            //Return a path to file which may be read normally
            return outFile.getAbsolutePath();
        }
        catch (IOException ex){
            Log.i(TAG, "Failed to upload a file");
        }
        return "";
    }




}


