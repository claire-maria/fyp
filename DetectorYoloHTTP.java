
package ie.tcd.netlab.objecttracker.detectors;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.support.annotation.Dimension;
import android.support.annotation.NonNull;
import android.support.v4.content.ContextCompat;
import android.support.v4.app.ActivityCompat;
import android.graphics.RectF;
import android.media.Image;
import android.content.Context;
import android.os.Build;
import android.graphics.Paint;

import java.io.ByteArrayOutputStream;
import java.io.FileOutputStream;
import android.graphics.drawable.BitmapDrawable;
import android.graphics.drawable.Drawable;

import java.lang.reflect.Array;
import java.util.Collections;
import java.util.Comparator;
import java.util.Formatter;
import java.util.Random;
import android.graphics.Canvas;
import java.util.ArrayList;
import android.graphics.BitmapFactory;
import android.graphics.Bitmap;
import java.io.IOException;
import java.io.File;
import android.os.Environment;
import android.util.Size;
import android.widget.Toast;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfFloat;
import org.opencv.core.Point;

import org.json.JSONArray;
import org.json.JSONObject;
import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.net.Socket;
import java.net.InetAddress;
import java.net.Inet4Address;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Core;
import org.opencv.core.Rect;

import org.opencv.imgcodecs.Imgcodecs;
import java.util.Arrays;
import ie.tcd.netlab.objecttracker.R;
import ie.tcd.netlab.objecttracker.helpers.Recognition;
import ie.tcd.netlab.objecttracker.helpers.Transform;
import ie.tcd.netlab.objecttracker.testing.Logger;

public class DetectorYoloHTTP extends Detector {

//    private final int jpegQuality;
    private int comQuality;
    private InetAddress IP;
    ByteBuffer IPbuf;
    private final String server;
    private final int port;
    private final boolean useUDP;
    private int udpsockfd=-1;
    private Socket tcpsock;
    BufferedOutputStream out;
    BufferedReader in;
    boolean executed = false;
    boolean saved = false;
    private final static int LISTSIZE=1000; // if change this then also change value in udp_socket_jni.c
    public static final String TAG = "MyActivity";
    private int outputWidth=300;
    private int outputHW=50;
    private Mat mOutputROI;

    private boolean bpUpdated = false;

    private Mat mRgba;
    private Mat mHSV;
    private Mat mask;

    private int lo = 20;
    private int up = 20;
    ByteBuffer recvbuf, image_bytes, req_buf, first_img_bytes;
    private final static int MSS=1472;          // max UDP payload (assuming 1500B packets)
    private static final boolean DEBUGGING = false;  // generate extra debug output ?
    public static final int REQUEST_WRITE_EXTERNAL = 3;
//    ArrayList<Bitmap> bitmaps = new ArrayList<>();
    ArrayList<Mat> croppedMatObjs = new ArrayList<>();
    ArrayList<Recognition> recArr = new ArrayList<>();
    ArrayList<Bitmap> coco = new ArrayList<>();
    ArrayList<Bitmap> cropped = new ArrayList<>();
    ArrayList<Bitmap> croppedBitmaps = new ArrayList<>();
    ArrayList<ArrayList<Bitmap>> listOfCMBLists = new ArrayList<>();
    byte[] combined;
    byte[] tstSize;
    byte[] cmbJpeg;
    int cmbH, cmbW;
    Rect origCords = new Rect();
    Bitmap temp;
    static {
        System.loadLibrary("udpsocket");
    }
    private native int socket(ByteBuffer addr, int port);
    private native void closesocket(int fd);
    private native String sendto(int fd, ByteBuffer sendbuf, int offset, int len, int MSS);
    private native String sendmmsg(int fd, ByteBuffer req, int req_len, ByteBuffer img, int img_len, int MSS);
    private native int recv(int fd, ByteBuffer recvbuf, int len, int MSS);
    private Context context;
    int delete;

    //private native void keepalive();

    public DetectorYoloHTTP(@NonNull Context context, String server, int comQuality, boolean useUDP) {

        String parts[] = server.split(":");
        this.server=parts[0]; this.port=Integer.valueOf(parts[1]); //server details
        this.IP=null; // this will force DNS resolution of server name in background thread below
                      // (since it may take a while and anyway DNS on the UI thread is banned by android).
        this.comQuality = 100;
        this.useUDP = useUDP;
        this.tcpsock = null;
        this.context = context;
        // can't open sockets here as may not yet have internet permission
        // only open them once, so that tcp syn-synack handshake is not repeated for every image
        if (!hasPermission(context)) { // need internet access to use YoloHTTP
            requestPermission((Activity) context);
        }
        // allocate byte buffers used to pass data to jni C
        recvbuf = ByteBuffer.allocateDirect(MSS*LISTSIZE);
        IPbuf = ByteBuffer.allocateDirect(4); // size of an IPv4 address
        image_bytes=ByteBuffer.allocateDirect(MSS*LISTSIZE);
        first_img_bytes = ByteBuffer.allocateDirect(MSS*LISTSIZE);
        req_buf=ByteBuffer.allocateDirect(MSS);

    }

    protected void finalize() {
        if (udpsockfd >0) {
            closesocket(udpsockfd);
        }
        if (this.tcpsock != null) {
            try {
                this.tcpsock.close();
            } catch(Exception e) {
                Logger.addln("\nWARN Problem closing TCP socket ("+e.getMessage()+")");
            }
        }
    }

    public Detections recognizeImage(Image image, int rotation) {


        android.graphics.Rect crop = image.getCropRect();
        int format = image.getFormat();
        delete = crop.width() * crop.height() *  ImageFormat.getBitsPerPixel(format)/8;
        if (image.getFormat() != ImageFormat.YUV_420_888) {
            // unsupported image format
            Logger.addln("\nWARN YoloHTTP.recognizeImage() unsupported image format");
            Log.e("HELP", "WRONG FORMAT");
            return new Detections();
        }
        return recognize(Transform.YUV420toNV21(image), image.getWidth(),image.getHeight(), rotation);
    }

    @Override
    public Detections recognize(byte[] yuv, int image_w, int image_h, int rotation) {


        ArrayList<Bitmap> arr = new ArrayList<>();

        float aspectRatio;
        //Create bitmap for splitting
        //TODO: OPENCV OBJECT DETECTION ON PHONE, SHOULD BE FAST ENOUGH ON REAL PHONES
        Bitmap temp = null;
        byte[] by = Transform.NV21toJPEG(yuv, image_w, image_h,100);
        Bitmap origialBitmap = BitmapFactory.decodeByteArray(by, 0, by.length);
        Log.e("Wtff", String.valueOf(by.length));
//        Bitmap[] quarters = splitBitmap(origialBitmap);
//        Bitmap quarterUsed = quarters[0];

        //TODO: Make this work for N coco images
//        if(!executed) {
//            Bitmap b = null;
//            Bitmap b1 = saveCoco(R.drawable.img0);
//            Bitmap b2 = saveCoco(R.drawable.img1);
//            Bitmap b3 = saveCoco(R.drawable.coco_val2014_000000001503);
//            Bitmap b4 = saveCoco(R.drawable.coco_val2014_000000001675);
//            Bitmap b5 = saveCoco(R.drawable.coco_val2014_000000002592);
//            Bitmap b6 = saveCoco(R.drawable.coco_val2014_000000003703);
//            Bitmap b7 = saveCoco(R.drawable.coco_val2014_000000004795);
//            Bitmap b8 = saveCoco(R.drawable.coco_val2014_000000006608);
//            Bitmap b9 = saveCoco(R.drawable.coco_val2014_000000008583);
//            Bitmap b10 = saveCoco(R.drawable.coco_val2014_000000009527);
//            Bitmap b11 = saveCoco(R.drawable.coco_val2014_000000011712);
//            Bitmap[] q1 = splitBitmap(b1);
//            Bitmap[] q2 = splitBitmap(b2);
//            Bitmap[] q3 = splitBitmap(b3);
//            Bitmap[] q4 = splitBitmap(b4);
//            Bitmap[] q5 = splitBitmap(b5);
//            Bitmap[] q6 = splitBitmap(b6);
//            Bitmap[] q7 = splitBitmap(b7);
//            Bitmap[] q8 = splitBitmap(b8);
//            Bitmap[] q9 = splitBitmap(b9);
//            Bitmap[] q10 = splitBitmap(b10);
//            Bitmap[] q11 = splitBitmap(b11);
//            b1 = q1[0];
//            b2 = q2[0];
//            b3 = q3[0];
//            b4 = q4[0];
//            b5 = q5[0];
//            b6 = q6[0];
//            b7 = q7[0];
//            b8 = q8[0];
//            b9 = q9[0];
//            b10 = q1[0];
//            b11 = q11[0];
//
//            coco.add(b1);
//            coco.add(b2);
//            coco.add(b3);
//            coco.add(b4);
//            coco.add(b5);
//            coco.add(b6);
//            coco.add(b7);
//            coco.add(b8);
//            coco.add(b9);
//            coco.add(b10);
//            coco.add(b11);
//
//            executed = true;
//

        int[] drawables = new int[] {R.drawable.img0,R.drawable.img1, R.drawable.coco_val2014_000000001503, R.drawable.coco_val2014_000000001675,
                R.drawable.coco_val2014_000000002592, R.drawable.coco_val2014_000000003703, R.drawable.coco_val2014_000000004795, R.drawable.coco_val2014_000000006608,
                R.drawable.coco_val2014_000000008583,R.drawable.coco_val2014_000000009527, R.drawable.coco_val2014_000000011712};
        if(!executed) {
            Bitmap b = null;
            for (int i = 0; i < 10; i++) {
                coco.add(saveCoco(drawables[i]));
            }
            for (int i = 0; i < coco.size(); i++) {
                b = objectDet(coco.get(i), outputHW, outputHW);
                Log.e("Change", String.valueOf(b.getHeight()));
                if (b.getHeight() > 100) {
                    cropped.add(b);
                }
            }
            executed = true;
        }

        Bitmap testOne = saveCoco(R.drawable.singcrop);
        testOne = getResizedBitmap(testOne,1000,1000);
        byte[] convrgbtoyuv = Transform.convertRGBtoYUV(testOne);
        combined = Transform.NV21toJPEG(convrgbtoyuv, 1000, 1000, comQuality);
        image_h = 1000;
        image_w = 1000;
//        Log.e("WHYYYY", String.valueOf(testOne.getWidth()));
//        combinedconvrgbtoyuv = bitmapToByteArray(testOne);

//        combined = Transform.YUVtoJPEG(combined, testOne.getWidth(), testOne.getHeight(), comQuality);
//        Bitmap bitmap = BitmapFactory.decodeByteArray(combined, 0, combined.length);
//        saveBitmapToExternalStorage(bitmap);

        cropped.add(origialBitmap);
        Bitmap b = try2(cropped);
      //  saveBitmapToExternalStorage(b);
        cropped.remove(origialBitmap);
////
//        image_h = testOne.getHeight();
//        image_w = testOne.getWidth();
     //   combined = bitmapToByteArray(b);
      //  cmbJPEG = Transform.NV21toJPEG(combined, b.getWidth(), b.getHeight(), comQuality);
        int bh;
        double bw;
//        image_h = 414;
//        bw = 480.3123993558776167471819645732689210950080515297906602254;

      //  Log.e("MVP", "Height:  " + bh + " Width:  " + bw + " Size:  " + combined.length);


      //  Log.e("SZ","Size  " + String.valueOf(send.length) + " Wid   " + ww + " Hei   " + hh);
        // takes yuv byte array as input
        Detections detects = new Detections();

        Logger.tick("d");
        Logger.tick("yuvtoJPG");
        int isYUV;
        image_bytes.clear();
        if (comQuality>0) {
            // we do rotation server-side, android client too slow (takes around 10ms in both java
            // and c on Huawei P9, while jpeg compressiovoidn takes around 8ms).
            try {
               // image_bytes.put(Transform.NV21toJPEG(combined, image_w, image_h, comQuality));

                image_bytes.put(combined);
                isYUV = 0;
            } catch (Exception e) {
                // most likely encoded image is too big for image_bytes buffer
                Logger.addln("WARN: Problem encoding jpg: "+e.getMessage());
                return detects; // bail
            }
        } else {
            // send image uncompressed
            image_bytes.put(combined);
            isYUV=1;
        }
 //       byte[] r = Transform.NV21toJPEG(yuv, image_w, image_h, 100);
        //TODO: What the fuck is it even seeing????????
//        byte[] killme = Transform.NV21toJPEG(yuv, image_w, image_h, comQuality);
//        Formatter formatter = new Formatter();
//        for (byte bb : killme) {
//            formatter.format("%02x", bb);
//        }
//        String hex = formatter.toString();
//        Log.e("DEL", hex);


//        byte[] testAfter = new byte[image_bytes.remaining()];
//        image_bytes.get(testAfter);

//        saveBitmapToExternalStorage(bitmap);



        detects.addTiming("yuvtoJPG",Logger.tockLong("yuvtoJPG"));
        Log.e("imgby", "After  " + String.valueOf(image_bytes.array()[0]));
        int dst_w=image_w, dst_h=image_h;
        if ((rotation%180 == 90) || (rotation%180 == -90)) {
            dst_w = image_h; dst_h = image_w;
        }
        Matrix frameToViewTransform = Transform.getTransformationMatrix(
                image_w, image_h,
                dst_w, dst_h,
                rotation, false);
        // used to map received response rectangles back to handset view
        Matrix viewToFrameTransform = new Matrix();
        frameToViewTransform.invert(viewToFrameTransform);

        if (IP==null) {
            // resolve server name to IP address
            try {
                InetAddress names[] = InetAddress.getAllByName(server);
                StringBuilder n = new StringBuilder();
                for (InetAddress name : names) {
                    n.append(name);
                    if (name instanceof Inet4Address) {IP = name; break;}
                }
                Logger.addln("\nResolved server to: "+IP);
                if (IP == null) {
                    Logger.addln("\nWARN Problem resolving server: "+n);
                    return detects;
                }

            } catch (IOException e) {
                Logger.addln("\nWARNProblem resolving server "+server+" :"+e.getMessage());
                return detects;
            }
        }

        String req = "POST /api/edge_app2?r=" + rotation
                + "&isYUV=" + isYUV + "&w="+ image_w + "&h="+image_h
                + " HTTP/1.1\r\nContent-Length: " + image_bytes.position() + "\r\n\r\n";
        StringBuilder response = new StringBuilder();
        if (useUDP) {
            try {
                Logger.tick("url2");
                // open connection (if not already open) and send request+image
                if (udpsockfd <0) {
                    // put the server IP address into a byte buffer to make it easy to pass to jni C
                    IPbuf.position(0);
                    IPbuf.put(IP.getAddress());
                    udpsockfd=socket(IPbuf,port);
                    Debug.println("sock_fd="+udpsockfd);
                }
                Debug.println("data len=("+req.length()+","+image_bytes.position()+")");
                Logger.tick("url2a");
                // copy request to byte buffer so easy to pass to jni C
                req_buf.clear();
                req_buf.put(req.getBytes(),0,req.length());
                String str = sendmmsg(udpsockfd, req_buf, req.length(), image_bytes, image_bytes.position(), MSS);
                Debug.println("s: "+str);
                //Logger.add("s: "+str);
                detects.addTiming("url2a",Logger.tockLong("url2a"));
                detects.addTiming("url2",Logger.tockLong("url2"));
                int count=1+(req.length()+image_bytes.position())/(MSS-2);
                detects.addTiming("pkt count", count*1000);

                // read the response ...
                Logger.tick("url3");
                // need to receive on same socket as used for sending or firewall blocks reception
                int resplen = recv(udpsockfd, recvbuf, MSS*LISTSIZE, MSS);
                if (resplen<0) {
                    Logger.addln("\nWARN UDP recv error: errno="+resplen);
                } else if (resplen==0) {
                    Logger.addln("\nWARN UDP timeout");
                } else {
                    response.append(new String(recvbuf.array(), recvbuf.arrayOffset(), resplen));
                }
                if (response.length()<=10) {
                    Debug.println(" received " + response.length());
                }
                detects.addTiming("url3",Logger.tockLong("url3"));
                Logger.addln(detects.client_timings.toString());
                //String pieces[] = response.split("\n");
                //response = pieces[pieces.length-1];  // ignore all the headers (shouldn't be any !)
            } catch(Exception e) {
                Logger.addln("\nWARN Problem with UDP on "+IP+":"+port+" ("+e.getMessage()+")");
            }
        } else { // use TCP
            try {
                // open connection and send request+image
                Logger.tick("url2");
                if (tcpsock == null) {
                    tcpsock = new Socket(IP, port);
                    out = new BufferedOutputStream(tcpsock.getOutputStream());
                    in = new BufferedReader(new InputStreamReader(tcpsock.getInputStream()));
                }
                try {
                    out.write(req.getBytes());
                    out.write(image_bytes.array(),image_bytes.arrayOffset(),image_bytes.position());
                    out.flush();
                } catch(IOException ee) {
                    // legacy server closes TCP connection after each response, in which case
                    // we reopen it here.
                    Logger.addln("Retrying TCP: "+ee.getMessage());
                    tcpsock.close();
                    tcpsock = new Socket(IP, port);
                    out = new BufferedOutputStream(tcpsock.getOutputStream());
                    in = new BufferedReader(new InputStreamReader(tcpsock.getInputStream()));
                    out.write(req.getBytes());
                    out.write(image_bytes.array());
                    out.flush();
                }
                detects.addTiming("url2",Logger.tockLong("url2"));

                Logger.tick("url3");
                // read the response ...
                // read the headers, we ignore them all !
                String line;
                while ((line = in.readLine()) != null) {
                    if (line.length() == 0) break; // end of headers, stop
                }
                // now read to end of response
                response.append(in.readLine());
                detects.addTiming("url3",Logger.tockLong("url3"));
            } catch(Exception e) {
                Logger.addln("\nWARN Problem connecting TCP to "+IP+":"+port+"");
                try {
                    tcpsock.close();
                } catch(Exception ee) {};
                tcpsock = null; // reset connection
            }
        }
        if (response.length()==0 || response.toString().equals("null")) {
            Logger.add(" empty response");
            Logger.add(": "+Logger.tock("d"));
            return detects; // server has dropped connection
        }
        // now parse the response as json ...
        try {
            // testing
            //response = "{"server_timings":{"size":91.2,"r":0.4,"jpg":8.4,"rot":34.1,"yolo":48.3,"tot":0},"results":[{"title":"diningtable","confidence":0.737176,"x":343,"y":415,"w":135,"h":296},{"title":"chair","confidence":0.641756,"x":338,"y":265,"w":75,"h":57},{"title":"chair","confidence":0.565877,"x":442,"y":420,"w":84,"h":421}]}
            //              [{"title":"diningtable","confidence":0.737176,"x":343,"y":415,"w":135,"h":296},{"title":"chair","confidence":0.641756,"x":338,"y":265,"w":75,"h":57},{"title":"chair","confidence":0.565877,"x":442,"y":420,"w":84,"h":421}]
            //              cam: 39 {"yuvtoJPG":8,"url2":15,"url3":128,"d":152}"
            JSONObject json_resp = new JSONObject(response.toString());
            JSONArray json = json_resp.getJSONArray("results");
            int i; JSONObject obj;
            for (i = 0; i < json.length(); i++) {
                obj = json.getJSONObject(i);
                String title = obj.getString("title");
                Float confidence = (float) obj.getDouble("confidence");
                Float x = (float) obj.getInt("x");
                Float y = (float) obj.getInt("y");
                Float w = (float) obj.getInt("w");
                Float h = (float) obj.getInt("h");
                RectF location = new RectF(
                        Math.max(0, x - w / 2),  // left
                        Math.max(0, y - h / 2),  // top
                        Math.min(dst_w - 1, x + w / 2),  //right
                        Math.min(dst_h - 1, y + h / 2));  // bottom



                viewToFrameTransform.mapRect(location); // map boxes back to original image coords
                Recognition result = new Recognition(title, confidence, location, new Size(image_w, image_h));
                detects.results.add(result);
                recArr.add(result);
                String sss = String.valueOf(response);
                Log.e("MB", sss);

//
//
//
//                byte[] r = Transform.NV21toJPEG(yuv, image_w, image_h, 100);
//
//                Bitmap origialBitmap = BitmapFactory.decodeByteArray(r, 0, r.length);
//                int wid = Math.round(location.right-location.left);
//                int hei = Math.round(location.bottom-location.top);
//                Bitmap resultBmp = Bitmap.createBitmap(wid, hei, Bitmap.Config.ARGB_8888);
//                bitmaps.add(resultBmp);

            }
            detects.server_timings = json_resp.getJSONObject("server_timings");
        } catch(Exception e) {
            Logger.addln("\nWARN Problem reading JSON:  "+response+" ("+e.getMessage()+")");
        }
        detects.addTiming("d",Logger.tockLong("d"));
        return detects;
    }

    /***************************************************************************************/
    private boolean hasPermission(Context context) {
        return ContextCompat.checkSelfPermission(context, Manifest.permission.INTERNET)
                == PackageManager.PERMISSION_GRANTED;
     }

    private void requestPermission(final Activity activity) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (ActivityCompat.shouldShowRequestPermissionRationale(activity,Manifest.permission.INTERNET)) {
                // send message to user ...
                activity.runOnUiThread(
                        new Runnable() {
                            @Override
                            public void run() {
                                Toast.makeText(activity,
                                        "Internet permission is required to use YoloHTTP",
                                        Toast.LENGTH_SHORT).show();
                            }
                        });
            }
            ActivityCompat.requestPermissions(activity,new String[]{Manifest.permission.INTERNET},
                    2);
            // will enter onRequestPermissionsResult() callback in class cameraFragment following
            // user response to permissions request (bit messy that its hidden inside that class,
            // should probabyl tidy it up).

        }
    }
    //  SAVE FILE, FOR TESTING USAGE
    //HISTO IS DIFFERENT THRESH VALUES
    //CONTOURS IS DRAWN CONTOURS

    //Testing for two levels of blurred (13,13)(2,2) saved in lookatblurredimage AND (7,7)(0) saved in lookatblurredimages2


    public static void saveBitmapToExternalStorage(Bitmap b){
        try {
            String root = Environment.getExternalStorageDirectory().toString();
            File myDir = new File(root + "/whatIsServerSeeingUmbrella10??");
            myDir.mkdirs();
            Random generator = new Random();
            int n = 10000;
            n = generator.nextInt(n);
            String fname = "Image-"+ n +".jpeg";
            File file = new File (myDir, fname);
            try {
                FileOutputStream outF = new FileOutputStream(file);
                b.compress(Bitmap.CompressFormat.JPEG, 100, outF);
                outF.close();

            } catch (Exception e) {
               Log.e("HELP", "NOT SAVED");
            }
        }
            catch (Exception e){
                Log.e("HELP", "Nope");
            }
    }
    private void removeDuplicates(ArrayList<Bitmap> arr){
        for(int i = 0 ; i < arr.size() ; i++){
            for(int j = 1 ; j < arr.size() ; j++){
                String s1 = String.valueOf(arr.get(i).getByteCount());
                String s2 = String.valueOf(arr.get(j).getByteCount());
                if(!s1.equals(s2)){
                    Log.e("Lmao", "Non Duplicate: ");
                }
                else{
                    arr.remove(j);
                }
            }
        }
//        while(i1 <= coco.size()-1){
//            i2 = i1+1;
//            String s1 = String.valueOf(coco.get(i1).getByteCount());
//            String s2 = String.valueOf(coco.get(i2).getByteCount());
//
//            if(!s1.equals(s2)){
//                Log.e("Lmao", "Non Duplicate: ");
//            }
//            else{
//                Log.e("Lmao", "Duplicate Location: " );
//            }
//            i1++;
//        }
    }



    /***************************************************************************************/
    // debugging
    private static class Debug {
        static void println(String s) {
            if (DEBUGGING) System.out.println("YoloHTTP: "+s);
        }
    }
    private Bitmap saveCoco(int imgID) {
        Drawable drawable = context.getResources().getDrawable(imgID);
        Bitmap bitmap = ((BitmapDrawable) drawable).getBitmap();
        return bitmap;
    }

    public Bitmap try2(ArrayList<Bitmap> bitmap){
        //TODO ADD B IN RANDOM SPOT

        int w = outputHW * 2, h = outputHW * bitmap.size()/2;
        Bitmap bigbitmap    = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
        Canvas bigcanvas    = new Canvas(bigbitmap);

        Paint paint = new Paint();
        int iWidth = 0;
        int iHeight = 0;
        Bitmap bmp;
        for (int i = 0; i < bitmap.size(); i++) {
            bmp = bitmap.get(i);
            bmp = getResizedBitmap(bmp, outputHW, outputHW);
            if(iWidth < 1000) {
                bigcanvas.drawBitmap(bmp, iWidth , iHeight, paint);
                iWidth += bmp.getWidth();

            }
            else {
                iWidth = 0;
                bigcanvas.drawBitmap(bmp, iWidth, iHeight, paint);
                iHeight += bmp.getHeight();
            }
        }
        return bigbitmap;
    }
    public Bitmap tryLists(ArrayList<ArrayList<Bitmap>> bitmap){
        //TODO ADD B IN RANDOM SPOT
        Bitmap bitm = null;
        int totalSize = 0;
        for(int n = 0 ; n < bitmap.size() ; n++){
            removeDuplicates(bitmap.get(n));
            totalSize += bitmap.get(n).size();
        }

        Log.e("SizeCheck", String.valueOf(totalSize));
        int w = outputHW * totalSize, h = outputHW*10;
        Bitmap bigbitmap    = Bitmap.createBitmap(w/10, h, Bitmap.Config.ARGB_8888);
        Canvas bigcanvas    = new Canvas(bigbitmap);

        Paint paint = new Paint();
        int iWidth = 0;
        int iHeight = 0;
        Bitmap bmp;
        ArrayList myList;
        for(int i = 0; i < bitmap.size(); i++){

            for(int j = 0; j < bitmap.get(i).size(); j++){
                bmp = bitmap.get(i).get(j);
                bmp = getResizedBitmap(bmp, outputHW,outputHW);
                if(iWidth < w/10) {
                    bigcanvas.drawBitmap(bmp, iWidth , iHeight, paint);
                    iWidth += bmp.getWidth();

                }
                else {
                    iWidth = 0;
                    bigcanvas.drawBitmap(bmp, iWidth, iHeight, paint);
                    iHeight += bmp.getHeight();
                }

            }

        }
//        for (int i = 0; i < bitmap.size(); i++) {
//            for (int j = 0; j < bitmap.get(i).size(); j++) {
//                removeDuplicates(bitmap.get(i));
//                bmp = bitmap.get(i).get(j);
//                bmp = getResizedBitmap(bmp, outputHW,outputHW);
//                if(iWidth < w/2) {
//                    bigcanvas.drawBitmap(bmp, iWidth , iHeight, paint);
//                    iWidth += bmp.getWidth();
//
//                }
//                else {
//                    iWidth = 0;
//                    bigcanvas.drawBitmap(bmp, iWidth, iHeight, paint);
//                    iHeight += bmp.getHeight();
//                }
//            }

        return bigbitmap;
    }


    public static byte[] bitmapToByteArray(Bitmap bitmap){
        ByteArrayOutputStream stream = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream);
        byte[] byteArray = stream.toByteArray();
        return byteArray;

    }
    public Bitmap[] splitBitmap(Bitmap picture)
    {
        Bitmap[] imgs = new Bitmap[1];
        imgs[0] = Bitmap.createBitmap(picture, 0, 0, picture.getWidth()/2 , picture.getHeight()/2);
        return imgs;
    }
    public void tstSave(ArrayList<Bitmap> bms, Bitmap b){
        bms.add(b);
        for(int i = 0 ; i < bms.size(); i++){
            saveBitmapToExternalStorage(bms.get(i));
        }
    }

    public Bitmap getResizedBitmap(Bitmap bm, int newWidth, int newHeight) {
        int width = bm.getWidth();
        int height = bm.getHeight();
        float scaleWidth = ((float) newWidth) / width;
        float scaleHeight = ((float) newHeight) / height;
        // CREATE A MATRIX FOR THE MANIPULATION
        Matrix matrix = new Matrix();
        // RESIZE THE BIT MAP
        matrix.postScale(scaleWidth, scaleHeight);

        // "RECREATE" THE NEW BITMAP
   //     Bitmap resizedBitmap = Bitmap.createScaledBitmap(bm, 640,480, true);
        Bitmap resizedBitmap = Bitmap.createBitmap(
                bm, 0, 0, width, height, matrix, true);
    //    bm.recycle();
        return resizedBitmap;
    }


public Bitmap objectDet(Bitmap b, int bWid , int bHei){
    Mat srcMat = new Mat (bHei, bWid, CvType.CV_8U, new Scalar(4));
    Mat gray = new Mat (bHei, bWid, CvType.CV_8U, new Scalar(4));
    Mat clone = srcMat.clone();
    Mat result = new Mat (bHei, bWid, CvType.CV_8U, new Scalar(4));

    Utils.bitmapToMat(b,srcMat);
//THIS IS CONTOUR DETECTION
    Imgproc.cvtColor(srcMat, gray, Imgproc.COLOR_RGBA2GRAY);
    Imgproc.GaussianBlur(gray,gray,new org.opencv.core.Size(13,13),2,2);
    Imgproc.threshold(gray, gray, 120, 255,Imgproc.THRESH_BINARY);
    Imgproc.Canny(gray,gray,50,150);
    // apply erosion and dilation
    Imgproc.dilate(gray, gray, Mat.ones(new org.opencv.core.Size(5, 5), CvType.CV_8UC1));
    Imgproc.erode(gray, gray, Mat.ones(new org.opencv.core.Size(5, 5), CvType.CV_8UC1));
//ffind and draw contours
    ArrayList<MatOfPoint> contours = new ArrayList<MatOfPoint>();
    Mat hierarchy = new Mat();
    //find contours:
    Imgproc.findContours(gray, contours, hierarchy, Imgproc.RETR_TREE,Imgproc.CHAIN_APPROX_SIMPLE);


    double contSize = 0;
    MatOfPoint2f  approxCurve = new MatOfPoint2f();

    for(int i = 0 ; i < contours.size() ; i++){
        contSize = Imgproc.contourArea(contours.get(i));
        if(contSize > 3000){
            //Convert contours(i) from MatOfPoint to MatOfPoint2f
            MatOfPoint2f contour2f = new MatOfPoint2f( contours.get(i).toArray() );
            //Processing on mMOP2f1 which is in type MatOfPoint2f
            double approxDistance = Imgproc.arcLength(contour2f, true)*0.02;
            Imgproc.approxPolyDP(contour2f, approxCurve, approxDistance, true);
            //Convert back to MatOfPoint
            MatOfPoint points = new MatOfPoint( approxCurve.toArray() );
            // Get bounding rect of contour
            Rect rect = Imgproc.boundingRect(points);


            // draw enclosing rectangle (all same color, but you could use variable i to make them unique)
          //  Imgproc.rectangle(srcMat,new Point(rect.x,rect.y), new Point(rect.x+rect.width,rect.y+rect.height), new Scalar(0,255,0), 3);
           //Now to crop
            Rect rectCrop = new Rect(rect.x, rect.y , rect.width, rect.height);
            result = srcMat.submat(rectCrop);


            //cropped.add(bmp);

        }
    }


        Bitmap bitmap = Bitmap.createBitmap(result.cols(), result.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(result, bitmap);


    return bitmap;
  //  return cropped;
}


    public ArrayList<Bitmap> objectDetforArray(Bitmap b, int bWid , int bHei){
        ArrayList<Bitmap> DetectionsCropped = new ArrayList<>();
        Mat srcMat = new Mat (bHei, bWid, CvType.CV_8U, new Scalar(4));
        Mat gray = new Mat (bHei, bWid, CvType.CV_8U, new Scalar(4));
        Mat clone = srcMat.clone();
        Mat result = new Mat (bHei, bWid, CvType.CV_8U, new Scalar(4));

        Utils.bitmapToMat(b,srcMat);
//THIS IS CONTOUR DETECTION
        Imgproc.cvtColor(srcMat, gray, Imgproc.COLOR_RGBA2GRAY);
        Imgproc.GaussianBlur(gray,gray,new org.opencv.core.Size(13,13),2,2);
        Imgproc.threshold(gray, gray, 120, 255,Imgproc.THRESH_BINARY);
        Imgproc.Canny(gray,gray,50,150);
        // apply erosion and dilation
        Imgproc.dilate(gray, gray, Mat.ones(new org.opencv.core.Size(5, 5), CvType.CV_8UC1));
        Imgproc.erode(gray, gray, Mat.ones(new org.opencv.core.Size(5, 5), CvType.CV_8UC1));
//ffind and draw contours
        ArrayList<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Mat hierarchy = new Mat();
        //find contours:
        Imgproc.findContours(gray, contours, hierarchy, Imgproc.RETR_TREE,Imgproc.CHAIN_APPROX_SIMPLE);


        double contSize = 0;
        MatOfPoint2f  approxCurve = new MatOfPoint2f();

        for(int i = 0 ; i < contours.size() ; i++){
            contSize = Imgproc.contourArea(contours.get(i));
            if(contSize > 3000){
                //Convert contours(i) from MatOfPoint to MatOfPoint2f
                MatOfPoint2f contour2f = new MatOfPoint2f( contours.get(i).toArray() );
                //Processing on mMOP2f1 which is in type MatOfPoint2f
                double approxDistance = Imgproc.arcLength(contour2f, true)*0.02;
                Imgproc.approxPolyDP(contour2f, approxCurve, approxDistance, true);
                //Convert back to MatOfPoint
                MatOfPoint points = new MatOfPoint( approxCurve.toArray() );
                // Get bounding rect of contour
                Rect rect = Imgproc.boundingRect(points);


                // draw enclosing rectangle (all same color, but you could use variable i to make them unique)
                //  Imgproc.rectangle(srcMat,new Point(rect.x,rect.y), new Point(rect.x+rect.width,rect.y+rect.height), new Scalar(0,255,0), 3);
                //Now to crop
                Rect rectCrop = new Rect(rect.x, rect.y , rect.width, rect.height);
                result = srcMat.submat(rectCrop);
                Bitmap bitmap = Bitmap.createBitmap(result.cols(), result.rows(), Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(result, bitmap);
                DetectionsCropped.add(bitmap);

            }
        }


        return DetectionsCropped;
        //  return cropped;
    }

    public Bitmap resizeImageForImageView(Bitmap bitmap) {
        int width = 640;
        int height = 480;
        Bitmap background = Bitmap.createBitmap((int) width, (int) height, Bitmap.Config.ARGB_8888);

        float originalWidth = bitmap.getWidth();
        float originalHeight = bitmap.getHeight();

        Canvas canvas = new Canvas(background);

        float scale = width / originalWidth;

        float xTranslation = 0.0f;
        float yTranslation = (height - originalHeight * scale) / 2.0f;

        Matrix transformation = new Matrix();
        transformation.postTranslate(xTranslation, yTranslation);
        transformation.preScale(scale, scale);

        Paint paint = new Paint();
        paint.setFilterBitmap(true);

        canvas.drawBitmap(bitmap, transformation, paint);

        return background;
    }

    public Bitmap actuallyKillMe(Bitmap b){
        int reqWidth = 640;
        int reqHeight = 480;
        Matrix m = new Matrix();
        m.setRectToRect(new RectF(0, 0, b.getWidth(), b.getHeight()), new RectF(0, 0, reqWidth, reqHeight), Matrix.ScaleToFit.CENTER);
        return Bitmap.createBitmap(b, 0, 0, b.getWidth(), b.getHeight(), m, true);
    }


}


