/*
in this sample we will implement binary segmentation using threshold 
*/
#include "Halide.h"
#include "halide_image_io.h"
#include <iostream>
// #include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace Halide::Tools;
using namespace cv;


// High-speed Test： 用來加速剔除非 corner 的像素。
// 這個方法會先檢查位於 1, 9, 5, 13 的像素值 (先是 1, 9，如果都大於或都小於，才會繼續檢查位於 5, 13)，如果 P 是 corner，則這四個像素中，至少會有三個的像素值為都大於 (Ip + t) 或都小於 (Ip - t) ，否則 P 不為 corner，判斷可能是 corner 之後，才會對周圍的 16 個像素作完整的判斷。
// 雖然 High-speed Test 提高了效率，不過還是有一些缺點：
// 1, 如果 n < 12，則效率會變差。
// 2, 關於所拿來測試的 4 個像素的位置選擇，並沒有作最佳化。
// 3, High-speed Test 的結果，在判斷完是否可能為 corner 之後，即沒再作利用。
// 4, 對同一個區域，可能會偵測到多個重疊的 feature。


// 對於上述的前三項缺點，可以用 machine learning 的方式來作最佳化，第四項則可以採用 non-maximal suppression。



// non-maximal suppression 的作法為：
// 對所有已經判斷為 feature 的像素打分數，分數為該像素的像素值與鄰近 16 個像素的像素值之差的總和。
// 如果有兩個為相鄰的 feature，則比較他們的分數。
// 捨棄分數較低的 feature。



void HalideBuffer2Mat(const Halide::Buffer<uint8_t> &src, cv::Mat &dest, bool RGBLayout=true)
{
    if (dest.empty())
        dest.create(cv::Size(src.width(), src.height()), CV_MAKETYPE(CV_8U, src.channels()));

    const int ch = dest.channels();
    switch (ch)
    {
    case 1:
        for (int j = 0; j < dest.rows; j++)
        {   
            uchar* ptr = dest.ptr<uchar>(j);            
            for (int i = 0; i < dest.cols; i++)
            {
                // dest.at<uchar>(j, i) = src(i, j);
                ptr[i] = src(i, j);

            }
        }
        break;
    case 3:
        for (int j = 0; j < dest.rows; j++)
        {   
            uchar* ptr = dest.ptr<uchar>(j);
            for (int i = 0; i < dest.cols; i++)
            {     
                /*set RGBLayout condition*/
                if(RGBLayout){
                    ptr[i * 3 + 0 ] = src(i, j, 2);
                    ptr[i * 3 + 1 ] = src(i, j, 1);
                    ptr[i * 3 + 2 ] = src(i, j, 0);
                }
                if(!RGBLayout){
                    ptr[i * 3 + 0 ] = src(i, j, 0);
                    ptr[i * 3 + 1 ] = src(i, j, 1);
                    ptr[i * 3 + 2 ] = src(i, j, 2);
                }
            }
        }
        break;
    }
}

int main(int argc, char **argv) {

    //Load image
    cv::Mat image = cv::imread("Img/AWB_before1.jpg");
    Halide::Buffer<uint8_t> input = Halide::Buffer<uint8_t>::make_interleaved(image.data,image.cols, image.rows, image.channels());
    
    Halide::Var x,y,c;


    uint16_t R2GRAY = 77, G2GRAY = 150, B2GRAY = 29;
    Halide::Func gray;
    Halide::Expr r = Halide::cast<uint16_t>(input(x, y, 0));
    Halide::Expr g = Halide::cast<uint16_t>(input(x, y, 1));
    Halide::Expr b = Halide::cast<uint16_t>(input(x, y, 2));
    gray(x, y) = Halide::cast<uint8_t>((r * R2GRAY +
                                        g * G2GRAY + 
                                        b * B2GRAY) >> 8 );

 
    Halide::Func clamped;
    Halide::Expr clamped_x = clamp(x, 0, input.width() - 1);
    Halide::Expr clamped_y = clamp(y, 0, input.height() - 1);
    // Load from input at the clamped coordinates. This means that
    // no matter how we evaluated the Func 'clamped', we'll never
    // read out of bounds on the input. This is a clamp-to-edge
    // style boundary condition, and is the simplest boundary
    // condition to express in Halide.
    clamped(x, y) = gray(clamped_x, clamped_y);

    Halide::Func HighSpeedTestMask;
    int threshold =30; 
    int radius = 3;
    Halide::Expr p1 = select(Halide::absd(clamped(x-radius,y) , clamped(x,y)) > threshold, 1,0);
    Halide::Expr p2 = select(Halide::absd(clamped(x+radius,y) , clamped(x,y)) > threshold, 1,0);
    Halide::Expr p3 = select(Halide::absd(clamped(x,y-radius) , clamped(x,y)) > threshold, 1,0);
    Halide::Expr p4 = select(Halide::absd(clamped(x,y+radius) , clamped(x,y)) > threshold, 1,0);


    Halide::Expr p1v = select(Halide::absd(clamped(x-radius,y) , clamped(x,y)) > threshold, threshold,0);
    Halide::Expr p2v = select(Halide::absd(clamped(x+radius,y) , clamped(x,y)) > threshold, threshold,0);
    Halide::Expr p3v = select(Halide::absd(clamped(x,y-radius) , clamped(x,y)) > threshold, threshold,0);
    Halide::Expr p4v = select(Halide::absd(clamped(x,y+radius) , clamped(x,y)) > threshold, threshold,0);
     
    HighSpeedTestMask(x,y) = Halide::cast<uint8_t>(p1+p2+p3+p4);
    HighSpeedTestMask(x,y) = Halide::cast<uint8_t>(select(HighSpeedTestMask(x,y) >= 3 , 255,0));
    // HighSpeedTestMask(x,y) = Halide::cast<uint8_t>(select(HighSpeedTestMask(x,y) >= 0 , 255, 0));
    


    Halide::Func CalValue;
    CalValue(x,y) = select(HighSpeedTestMask(x,y) == 255 ,p1v+p2v+p3v+p4v, 0);
    
    Halide::Func NonMaximalSup;
    NonMaximalSup(x,y) = Halide::cast<uint8_t>(0);
    Halide::RDom r_(-2,5,-2,5);
    NonMaximalSup(x,y) += Halide::cast<uint8_t>(select(CalValue(x,y)> 0 && CalValue(x,y) > CalValue(x+r_.x, y+r_.y), 1, 0));
    NonMaximalSup(x,y) = Halide::cast<uint8_t>(select(NonMaximalSup(x,y) >20 , 255, 0));
    NonMaximalSup(x,y) = Halide::cast<uint8_t>(NonMaximalSup(x,y));

    Halide::Buffer<uint8_t> result(input.width(), input.height());
    NonMaximalSup.realize(result);
    
 
    /*
    -3 -1 
    -3 0
    -3 1
    -2 -2
    -2 2
    -1 -3
    -1 3
    0 -3
    0 3
    1 -3
    1 3
    2 -2
    2 2
    3 -1
    3 0
    3 1
    */

    imshow("ORI img" , image);
    Mat Result;
    HalideBuffer2Mat(result,Result,false);
    imshow("Sobel img", Result);

   
    waitKey(0);
  
    return 0;
}
