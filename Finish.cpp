#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <gsl/gsl_randist.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_rng.h>


const double w0=0.5,w1=-0.5,s0=0.5,s1=0.5;
using namespace cv;
quint8 attack = 0;
int Q=30, angle = 45, Cr=32, pop = 10, it = 1, mbin = 1, opt = 1;
double white=0.002, black = 0.998, mat = 0, Otk = 0.03, MedF=3;
 int rand_position;
 int Cuckoo = 0;
 double NCValue;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

void MainWindow::DecomposeMatrix(float **MatrixLH, float **MatrixHL,float **MatrixHH,float **source )
{
    for (int x=0;x<WidthOfImage/2;x++)
          for (int y=0;y<HeightOfImage/2;y++)
          {
               MatrixLH[x][y] = source[x][y+HeightOfImage/2];
               MatrixHL[x][y] = source[x+WidthOfImage/2][y];
               MatrixHH[x][y] = source[x+WidthOfImage/2][y+HeightOfImage/2];
          }
}

void MainWindow::CompileMatrix(float **MatrixLH, float **MatrixHL,float **MatrixHH,float **source )
{
    for (int x=0;x<WidthOfImage/2;x++)
          for (int y=0;y<HeightOfImage/2;y++)
          {
               source[x][y+HeightOfImage/2] = MatrixLH[x][y];
               source[x+WidthOfImage/2][y]= MatrixHL[x][y];
               source[x+WidthOfImage/2][y+HeightOfImage/2]= MatrixHH[x][y];
          }
}

void SaltAndPepper(float** cordata, int H, int W)
{
    double temp;
    std::mt19937 gen(time(0));
    std::uniform_real_distribution<> urd(0, 1);
    for (int i=0; i<H; i++)
        for (int j=0; j<W; j++)
        {
            temp = urd(gen);
          if (temp <= white && white != 0)
            cordata[i][j] = 255;
          else if (temp >= black && black != 1)
              cordata[i][j] = 0;
        }
}

void GausNoise(float** cordata, int H, int W)
{
    Mat imGray = Mat(H,W,CV_64F);
    for (int i=0;i<H;i++)
        for (int j=0;j<W;j++)
            imGray.at<double>(H*i+j)=cordata[i][j];
    Mat noise = Mat(imGray.size(),CV_64F);
    Mat result;
    normalize(imGray, result, 0.0, 1.0, CV_MINMAX, CV_64F);
    randn(noise, mat, Otk);
    result = result + noise;
    normalize(result, result, 0.0, 1.0, CV_MINMAX, CV_64F);
    result.convertTo(result, CV_64F, 255, 0);
    for (int i=0;i<H;++i)
        for (int j=0;j<W;++j)
            cordata[i][j]=result.at<double>(H*i+j);
    imGray.release();
    noise.release();
    result.release();
}

void Rotation(float** cordata, int H, int W)
{
    Mat src = Mat(H,W,CV_64F);
    for (int i=0;i<H;i++)
        for (int j=0;j<W;j++)
            src.at<double>(H*i+j)=cordata[i][j];
    Point2f pc(src.cols/2.-1, src.rows/2.-1);
    Mat r = getRotationMatrix2D(pc, 0-angle, 1.0);
    warpAffine(src, src, r, src.size());
    for (int i=0;i<H;i++)
        for (int j=0;j<W;j++)
            cordata[i][j]=src.at<double>(H*i+j);
    src.release();
    r.release();
}

void MedianFilter(float** cordata, int H, int W)
{
    Mat src = Mat(H,W,CV_32F);
    Mat dst;
    for (int i=0;i<H;i++)
        for (int j=0;j<W;j++)
            src.at<float>(H*i+j)=cordata[i][j];
    medianBlur(src,dst,MedF);
    for (int x=0;x<H;x++)
        for (int y=0;y<W;y++)
           cordata[x][y] = dst.at<float>(H*x+y);
    src.release();
    dst.release();
}

void Cropping(float** cordata, int H, int W)
{
    for (int x=0;x<H/Cr;x++)
        for (int y=0;y<W/Cr;y++)
        {
            cordata[H/4+x][W/2+y]=0;
            cordata[H/4+x][W/2-y]=0;
            cordata[H/4-x][W/2+y]=0;
            cordata[H/4-x][W/2-y]=0;
        }
}

void JpegComp(float** cordata, int H, int W)
{
    QImage i(W,H, QImage::Format_RGB32);
    for (int x=0;x<H;x++)
        for (int y=0;y<W;y++)
            i.setPixel(y,x,qRgb(cordata[x][y],cordata[x][y],cordata[x][y]));
    i.save("JpegComp.jpg","JPG", 100-Q);
    QImage i2("JpegComp.jpg");
    for (int x=0;x<H;x++)
        for (int y=0;y<W;y++)
        {   QColor color(i2.pixel(y,x));
            int G=color.green();
            cordata[x][y]=G;
        }
}

void MainWindow::on_ButtonLoadImage_clicked()
{
    PathToImage = QFileDialog::getOpenFileName(0, "Open Dialog", "", "*.jpg *.png" );
    image = new QImage(PathToImage);
    WaterMarked = new QImage(PathToImage);
    WidthOfImage = image->width();
    HeightOfImage = image->height();

    ui->progressBar->setMaximum(number_of_generations*number_of_nests);
    ui->progressBar->reset();
    ui->progressBar->value();
    WaterMark = new QImage("/home/hunk/123.png");
   *WaterMark = WaterMark->convertToFormat(QImage::Format_Mono);

    number_of_nests = ui->SB_number_of_nests->value();
    nest_size = ui->SB_Size_Of_Nest->value();
    step_size = 1;
    number_of_generations = ui->SB_Generations->value();

    WaterWidth = WaterMark->width();
    WaterHeight = WaterMark->height();

    dataImage=new float*[WidthOfImage];
    float **olddata = new float*[WidthOfImage];
    for(int i=0;i<WidthOfImage;i++)
        {
            dataImage[i]=new float[HeightOfImage];
            olddata[i] = new float[HeightOfImage];
        }

    colortogray(image,olddata,HeightOfImage,WidthOfImage);
    colortogray(WaterMarked,dataImage,HeightOfImage,WidthOfImage);

    WaterMark_Vector = new int[WaterWidth * WaterHeight];
    MatrixToVector(WaterMark,WaterWidth,WaterHeight,WaterMark_Vector);

    MatrixHL = new float*[WidthOfImage/2];
    MatrixLH = new float*[WidthOfImage/2];
    MatrixHH = new float*[WidthOfImage/2];
    for(int i = 0; i < WidthOfImage/2;i++)
        {
            MatrixLH[i] = new float[HeightOfImage/2];
            MatrixHL[i] = new float[HeightOfImage/2];
            MatrixHH[i] = new float[HeightOfImage/2];
        }

       blocks = new int[WaterWidth*WaterHeight];

        for(int i = 0;i<WaterWidth*WaterHeight;i++)
            blocks[i] = 0;

       ResultVector = new int[WaterWidth*WaterHeight];
}

void MainWindow::on_buttonSetView_clicked()
{

}

void MainWindow::colortogray(QImage* a, float** b, int n, int m)
{
    for (int x=0;x<m;x++)
        for (int y=0;y<n;y++)
        {   QColor color(a->pixel(x,y));
            int G=color.green();
            int B=color.blue();
            int R=color.red();
            int Grey=0.299*R+0.587*G+0.114*B;
            a->setPixel(x,y,qRgb(Grey,Grey,Grey));
            b[x][y]=Grey;
        }
}
void MainWindow::FWT(int* mas, int m)
{
    double* temp = new double[m];

    int h = m >> 1;
    for (int i = 0; i < h; i++)
    {
        int k = (i << 1);
        temp[i] = mas[k] * s0 + mas[k + 1] * s1;
        temp[i + h] = mas[k] * w0 + mas[k + 1] * w1;
    }

    for (int i = 0; i < m; i++)
        mas[i] = temp[i];
}
void MainWindow::FWT(float** matr, int n, int m, int iterations)
{
    int rows = n;
    int cols = m;

    int* row;
    int* col;

    for (int k = 0; k < iterations; k++)
    {
        int lev = 1 << k;

        int levCols = cols / lev;
        int levRows = rows / lev;

        row = new int[levCols];
        for (int i = 0; i < levRows; i++)
        {
            for (int j = 0; j < levCols; j++)
                row[j] = matr[i][j];

            FWT(row, levCols);

            for (int j = 0; j < levCols; j++)
                matr[i][j] = row[j];
        }


        col = new int[levRows];
        for (int j = 0; j < levCols; j++)
        {
            for (int i = 0; i < levRows; i++)
                col[i] = matr[i][j];

            FWT(col,levRows);

            for (int i = 0; i < levRows; i++)
                matr[i][j] = col[i];
        }
    }
}
void MainWindow::IWT(int* mas,int m)
{
    int* temp = new int[m];

    int h = m >> 1;
    for (int i = 0; i < h; i++)
    {
        int k = (i << 1);
        temp[k] = (mas[i] * s0 + mas[i + h] * w0) / w0;
        temp[k + 1] = (mas[i] * s1 + mas[i + h] * w1) / s0;
    }

    for (int i = 0; i < m; i++)
        mas[i] = temp[i];
}
void MainWindow::IWT(float** matr, int n, int m, int iterations)
{
    int rows = n;
    int cols = m;
    int* col;
    int* row;

    for (int k = iterations - 1; k >= 0; k--)
    {
        int lev = 1 << k;

        int levCols = cols / lev;
        int levRows = rows / lev;

        col = new int[levRows];
        for (int j = 0; j < levCols; j++)
        {
            for (int i = 0; i < levRows; i++)
                col[i] = matr[i][j];

            IWT(col, levRows);

            for (int i = 0; i < levRows; i++)
                matr[i][j] = col[i];
        }

        row = new int[levCols];
        for (int i = 0; i < levRows; i++)
        {
            for (int j = 0; j < levCols; j++)
                row[j] = matr[i][j];

            IWT(row, levCols);

            for (int j = 0; j < levCols; j++)
                matr[i][j] = row[j];
        }
    }
}

void MainWindow::MatrixToVector(QImage* image,int maxW, int maxH, int *vector)
 {
    int k = 0;
    for(int x = 0; x < maxW; x++)
        for(int y = 0; y<maxH;y++)
            {
                if(WaterMark->pixel(x,y) == 4294967295)
                    vector[k++] = 1;
                else
                    vector[k++] = 0;
            }
//         qDebug()<<vector[0] << "  "<< vector[1];
}

void MainWindow::randomization(int size, int* mas, int pixel)
{
    rand_position = qrand()%(size-1);//uid1(gen1);
        for(int i = 0; i<pixel;i++)
            {
            if(mas[i] == rand_position)
            {
                randomization(size, mas, pixel);
            }
            }
            mas[pixel] = rand_position;
}

void bubbleSort(int* arrayPtr, int length_array) // сортировка пузырьком
{
 int temp = 0; // временная переменная для хранения элемента массива
 bool exit = false; // болевая переменная для выхода из цикла, если массив отсортирован

 while (!exit) // пока массив не отсортирован
 {
  exit = true;
  for (int int_counter = 0; int_counter < (length_array - 1); int_counter++) // внутренний цикл
    //сортировка пузырьком по возрастанию - знак >
    //сортировка пузырьком по убыванию - знак <
    if (arrayPtr[int_counter] < arrayPtr[int_counter + 1]) // сравниваем два соседних элемента
    {
     // выполняем перестановку элементов массива
     temp = arrayPtr[int_counter];
     arrayPtr[int_counter] = arrayPtr[int_counter + 1];
     arrayPtr[int_counter + 1] = temp;
     exit = false; // на очередной итерации была произведена перестановка элементов
    }
 }
}

double MainWindow::MSE(int height, int width, float** Source, float** Watermarked)
{
    double sum = 0;
    for(int x = 0; x < width; x++)
        for(int y = 0; y < height; y++)
        {
            sum += pow(Source[x][y] - Watermarked[x][y],2.0f);
        }
    int Delitel = height*width;
    sum = (sum / Delitel*1.0f);
    return sum;
}

double MainWindow::PSNR(int height, int width, float** Source, float** Watermarked){
            double PSNR_Value = 0.f;
            double value_MSE = MSE(height, width, Source, Watermarked);
            PSNR_Value = 10.0f * (log10(65025/value_MSE*1.f))*1.0f;
            return PSNR_Value*1.0f;
}

double MainWindow::NC(int *SourceVector, int *ChangedVector,int n)
{
    double SumS = 0;
    double SumC = 0;
    double Multipl = 0;
    for(int i = 0; i<n; i++)
    {
        SumS += pow(SourceVector[i],2);
        SumC += pow(ChangedVector[i],2);
        Multipl += SourceVector[i]*ChangedVector[i];
    }
    return Multipl/(sqrt(SumS)*sqrt(SumC));
}
void MainWindow::Embeded_CVZ_With_CuckooSearch(int heightWater, int widthWater, int width, int height, float **MatrixLH,float **MatrixHL,float **MatrixHH,float** source,int* blocks, int *vector)
{
    QString P;
    double tempPSNR = 0.f;
    double Best_Psnr = 0.f;
    const gsl_rng_type * T;
     gsl_rng * r;

     gsl_rng_env_setup();

     T = gsl_rng_default;
     r = gsl_rng_alloc (T);

     int rand_nest = 0;


     double* PSNRValues = new double[number_of_nests];

     for(int i = 0; i<number_of_nests;i++)
         PSNRValues[i] = 0;

    int* ResultBlocksPositions = new int[widthWater*heightWater];

    float **temp = new float*[width];
        for(int i = 0; i<width; i++)
            temp[i] = new float[height];

        float **temp_blocks = new float*[number_of_nests];
            for(int i = 0; i< number_of_nests; i++)
                temp_blocks[i] = new float[widthWater*heightWater];

    float **generation_blocks = new float*[number_of_generations];
        for(int i = 0; i< number_of_generations; i++)
            generation_blocks[i] = new float[widthWater*heightWater];

    double* BestOfGenerationPSNR = new double[number_of_generations];
    int best_index_of_nests;
    int best_generation;

    double BestPSNR = 0;

        int cuckoo = 0;
        for(int i = 0; i< width; i++)
            for(int j = 0; j< height; j++)
            {
                     temp[i][j] = source[i][j];
            }
        if(Cuckoo == 0)
        {
               for(int i = 0; i< width; i++)
                   for(int j = 0; j< height; j++)
                   {
                            temp[i][j] = source[i][j];
                   }
               for(int i = 0; i< heightWater*widthWater;i++)
                   blocks[i] = 0;
                    DecomposeMatrix(MatrixLH,MatrixHL,MatrixHH,source);
                    for(int PixelOfWatermark = 0; PixelOfWatermark<widthWater*heightWater; PixelOfWatermark++)
                    {
                        randomization(heightWater*widthWater*3,blocks,PixelOfWatermark);
                    }
                    MatrixTransform(MatrixLH,MatrixHL,MatrixHH,vector,blocks, widthWater,1.95,2.0,2.55);
                    CompileMatrix(MatrixLH,MatrixHL,MatrixHH,temp);
                    Best_Psnr = PSNR(HeightOfImage,WidthOfImage,source,temp);
                        for(int i = 0;i<heightWater*widthWater;i++)
                            ResultBlocksPositions[i] = blocks[i];
                            P = QString("%2").arg(Best_Psnr);
                            ui->l_PSNR->setText(P);
                           // MatrixTransform(MatrixLH,MatrixHL,MatrixHH,vector, ResultBlocksPositions, widthWater,1.95,2.0,2.55);
                            //CompileMatrix(MatrixLH,MatrixHL,MatrixHH,source);
        }
        else
        {
            ui->progressBar->reset();
            for(int i = 0; i< heightWater*widthWater;i++)
            {
                blocks[i] = 0;
                ResultBlocksPositions[i] = 0;
            }
                DecomposeMatrix(MatrixLH,MatrixHL,MatrixHH,source);
                 for(int PixelOfWatermark = 0; PixelOfWatermark<widthWater*heightWater; PixelOfWatermark++)
                 {
                     randomization(heightWater*widthWater*3,ResultBlocksPositions,PixelOfWatermark);
                 }
                 MatrixTransform(MatrixLH,MatrixHL,MatrixHH,vector,ResultBlocksPositions, widthWater,1.95,2.0,2.55);
                 CompileMatrix(MatrixLH,MatrixHL,MatrixHH,temp);
                 Best_Psnr = PSNR(HeightOfImage,WidthOfImage,source,temp);
            for(int generation = 0; generation < number_of_generations; generation++)
            {
               // DecomposeMatrix(MatrixLH,MatrixHL,MatrixHH,temp);
                for(int nest = 0; nest < number_of_nests;nest++)
                {
                    for(int i = 0; i < width; i++)
                        for(int j = 0; j < height; j++)
                        {
                                 temp[i][j] = source[i][j];
                        }
                    for(int i = 0; i< heightWater*widthWater;i++)
                        blocks[i] = 0;
                         DecomposeMatrix(MatrixLH,MatrixHL,MatrixHH,source);
                         for(int PixelOfWatermark = 0; PixelOfWatermark<widthWater*heightWater; PixelOfWatermark++)
                         {
                             randomization(heightWater*widthWater*3,blocks,PixelOfWatermark);
                         }
                         for(int i = 0; i<widthWater*heightWater; i++)
                            temp_blocks[nest][i] = blocks[i];
                         MatrixTransform(MatrixLH,MatrixHL,MatrixHH,vector,blocks, widthWater,1.95,2.0,2.55);
                         CompileMatrix(MatrixLH,MatrixHL,MatrixHH,temp);
                         PSNRValues[nest] = PSNR(HeightOfImage,WidthOfImage,source,temp);
                         ui->progressBar->setValue(ui->progressBar->value()+1);
                         ui->progressBar->activateWindow();
                }
                rand_nest = qrand()%number_of_nests;
                qDebug()<<"Cuckoo choose "<<rand_nest<<" nest\n";

                for(int i = 0; i<heightWater*widthWater; i++)
                {
                    double levy = gsl_ran_levy(r,250,1);
                    if(levy<0)
                        levy*=-1;
                    ResultBlocksPositions[i] = (int)levy;
                }
                for(int i = 0; i<heightWater*widthWater;i++)
                    {
                        for(int j = 0; j<i; j++)
                            if(ResultBlocksPositions[i] == ResultBlocksPositions[j] || ResultBlocksPositions[i]>=3071 || ResultBlocksPositions[i]<=0)
                            {
                                randomization(heightWater*widthWater*3,ResultBlocksPositions,i);
                            }
                    }
                if(qrand()%100 > ui->SB_propability->value())
                {
                    for(int i = 0; i < width; i++)
                        for(int j = 0; j < height; j++)
                        {
                                 temp[i][j] = source[i][j];
                        }
                         DecomposeMatrix(MatrixLH,MatrixHL,MatrixHH,source);
                         for(int PixelOfWatermark = 0; PixelOfWatermark<widthWater*heightWater; PixelOfWatermark++)
                         {
                             randomization(heightWater*widthWater*3,blocks,PixelOfWatermark);
                         }
                         for(int i = 0; i<widthWater*heightWater; i++)
                            temp_blocks[rand_nest][i] = blocks[i];
                         MatrixTransform(MatrixLH,MatrixHL,MatrixHH,vector,blocks, widthWater,1.95,2.0,2.55);
                         CompileMatrix(MatrixLH,MatrixHL,MatrixHH,temp);
                         PSNRValues[rand_nest] = PSNR(HeightOfImage,WidthOfImage,source,temp);
                }
                else
                {
                    for(int i = 0; i < width; i++)
                        for(int j = 0; j < height; j++)
                        {
                                 temp[i][j] = source[i][j];
                        }
                    MatrixTransform(MatrixLH,MatrixHL,MatrixHH,vector,ResultBlocksPositions, widthWater,1.95,2.0,2.55);
                    CompileMatrix(MatrixLH,MatrixHL,MatrixHH,temp);
                    PSNRValues[rand_nest] = PSNR(HeightOfImage,WidthOfImage,source,temp);
                }
                MatrixTransform(MatrixLH,MatrixHL,MatrixHH,vector, ResultBlocksPositions, widthWater,1.95,2.0,2.55);
                CompileMatrix(MatrixLH,MatrixHL,MatrixHH,temp);
                tempPSNR = PSNR(HeightOfImage,WidthOfImage,source,temp);
                if(tempPSNR >= BestPSNR)
                {
                    for(int i = 0; i<heightWater*widthWater; i++)
                    {
                        blocks[i] = ResultBlocksPositions[i];
                    }
                    BestPSNR = tempPSNR;
                }
                            best_index_of_nests = 0;

                            for(int i = 0; i<number_of_nests;i++)
                                if(PSNRValues[i]>=Best_Psnr)
                                {
                                    Best_Psnr = PSNRValues[i];
                                    best_index_of_nests = i;
                                }
                            if(best_index_of_nests == rand_nest)
                            {
                                P = QString("%2").arg(Best_Psnr);
                                ui->l_PSNR->setText(P);
                                MatrixTransform(MatrixLH,MatrixHL,MatrixHH,vector,ResultBlocksPositions, widthWater,1.95,2.0,2.55);
                                CompileMatrix(MatrixLH,MatrixHL,MatrixHH,temp);
                                for(int i = 0; i<widthWater*heightWater; i++)
                                generation_blocks[generation][i] = ResultBlocksPositions[i];
                                BestOfGenerationPSNR[generation] = Best_Psnr;
                            }
                            else
                            {
                                for(int i = 0; i<widthWater*heightWater; i++)
                                {
                                    ResultBlocksPositions[i] = temp_blocks[best_index_of_nests][i];
                                    generation_blocks[generation][i] = temp_blocks[best_index_of_nests][i];
                                }
                                P = QString("%2").arg(Best_Psnr);
                                BestOfGenerationPSNR[generation] = Best_Psnr;
                                ui->l_PSNR->setText(P);
                                MatrixTransform(MatrixLH,MatrixHL,MatrixHH,vector,ResultBlocksPositions, widthWater,1.95,2.0,2.55);
                                CompileMatrix(MatrixLH,MatrixHL,MatrixHH,temp);
                            }

                        }
                       // Best_Psnr = 0;
                        for(int gen = 0; gen < number_of_generations; gen++)
                        {
                            if(BestOfGenerationPSNR[gen] >= Best_Psnr)
                                best_generation = gen;
                                Best_Psnr = BestOfGenerationPSNR[gen];
                        }

                        for(int i = 0; i<widthWater*heightWater; i++)
                            ResultBlocksPositions[i] = generation_blocks[best_generation][i];
           // DecomposeMatrix(MatrixLH,MatrixHL,MatrixHH,source);
            CompileMatrix(MatrixLH,MatrixHL,MatrixHH,temp);
            P = QString("%2").arg(Best_Psnr);
            ui->l_PSNR->setText(P);
        //    DecomposeMatrix(MatrixLH,MatrixHL,MatrixHH,temp);

        }
        MatrixTransform(MatrixLH,MatrixHL,MatrixHH,vector, blocks, widthWater,1.95,2.0,2.55);
        CompileMatrix(MatrixLH,MatrixHL,MatrixHH,source);
}
void MainWindow::ExtractWatermark(float** source,float **MatrixLH,float **MatrixHL,float **MatrixHH, int* blocks,int *vector, int* ResultVector, int widthWater,int heightWater)
{
    FWT(source,WidthOfImage,HeightOfImage,1);
    QString P;
    DecomposeMatrix(MatrixLH,MatrixHL,MatrixHH,source);
    Average(MatrixLH,MatrixHL,MatrixHH,blocks,ResultVector,widthWater);
    NCValue = NC(vector,ResultVector,widthWater*heightWater);
        P = QString("%2").arg(NCValue);
        ui->l_NC->setText(P);
    extractCVZ = new QImage(widthWater,heightWater,QImage::Format_RGB32);
    QRgb value;
    int equal = 0;
    int x = 0, y = 0;
    for (int i=0;i<widthWater*heightWater;i++)
        if(vector[i] == ResultVector[i])
            equal++;
    qDebug()<<"Equal= "<<equal<<"\n";
    for (int i=0;i<widthWater*heightWater;i++)
        {
            value = qRgb(ResultVector[i]*255,ResultVector[i]*255,ResultVector[i]*255);
            extractCVZ->setPixel(y,x,value);
            x++;
            if(x == 32)
                {
                    x=0;
                    y++;
                }
        }
     extractCVZ->save("/home/hunk/CVZ.png","PNG");
     IWT(source,WidthOfImage,HeightOfImage,1);
}



void MainWindow::Average(float **matrixLH,float **matrixHL,float **matrixHH, int* blocks,int* vector, int sizeWater)
{
    int x,y;
    float average;
    for(int pixel = 0; pixel < sizeWater*sizeWater; pixel++)
    {
        average = 0;
        if(blocks[pixel] < 1024)
        {
            if((blocks[pixel]/sizeWater) != 0)
            {
                x =  (blocks[pixel] * 8) - (sizeWater*8*floor((blocks[pixel]/sizeWater)));
                y = floor((blocks[pixel]/sizeWater)) * 8;
            }
            else
            {
                y = 0;
                x = blocks[pixel]*8;
            }
            {
                for(int i = 0; i<8;i++)
                    average+=matrixLH[x+i][y];
                average = average/8.0f;
                if(matrixLH[x][y]>=average)
                    vector[pixel] = 1;
                else
                    vector[pixel] = 0;
            }
        }
        else if(blocks[pixel] >= 1024 && blocks[pixel] < 2048)
        {
            if((blocks[pixel]-1024)/sizeWater != 0)
            {
                x =  ((blocks[pixel]-1024) * 8) - (sizeWater*8*floor(((blocks[pixel]-1024)/sizeWater)));
                y = floor(((blocks[pixel]-1024)/sizeWater)) * 8;
            }
            else
            {
                y = 0;
                x = (blocks[pixel]-1024)*8;
            }
            {
                for(int i = 0; i<8;i++)
                    average+=matrixHL[x+i][y];
                average = floor(average/8.0f);
                if(matrixHL[x][y]>=average)
                    vector[pixel] = 1;
                else
                    vector[pixel] = 0;
            }
        }
        else if(blocks[pixel] >= 2048 && blocks[pixel] < 3072)
        {
            if((blocks[pixel]-2048)/sizeWater != 0)
            {
                x =  ((blocks[pixel]-2048) * 8) - (sizeWater*8*floor(((blocks[pixel]-2048)/sizeWater)));;
                y =   floor(((blocks[pixel]-2048)/sizeWater)) * 8;
            }
            else
            {
                y = 0;
                x = (blocks[pixel]-2048)*8;
            }
            {
                for(int i = 0; i<8;i++)
                    average+=matrixHH[x+i][y];
                average = average/8.0f;
                if(matrixHH[x][y]>=average)
                    vector[pixel] = 1;
                else
                    vector[pixel] = 0;
            }
        }
    }
    average = 0;
    x=0;
    y=0;
}


void MainWindow::MatrixTransform(float **matrixLH,float **matrixHL,float **matrixHH,int* vector, int* blocks,int sizeWater, float transformLH, float transformHL,float transformHH)
{
    int k = 0;
    int x,y;
    int max,min;
    for(int pixel = 0; pixel < sizeWater*sizeWater; pixel++)
    {
        if(blocks[pixel] == 0)
        {
            x = 0;
            y = 0;
        }
        if(blocks[pixel] < 1024)
        {
            if((blocks[pixel]/sizeWater) != 0)
            {
                x =  (blocks[pixel] * 8) - (sizeWater*8*floor((blocks[pixel]/sizeWater)));
                y = floor((blocks[pixel]/sizeWater)) * 8;
            }
            else
            {
                y = 0;
                x = blocks[pixel]*8;
            }
            if(vector[k++] == 1)
            {
                max = 0;
                for(int i = 0; i<8;i++)
                        if(matrixLH[x+i][y] >=max)
                        {
                            max = matrixLH[x+i][y];
                        }
                matrixLH[x][y] = ceil(max) + transformLH;

            }
            else
            {
                min = matrixLH[x][y];
                for(int i = 0; i<8;i++)
                        if(matrixLH[x+i][y] <=min)
                        {
                            min = matrixLH[x+i][y];
                        }
                matrixLH[x][y] = floor(min) - transformLH;
            }
        }
            if(blocks[pixel] >= 1024 && blocks[pixel] < 2048)
            {
                if((blocks[pixel]-1024)/sizeWater != 0)
                {
                    x =  ((blocks[pixel]-1024) * 8) - (sizeWater*8*floor(((blocks[pixel]-1024)/sizeWater)));
                    y = floor(((blocks[pixel]-1024)/sizeWater)) * 8;
                }
                else
                {
                    y = 0;
                    x = (blocks[pixel]-1024)*8;
                }
                if(vector[k++] == 1)
                {
                    max = 0;
                    for(int i = 0; i<8;i++)
                            if(matrixHL[x+i][y] >=max)
                            {
                                max = matrixHL[x+i][y];
                            }
                    matrixHL[x][y] = ceil(max) + transformHL;
                }
                else
                {
                    min = matrixHL[x][y];
                    for(int i = 0; i<8;i++)
                            if(matrixHL[x+i][y] <=min)
                            {
                                min = matrixHL[x+i][y];
                            }
                    matrixHL[x][y] = floor(min) - transformHL;
                }
            }
            if(blocks[pixel] >= 2048 && blocks[pixel] < 3072)
            {
                if((blocks[pixel]-2048)/sizeWater != 0)
                {
                    x =  ((blocks[pixel]-2048) * 8) - (sizeWater*8*floor(((blocks[pixel]-2048)/sizeWater)));;
                    y =   floor(((blocks[pixel]-2048)/sizeWater)) * 8;
                }
                else
                {
                    y = 0;
                    x = (blocks[pixel]-2048)*8;
                }
                if(vector[k++] == 1)
                {
                    max = 0;
                    for(int i = 0; i<8;i++)
                            if(matrixHH[x+i][y] >=max)
                            {
                                max = matrixHH[x+i][y];
                            }
                    matrixHH[x][y] = ceil(max) + transformHH;
                }
                else
                {
                    min = matrixHH[x][y];
                    for(int i = 0; i<8;i++)
                            if(matrixHH[x+i][y] <=min)
                            {
                                min = matrixHH[x+i][y];
                            }
                    matrixHH[x][y] = floor(min) - transformHH;
                }
            }
    }
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_radioButton_clicked()
{   if (attack==1)
    {
        attack = 0;
        ui->radioButton->setAutoExclusive(false);
        ui->radioButton->setChecked(false);
        ui->radioButton->setAutoExclusive(true);
    }
    else
        attack=1;
}

void MainWindow::on_radioButton_2_clicked()
{
    if (attack==2)
        {
            attack = 0;
            ui->radioButton_2->setAutoExclusive(false);
            ui->radioButton_2->setChecked(false);
            ui->radioButton_2->setAutoExclusive(true);
        }
        else
            attack=2;
}

void MainWindow::on_radioButton_3_clicked()
{
    if (attack==3)
        {
            attack = 0;
            ui->radioButton_3->setAutoExclusive(false);
            ui->radioButton_3->setChecked(false);
            ui->radioButton_3->setAutoExclusive(true);
        }
        else
            attack=3;
}

void MainWindow::on_radioButton_4_clicked()
{
    if (attack==4)
        {
            attack = 0;
            ui->radioButton_4->setAutoExclusive(false);
            ui->radioButton_4->setChecked(false);
            ui->radioButton_4->setAutoExclusive(true);
        }
        else
            attack=4;
}

void MainWindow::on_radioButton_7_clicked()
{
    if (attack==7)
        {
            attack = 0;
            ui->radioButton_7->setAutoExclusive(false);
            ui->radioButton_7->setChecked(false);
            ui->radioButton_7->setAutoExclusive(true);
        }
        else
            attack=7;
}

void MainWindow::on_radioButton_8_clicked()
{
    if (attack==8)
        {
            attack = 0;
            ui->radioButton_8->setAutoExclusive(false);
            ui->radioButton_8->setChecked(false);
            ui->radioButton_8->setAutoExclusive(true);
        }
        else
            attack=8;
}

void MainWindow::on_spinBox_Median_editingFinished()
{
    if (ui->spinBox_Median->value()==1)
        MedF = 3;
    else
        MedF = 5;
}

void MainWindow::on_doubleSpinBoxjpegsalt_2_editingFinished()
{
    if (ui->doubleSpinBoxjpegsalt_2->value() + ui->doubleSpinBoxjpegpepper_2->value() > 100)
        ui->doubleSpinBoxjpegpepper_2->setValue(100-ui->doubleSpinBoxjpegsalt_2->value());
    white = ui->doubleSpinBoxjpegsalt_2->value()/100;
    black = (100 - ui->doubleSpinBoxjpegpepper_2->value())/100;
}


void MainWindow::on_doubleSpinBoxjpegpepper_2_editingFinished()
{
    if (ui->doubleSpinBoxjpegsalt_2->value() + ui->doubleSpinBoxjpegpepper_2->value() > 100)
        ui->doubleSpinBoxjpegsalt_2->setValue(100-ui->doubleSpinBoxjpegpepper_2->value());
    black = (double)(100 - ui->doubleSpinBoxjpegpepper_2->value())/100;
    white = (double)ui->doubleSpinBoxjpegsalt_2->value()/100;
}


void MainWindow::on_spinBoxRotation_2_editingFinished()
{
        angle = ui->spinBoxRotation_2->value();
}

void MainWindow::on_spinBoxCrop_2_editingFinished()
{
        Cr = (4-ui->spinBoxCrop_2->value())*8;
}

void MainWindow::on_spinBoxjpeg_2_editingFinished()
{
        Q=ui->spinBoxjpeg_2->value();
}

void MainWindow::on_doubleSpinBoxmat_2_editingFinished()
{
        mat = ui->doubleSpinBoxmat_2->value();
}

void MainWindow::on_doubleSpinBoxOtk_2_editingFinished()
{
        Otk = ui->doubleSpinBoxOtk_2->value();
}

void MainWindow::on_CB_Cuckoo_clicked()
{

}

void MainWindow::on_CB_Cuckoo_stateChanged(int arg1)
{
    if (ui->CB_Cuckoo->isChecked())
        Cuckoo=1;
    else
        Cuckoo = 0;
}

void MainWindow::on_PB_Embed_clicked()
{
    FWT(dataImage,WidthOfImage,HeightOfImage,1);

    Embeded_CVZ_With_CuckooSearch(WaterWidth,WaterHeight,WidthOfImage,HeightOfImage,MatrixLH,MatrixHL,MatrixHH,dataImage,blocks,WaterMark_Vector);

    IWT(dataImage,WidthOfImage,HeightOfImage,1);

    for (int x=0;x<HeightOfImage;x++)
        for (int y=0;y<WidthOfImage;y++)
           WaterMarked->setPixel(x,y,qRgb(dataImage[x][y],dataImage[x][y],dataImage[x][y]));

    pixmap = QPixmap::fromImage(*image);
    ui->lb_SourceImage->setMinimumHeight(128);
    ui->lb_SourceImage->setMinimumWidth(128);
    ui->lb_SourceImage->setScaledContents(true);
    ui->lb_SourceImage->setPixmap(pixmap);


    pixmap2 = QPixmap::fromImage(*WaterMark);
    ui->lb_SourceCVZ->setMinimumHeight(128);
    ui->lb_SourceCVZ->setMinimumWidth(128);
    ui->lb_SourceCVZ->setScaledContents(true);
    ui->lb_SourceCVZ->setPixmap(pixmap2);


    pixmap3 = QPixmap::fromImage(*WaterMarked);
    ui->lb_ExtractImage_2->setMinimumHeight(128);
    ui->lb_ExtractImage_2->setMinimumWidth(128);
    ui->lb_ExtractImage_2->setScaledContents(true);
    ui->lb_ExtractImage_2->setPixmap(pixmap3);
}

void MainWindow::on_PB_Extract_clicked()
{
    float **olddata = new float*[WidthOfImage];
    for(int i=0;i<WidthOfImage;i++)
        {
            olddata[i] = new float[HeightOfImage];
        }

    for (int x=0;x<HeightOfImage;x++)
        for (int y=0;y<WidthOfImage;y++)
           olddata[x][y] = dataImage[x][y];

    switch (attack)
    {
        case 1:
        {
            MedianFilter(olddata, HeightOfImage, WidthOfImage);
            break;

        }
        case 2:
        {
            SaltAndPepper(olddata, HeightOfImage, WidthOfImage);
            break;
        }
        case 3:
        {
            Rotation(olddata, HeightOfImage, WidthOfImage);
            break;
        }
        case 4:
        {
            Cropping(olddata, HeightOfImage, WidthOfImage);
            break;
        }
        case 7:
        {
            JpegComp(olddata, HeightOfImage, WidthOfImage);
            break;
        }
        case 8:
        {
            GausNoise(olddata, HeightOfImage, WidthOfImage);
            break;
        }
        default:
        {
        }
    }
    ExtractWatermark(olddata,MatrixLH,MatrixHL,MatrixHH,blocks,WaterMark_Vector,ResultVector,WaterWidth,WaterHeight);

    ui->progressBar->reset();

    for (int x=0;x<HeightOfImage;x++)
        for (int y=0;y<WidthOfImage;y++)
           WaterMarked->setPixel(x,y,qRgb(olddata[x][y],olddata[x][y],olddata[x][y]));

    pixmap3 = QPixmap::fromImage(*WaterMarked);
    ui->lb_ExtractImage_2->setMinimumHeight(128);
    ui->lb_ExtractImage_2->setMinimumWidth(128);
    ui->lb_ExtractImage_2->setScaledContents(true);
    ui->lb_ExtractImage_2->setPixmap(pixmap3);

    pixmap4 = QPixmap::fromImage(*extractCVZ);
    ui->lb_ExtractCVZ->setMinimumHeight(128);
    ui->lb_ExtractCVZ->setMinimumWidth(128);
    ui->lb_ExtractCVZ->setScaledContents(true);
    ui->lb_ExtractCVZ->setPixmap(pixmap4);


}

