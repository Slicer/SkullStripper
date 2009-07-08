
#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#include "stdlib.h"
#include "stdio.h"
#include "math.h"
#include "set"

#include "itkPoint.h"
#include "itkImage.h"
#include "itkOrientedImage.h"
#include "itkImageFileReader.h"
#include "itkImageSeriesReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h"

#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkSignedMaurerDistanceMapImageFilter.h" 
#include "itkSphereSpatialFunction.h"
#include "itkFloodFilledSpatialFunctionConditionalIterator.h"
#include "itkFloodFilledImageFunctionConditionalIterator.h"
#include "itkVotingBinaryIterativeHoleFillingImageFilter.h" 
#include "itkBinaryThresholdImageFunction.h"

#include "itkBinaryDilateImageFilter.h"
#include "itkBinaryErodeImageFilter.h"
#include "itkExtractImageFilter.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkBinaryBallStructuringElement.h"

#include "itkScalarImageToListAdaptor.h"
#include "itkListSampleToHistogramGenerator.h"
#include "itkFuzzyClassificationImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkConnectedComponentImageFilter.h"
#include "itkRelabelComponentImageFilter.h"

#include "itkSignedMaurerDistanceMapImageFilter.h"

#include "vtkSphereSource.h"
#include "vtkPolyDataMapper.h"
#include "vtkActor.h"
#include "vtkRenderWindow.h"
#include "vtkRenderer.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkProperty.h"
#include "vtkTriangleFilter.h"
#include "vtkPlatonicSolidSource.h" 
#include "vtkIdList.h"
#include "vtkPolyData.h"
#include "vtkPolyDataWriter.h"
#include "vtkXMLPolyDataWriter.h"

#include "vtkPolyDataPointSampler.h"

#include "SkullStripperCLP.h"

static const float PI = 3.1415926535897932;
#define X .525731112119133606
#define Z .850650808352039932

#ifndef M_PI
#define M_PI 3.1415926535897932
#endif
#ifndef M_PI_2
#define M_PI_2 1.5707963267948966
#endif

//Vertices, triangles, edges of a single icosahedron
static double vert[12][3] = {
  {-X, 0.0, Z}, {X, 0.0, Z}, {-X, 0.0, -Z}, {X, 0.0, -Z},
  {0.0, Z, X}, {0.0, Z, -X}, {0.0, -Z, X}, {0.0, -Z, -X},
  {Z, X, 0.0}, {-Z, X, 0.0}, {Z, -X, 0.0}, {-Z, -X, 0.0}
};
static int triang[20][3] = {
  {0,4,1}, {0,9,4}, {9,5,4}, {4,5,8}, {4,8,1},
  {8,10,1}, {8,3,10}, {5,3,8}, {5,2,3}, {2,7,3},
  {7,10,3}, {7,6,10}, {7,11,6}, {11,0,6}, {0,1,6},
  {6,1,10}, {9,0,11}, {9,11,2}, {9,2,5}, {7,2,11}
};
static int edge[30][2] = {
  {0,1}, {0,4}, {0,6}, {0,9}, {0,11}, {1,4}, {1,6}, {1,8}, {1,10}, {2,3},
  {2,5}, {2,7}, {2,9}, {2,11}, {3,5}, {3,7}, {3,8}, {3,10}, {4,5}, {4,8},
  {4,9}, {5,8}, {5,9}, {6,7}, {6,10}, {6,11}, {7,10}, {7,11}, {8,10}, {9,11}
};

itk::Point<float, 3> COG;


typedef std::vector<vtkIdType> NeighborhoodType;

const unsigned int ImageDimension = 3;
typedef signed short PixelType;
typedef itk::OrientedImage< PixelType, ImageDimension > ImageType;
typedef itk::OrientedImage<unsigned char, ImageDimension> LabelImageType;
typedef itk::OrientedImage<double, ImageDimension> FloatImageType;
typedef itk::ImageFileReader< ImageType  > ImageReaderType;

typedef itk::Statistics::ScalarImageToListAdaptor< ImageType >   AdaptorType;
typedef float        HistogramMeasurementType;
typedef itk::Statistics::ListSampleToHistogramGenerator< AdaptorType, HistogramMeasurementType > GeneratorType;
typedef GeneratorType::HistogramType  HistogramType;

LabelImageType::Pointer BinaryErodeFilter3D ( LabelImageType::Pointer & img , unsigned int ballsize )
{
  typedef itk::BinaryBallStructuringElement<unsigned char, 3> KernalType;
  typedef itk::BinaryErodeImageFilter<LabelImageType, LabelImageType, KernalType> ErodeFilterType;
  ErodeFilterType::Pointer erodeFilter = ErodeFilterType::New();
  erodeFilter->SetInput( img );

  KernalType ball;
  KernalType::SizeType ballSize;
  for (int k = 0; k < 3; k++)
    {
    ballSize[k] = ballsize;
    }
  ball.SetRadius(ballSize);
  ball.CreateStructuringElement();
  erodeFilter->SetKernel( ball );
  erodeFilter->Update();
  return erodeFilter->GetOutput();

}

LabelImageType::Pointer BinaryDilateFilter3D ( LabelImageType::Pointer & img , unsigned int ballsize )
{
  typedef itk::BinaryBallStructuringElement<unsigned char, 3> KernalType;
  typedef itk::BinaryDilateImageFilter<LabelImageType, LabelImageType, KernalType> DilateFilterType;
  DilateFilterType::Pointer dilateFilter = DilateFilterType::New();
  dilateFilter->SetInput( img );
  KernalType ball;
  KernalType::SizeType ballSize;
  for (int k = 0; k < 3; k++)
    {
    ballSize[k] = ballsize;
    }
  ball.SetRadius(ballSize);
  ball.CreateStructuringElement();
  dilateFilter->SetKernel( ball );
  dilateFilter->Update();
  return dilateFilter->GetOutput();
}

LabelImageType::Pointer BinaryOpeningFilter3D ( LabelImageType::Pointer & img , unsigned int ballsize )
{
  LabelImageType::Pointer imgErode = BinaryErodeFilter3D( img, ballsize );
  return BinaryDilateFilter3D( imgErode, ballsize );
}

LabelImageType::Pointer BinaryClosingFilter3D ( LabelImageType::Pointer & img , unsigned int ballsize )
{
  LabelImageType::Pointer imgDilate = BinaryDilateFilter3D( img, ballsize );
  return BinaryErodeFilter3D( imgDilate, ballsize );
}



void PolyDataToLabelMap( vtkPolyData* polyData, LabelImageType::Pointer label)
{
  vtkPolyDataPointSampler* sampler = vtkPolyDataPointSampler::New();
  sampler->SetInput( polyData );
  sampler->SetDistance( 0.75 );
  sampler->GenerateEdgePointsOn();
  sampler->GenerateInteriorPointsOn();
  sampler->GenerateVertexPointsOn();
  sampler->Update();

  std::cout << polyData->GetNumberOfPoints() << std::endl;
  std::cout << sampler->GetOutput()->GetNumberOfPoints() << std::endl;

  label->FillBuffer( 0 );
  for (int k = 0; k < sampler->GetOutput()->GetNumberOfPoints(); k++)
  {
    double *pt = sampler->GetOutput()->GetPoint( k );
    LabelImageType::PointType pitk;
    pitk[0] = pt[0];
    pitk[1] = pt[1];
    pitk[2] = pt[2];
    LabelImageType::IndexType idx;
    label->TransformPhysicalPointToIndex( pitk, idx );

    if ( label->GetLargestPossibleRegion().IsInside(idx) )
    {
      label->SetPixel( idx, 255 );
    }
  }

  // do morphological closing
  int ballSize = 2;
  LabelImageType::Pointer closedLabel = BinaryClosingFilter3D( label, ballSize );
  //LabelImageType::Pointer closedLabel = BinaryDilateFilter3D( label2, 2 );

  itk::ImageRegionIteratorWithIndex<LabelImageType>
    itLabel (closedLabel, closedLabel->GetLargestPossibleRegion() );

  // do flood fill using binary threshold image function
  typedef itk::BinaryThresholdImageFunction<LabelImageType> ImageFunctionType;
  ImageFunctionType::Pointer func = ImageFunctionType::New();
  func->SetInputImage( closedLabel );
  func->ThresholdBelow(1);

  FloatImageType::IndexType idx;
  label->TransformPhysicalPointToIndex( COG, idx );

  itk::FloodFilledImageFunctionConditionalIterator<LabelImageType, ImageFunctionType>
    floodFill( closedLabel, func, idx );
  
  for (floodFill.GoToBegin(); !floodFill.IsAtEnd(); ++floodFill)
  {
    LabelImageType::IndexType i = floodFill.GetIndex();
    closedLabel->SetPixel( i, 255 );
  }

  LabelImageType::Pointer finalLabel = BinaryClosingFilter3D( closedLabel, ballSize );

  for (itLabel.GoToBegin(); !itLabel.IsAtEnd(); ++itLabel)
  {
    LabelImageType::IndexType i = itLabel.GetIndex();
    label->SetPixel( i, finalLabel->GetPixel(i) );
  }

  return;
}

int FindTopSlice( ImageType::Pointer fixed )
{
  ImageType::SpacingType spacing = fixed->GetSpacing();
  ImageType::RegionType region = fixed->GetLargestPossibleRegion();
  ImageType::SizeType size = region.GetSize();
  std::vector<double> projection( size[2] );
  std::vector<double> nVoxels( size[2] );
  for (::size_t k = 0; k < size[2]; k++)
  {
    projection[k] = 0;
    nVoxels[k] = 0;
  }

  ImageType::RegionType centerRegion = fixed->GetLargestPossibleRegion();
  for (int k = 0; k < 3; k++)
  {
    centerRegion.SetIndex( k, centerRegion.GetIndex(k)+centerRegion.GetSize(k)/2-5 );
    centerRegion.SetSize( k, 11 );
  }
  itk::ImageRegionIteratorWithIndex<ImageType> itc( fixed, centerRegion );

  double aveIntensity = 0;
  for ( itc.GoToBegin(); !itc.IsAtEnd(); ++itc )
  {
    aveIntensity += static_cast<double>( itc.Get() );
  }
  aveIntensity /= (11.0*11.0*11.0);

  // find the right slice
  itk::ImageRegionIteratorWithIndex<ImageType> it( fixed, region );
  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    ImageType::IndexType idx = it.GetIndex();
    double p = static_cast<double> (it.Get());
    if( p >= aveIntensity/10 )
    {
      projection[idx[2]] += p;
      nVoxels[idx[2]] += 1;
    }
  }
  for (::size_t k = 0; k < size[2]; k++)
  {
    projection[k] /= nVoxels[k];
  }

  double noiselevel = 0;
  double noiseslice = 0;
  unsigned int topslice=size[2]-1;
  for (unsigned int k = size[2]-1; k >=0; k--)
  {
    if (projection[k] == 0)
    {
      continue;
    }
    else if(noiselevel == 0)
    {
      noiselevel = projection[k];
      noiseslice = 1;
    }
    else
    {
      if (projection[k] > noiselevel*2 || projection[k] > aveIntensity/3.0)
      {
        topslice = k;
        break;
      }
      else
      {
        noiselevel = ((noiselevel*noiseslice)+projection[k])/(noiseslice+1);
        noiseslice ++;
      }
    }
  }

  unsigned short signalThreshold = static_cast<unsigned short>(static_cast<double>(projection[topslice])/2);
  
  //for (int k = 0; k < size[2]; k++)
  //{
  //std::cout << k << " " << projection[k] << "; \n";
  //}
  //std::cout << std::endl;

  std::cout << "Signal level: " << signalThreshold << std::endl;
  std::cout << "Top of head at slice " << topslice << std::endl;

  return topslice;

}


ImageType::PixelType FindWhiteMatterPeak ( HistogramType::Pointer histogram )
{
  HistogramType::SizeType size = histogram->GetSize();
  std::cout << "Histogram size: " << size << std::endl;

  PixelType t95 = static_cast<PixelType> (histogram->Quantile(0, 0.95));
  std::cout << "t95 = " << t95 << std::endl;

  std::vector<ImageType::PixelType> intensity( size[0] );
  std::vector<float> frequency( size[0] );

  for (::size_t k = 0; k < size[0]; k++)
  {
    HistogramType::IndexType hidx;
    hidx[0] = k;
    ImageType::PixelType p = static_cast<ImageType::PixelType>( histogram->GetHistogramMinFromIndex (hidx)[0] );
    intensity[k] = p;
    frequency[k] = histogram->GetFrequency ( hidx );
  }

  // suppress the backgroud values
  frequency[0] = 0;
  frequency[1] = 0;

  // do simple five point average
  std::vector<unsigned long> smoothedfrequency( size[0] );
  for (::size_t k = 2; k < size[0]-2; k++)
  {
    double d = 0;
    for (int m = -2; m <=2; m++)
    {
      d += static_cast<double> (frequency[k+m]);
    }
    smoothedfrequency[k] = static_cast<unsigned long> ( d/5 );
    //std::cout << intensity[k] << " " << smoothedfrequency[k] << std::endl;
  }

  return t95;
}

void ComputeVertexNeighbors(vtkIdType iVertexId, vtkPolyData* pMesh, std::vector<vtkIdType>& pIdRet)
{
  std::set<vtkIdType> setNeighbors;
  vtkIdType* pIncidentCells;
  unsigned short iNumCells;
      
  pMesh->GetPointCells(iVertexId, iNumCells, pIncidentCells);
            
  int j;
  vtkIdType* pIncidentPoints;
  vtkIdType iNumPoints;
                  
  for(int i=0; i<iNumCells; ++i)
    {
    pMesh->GetCellPoints(pIncidentCells[i], iNumPoints, pIncidentPoints);
    for(j=0; j<iNumPoints; ++j)
      if(pIncidentPoints[j]!=iVertexId)
        setNeighbors.insert(pIncidentPoints[j]);
    }
  
  // make pointIds in order
  std::vector<vtkIdType> pIds;
  for (std::set<vtkIdType>::iterator m = setNeighbors.begin(); m != setNeighbors.end(); m++)
    {
    pIds.push_back( *m );
    }
  
  // find first edge
  vtkIdType Id0 = pIds[0];
  vtkIdType Id1;
  for (::size_t k = 1; k < pIds.size(); k++)
    {
    if ( !pMesh->IsEdge( Id0, pIds[k]) )
      {
      continue;
      }
    Id1 = pIds[k];
    break;
    }

  // figure out if Id0 and Id1 is in the right order;
  double pc[3];
  memcpy( pc, pMesh->GetPoint(iVertexId), 3*sizeof(double) );
  double p0[3];
  memcpy( p0, pMesh->GetPoint(Id0), 3*sizeof(double) );
  double p1[3];
  memcpy( p1, pMesh->GetPoint(Id1), 3*sizeof(double) );
  for (int m = 0; m <3; m++)
    {
    p0[m] -= pc[m];
    p1[m] -= pc[m];
    }
  double op[3];
  op[0] = p0[1]*p1[2]-p0[2]*p1[1];
  op[1] = p0[2]*p1[0]-p0[0]*p1[2];
  op[2] = p0[0]*p1[1]-p0[1]*p1[0];
  double ip = op[0]*pc[0]+op[1]*pc[1]+op[2]*pc[2];
  if (ip < 0) // swap Id0 and Id1;
    {
    vtkIdType tempId = Id0;
    Id0 = Id1;
    Id1 = tempId;
    }

  pIdRet.push_back( Id0 );
  pIdRet.push_back( Id1 );

  for (::size_t k = 2; k < pIds.size(); k++)
    {
    vtkIdType curentId = pIdRet[k-1];
    for (::size_t m = 0; m < pIds.size(); m++)
      {
      vtkIdType tempId = pIds[m];
      int InSet = 0;
      for (::size_t n = 0; n < pIdRet.size(); n++)
        {
        if (pIdRet[n] == tempId)
          {
          InSet = 1;
          break;
          }
        }

      if (InSet == 1)
        {
        continue;
        }

      if ( pMesh->IsEdge(curentId, tempId) )
        {
        pIdRet.push_back( tempId );
        break;
        }
      }
    }
}
                    
vtkPolyData* TessellateIcosahedron(int level)
{

  //Calculate n_vertex, n_triag
  int n=0;
  if(level > 2) 
    {
    for(int i=1; i<(level-1); i++) 
      n += i;
    }
  int n_vert = 12 + (level - 1)*30 + n*20;
  int numtriags = 0;
  if(level == 1)
    {
    numtriags = 20;
    }
  else 
    {
    n = 1;
    do
      {
      for(int m=1; m<=n; m++)
        {
        numtriags = numtriags + 3;
        if(m != n)
          numtriags = numtriags + 3;
      
        }
      n++;
      }while(n<=level);
    numtriags = numtriags * 20;
    numtriags = numtriags / 3;
    }

  typedef double Point3[3];
  //Allocate datas
  Point3* all_vert = new Point3[n_vert];
  Point3* all_triangs = new Point3[numtriags*3];//all possible vertices in triangs
  int * triangs = new int[3*numtriags];

  int i, m, k;
  double x1, x2, y1, y2, z1, z2, x3, y3, z3; 
  double dx12, dy12, dz12, dx23, dy23, dz23;
  double length;   
   
  double epsilon = 0.00001;//machine epsilon??
   
  memcpy(all_vert, vert, 12*sizeof(Point3));
   
  //std::cout<<"after memcpy"<<std::endl;
   
  k=12;
  for(i=0; i<30; i++) 
    {
    x1 = vert[edge[i][0] ][0];
    y1 = vert[edge[i][0] ][1];
    z1 = vert[edge[i][0] ][2];
    x2 = vert[edge[i][1] ][0];
    y2 = vert[edge[i][1] ][1];
    z2 = vert[edge[i][1] ][2];
    dx12 = (x2 - x1)/level;
    dy12 = (y2 - y1)/level;
    dz12 = (z2 - z1)/level;
    for(n=1; n<level; n++) 
      {
      all_vert[k][0] = x1 + n*dx12;
      all_vert[k][1] = y1 + n*dy12;
      all_vert[k][2] = z1 + n*dz12;
      length = sqrt(static_cast<double> (all_vert[k][0]*all_vert[k][0]+
                                         all_vert[k][1]*all_vert[k][1]+ 
                                         all_vert[k][2]*all_vert[k][2]));
      all_vert[k][0] /= length;
      all_vert[k][1] /= length;
      all_vert[k][2] /= length;
      k++;
      }
    }

  if(level > 2) 
    {
    for(i=0; i<20; i++) 
      {
      x1 = vert[triang[i][0] ][0];
      y1 = vert[triang[i][0] ][1];
      z1 = vert[triang[i][0] ][2];
      x2 = vert[triang[i][1] ][0];
      y2 = vert[triang[i][1] ][1];
      z2 = vert[triang[i][1] ][2];
      x3 = vert[triang[i][2] ][0];
      y3 = vert[triang[i][2] ][1];
      z3 = vert[triang[i][2] ][2];
      dx12 = (x2 - x1)/level;
      dy12 = (y2 - y1)/level;
      dz12 = (z2 - z1)/level;
      dx23 = (x3 - x2)/level;
      dy23 = (y3 - y2)/level;
      dz23 = (z3 - z2)/level;

      n = 1;
      do 
        {
        for(m=1; m<=n; m++) 
          {
          all_vert[k][0] = x1 + (n+1)*dx12 + m*dx23;
          all_vert[k][1] = y1 + (n+1)*dy12 + m*dy23;
          all_vert[k][2] = z1 + (n+1)*dz12 + m*dz23;
          length = sqrt((double) all_vert[k][0]*all_vert[k][0]+
                        all_vert[k][1]*all_vert[k][1]+
                        all_vert[k][2]*all_vert[k][2]);
          all_vert[k][0] /= length;
          all_vert[k][1] /= length;
          all_vert[k][2] /= length;
          k++;
          }
        n++;
        }while( n<=(level-2) );
      }
    }
  numtriags=0;
   
  //std::cout<<"before get triangulation"<<std::endl;   
  //std::cout<<n_triangs<<std::endl;
   
  // get triangulation
  if (level > 1) 
    {
    for(i=0; i<20; i++) 
      {
      x1 = vert[triang[i][0] ][0];
      y1 = vert[triang[i][0] ][1];
      z1 = vert[triang[i][0] ][2];
      x2 = vert[triang[i][1] ][0];
      y2 = vert[triang[i][1] ][1];
      z2 = vert[triang[i][1] ][2];
      x3 = vert[triang[i][2] ][0];
      y3 = vert[triang[i][2] ][1];
      z3 = vert[triang[i][2] ][2];
      dx12 = (x2 - x1)/level;
      dy12 = (y2 - y1)/level;
      dz12 = (z2 - z1)/level;
      dx23 = (x3 - x2)/level;
      dy23 = (y3 - y2)/level;
      dz23 = (z3 - z2)/level;

      n = 1;
      do 
        {
        for(m=1; m<=n; m++) 
          {
          // Draw lower triangle
          all_triangs[numtriags][0] = x1 + n*dx12 + m*dx23;
          all_triangs[numtriags][1] = y1 + n*dy12 + m*dy23;
          all_triangs[numtriags][2] = z1 + n*dz12 + m*dz23;
          length = sqrt((double) all_triangs[numtriags][0]*all_triangs[numtriags][0]+
                        all_triangs[numtriags][1]*all_triangs[numtriags][1]+
                        all_triangs[numtriags][2]*all_triangs[numtriags][2]);
          all_triangs[numtriags][0] /= length;
          all_triangs[numtriags][1] /= length;
          all_triangs[numtriags][2] /= length;
          numtriags++;
          all_triangs[numtriags][0] = x1 + (n-1)*dx12 + (m-1)*dx23;
          all_triangs[numtriags][1] = y1 + (n-1)*dy12 + (m-1)*dy23;
          all_triangs[numtriags][2] = z1 + (n-1)*dz12 + (m-1)*dz23;
          length = sqrt((double) all_triangs[numtriags][0]*all_triangs[numtriags][0]+
                        all_triangs[numtriags][1]*all_triangs[numtriags][1]+
                        all_triangs[numtriags][2]*all_triangs[numtriags][2]);
          all_triangs[numtriags][0] /= length;
          all_triangs[numtriags][1] /= length;
          all_triangs[numtriags][2] /= length;
          numtriags++;
          all_triangs[numtriags][0] = x1 + n*dx12 + (m-1)*dx23;
          all_triangs[numtriags][1] = y1 + n*dy12 + (m-1)*dy23;
          all_triangs[numtriags][2] = z1 + n*dz12 + (m-1)*dz23;
          length = sqrt((double) all_triangs[numtriags][0]*all_triangs[numtriags][0]+
                        all_triangs[numtriags][1]*all_triangs[numtriags][1]+
                        all_triangs[numtriags][2]*all_triangs[numtriags][2]);
          all_triangs[numtriags][0] /= length;
          all_triangs[numtriags][1] /= length;
          all_triangs[numtriags][2] /= length;
          numtriags++;
          if ( m != n ) 
            {
            // Draw lower left triangle
            all_triangs[numtriags][0] = x1 + n*dx12 + m*dx23;
            all_triangs[numtriags][1] = y1 + n*dy12 + m*dy23;
            all_triangs[numtriags][2] = z1 + n*dz12 + m*dz23;
            length = sqrt((double) all_triangs[numtriags][0]*all_triangs[numtriags][0]+
                          all_triangs[numtriags][1]*all_triangs[numtriags][1]+
                          all_triangs[numtriags][2]*all_triangs[numtriags][2]);
            all_triangs[numtriags][0] /= length;
            all_triangs[numtriags][1] /= length;
            all_triangs[numtriags][2] /= length;
            numtriags++;
            all_triangs[numtriags][0] = x1 + (n-1)*dx12 + m*dx23;
            all_triangs[numtriags][1] = y1 + (n-1)*dy12 + m*dy23;
            all_triangs[numtriags][2] = z1 + (n-1)*dz12 + m*dz23;
            length = sqrt((double) all_triangs[numtriags][0]*all_triangs[numtriags][0]+
                          all_triangs[numtriags][1]*all_triangs[numtriags][1]+
                          all_triangs[numtriags][2]*all_triangs[numtriags][2]);
            all_triangs[numtriags][0] /= length;
            all_triangs[numtriags][1] /= length;
            all_triangs[numtriags][2] /= length;
            numtriags++;
            all_triangs[numtriags][0] = x1 + (n-1)*dx12 + (m-1)*dx23;
            all_triangs[numtriags][1] = y1 + (n-1)*dy12 + (m-1)*dy23;
            all_triangs[numtriags][2] = z1 + (n-1)*dz12 + (m-1)*dz23;
            length = sqrt((double) all_triangs[numtriags][0]*all_triangs[numtriags][0]+
                          all_triangs[numtriags][1]*all_triangs[numtriags][1]+
                          all_triangs[numtriags][2]*all_triangs[numtriags][2]);
            all_triangs[numtriags][0] /= length;
            all_triangs[numtriags][1] /= length;
            all_triangs[numtriags][2] /= length;
            numtriags++;
            }
          }
        n++;
        } while( n<=level );
      }
    }
   
  //std::cout<<"before indexing of triangs"<<std::endl;
   
  // indexing of triangs
  if (level == 1) 
    {
    memcpy(triangs, triang, 20*3*sizeof(int));
    numtriags = 20;
    } 
  else 
    {
    //find for every point in triangle list the corresponding index in all_vert
     
    // initialize
    for (i=0; i < numtriags; i ++) {
    triangs[i] = -1;
    }

    // find indexes
    for(i=0; i<n_vert; i++) 
      {
      for (int j = 0; j < numtriags; j++) 
        {
        if (triangs[j] < 0) 
          {
          if ( (fabs(all_vert[i][0] - all_triangs[j][0]) < epsilon) && 
               (fabs(all_vert[i][1] - all_triangs[j][1]) < epsilon) && 
               (fabs(all_vert[i][2] - all_triangs[j][2]) < epsilon ) ) 
            {
            triangs[j] = i;
            }
          }
        }
      }
     
    //for(i=0; i<n_vert; i++) 
    //  std::cout<<triangs[3*i]<<","<<triangs[3*i+1]<<","<<triangs[3*i+2]<<std::endl;

    for (i=0; i < numtriags; i ++) 
      {
      if (triangs[i] == -1)
        std::cerr << " - " << i << " :" << all_triangs[i][0] 
                  << "," << all_triangs[i][1] << "," << all_triangs[i][2] << std::endl;
      }
     
    // numtriags is the number of vertices in triangles -> divide it by 3 
    numtriags = numtriags / 3;
    }

  vtkIdList *ids = vtkIdList::New();
  vtkPoints *pts = vtkPoints::New();
  vtkPolyData* polyData = vtkPolyData::New();

  polyData ->SetPoints (pts);

  ids -> SetNumberOfIds(0);
  pts -> SetNumberOfPoints(0);
  polyData->Allocate();
  for (int k = 0; k < n_vert; k++)
    {
    vtkIdType id;
    id = pts->InsertNextPoint(all_vert[k][0],
                              all_vert[k][1], all_vert[k][2]);
    ids->InsertNextId(id);
    }
  for (int k = 0; k < numtriags; k++)
    {
    vtkIdList *tids = vtkIdList::New();
    tids->SetNumberOfIds(0);
    tids->InsertNextId(triangs[3*k]);
    tids->InsertNextId(triangs[3*k+1]);
    tids->InsertNextId(triangs[3*k+2]);
    polyData->InsertNextCell(VTK_TRIANGLE, tids);

    }

  return polyData;

  delete [] all_vert;
  delete [] all_triangs;
  delete [] triangs;
}

int main( int argc, char *argv[] )
{

  PARSE_ARGS;

  ImageReaderType::Pointer imageReader  = ImageReaderType::New();
  ImageType::Pointer image0;

  typedef itk::GDCMImageIO                                  ImageIOType;
  typedef itk::GDCMSeriesFileNames                          SeriesFileNames;
  typedef itk::ImageSeriesReader< ImageType >               SeriesReaderType;
  std::string subjectname;

  // Read image
  if (itksys::SystemTools::FileIsDirectory(inputVolume.c_str() ))     // dicom series input
    {
    std::vector<std::string> lines;
    itksys::SystemTools::Split(inputVolume.c_str(), lines, '/');
    subjectname = lines[lines.size()-1];
    
    ImageIOType::Pointer gdcmIO = ImageIOType::New();
    SeriesFileNames::Pointer it = SeriesFileNames::New();

    it->SetInputDirectory( inputVolume.c_str() );

    SeriesReaderType::Pointer sReader = SeriesReaderType::New();

    const SeriesReaderType::FileNamesContainer & filenames =
      it->GetInputFileNames();

    sReader->SetFileNames( filenames );
    sReader->SetImageIO( gdcmIO );

    try
      {
      sReader->Update();
      image0 = sReader->GetOutput();
      }
    catch (itk::ExceptionObject &excp)
      {
      std::cerr << "Exception thrown while writing the image" <<
        std::endl;
      std::cerr << excp << std::endl;
      return EXIT_FAILURE;
      }
    }
  else         // input is a file
    {

    itk::GDCMImageIO::Pointer dicomio = itk::GDCMImageIO::New();
    if ( dicomio->CanReadFile(inputVolume.c_str()) )                  // input is a single dicom file
      {

      std::string dicompathname = itksys::SystemTools::GetFilenamePath(inputVolume);
      std::vector<std::string> lines;
      itksys::SystemTools::Split(dicompathname.c_str(), lines, '/');
      subjectname = lines[lines.size()-1];

      ImageIOType::Pointer gdcmIO = ImageIOType::New();
      SeriesFileNames::Pointer it = SeriesFileNames::New();
      
      it->SetInputDirectory( dicompathname.c_str() );
      
      SeriesReaderType::Pointer sReader = SeriesReaderType::New();
      
      const SeriesReaderType::FileNamesContainer & filenames =
        it->GetInputFileNames();
      
      sReader->SetFileNames( filenames );
      sReader->SetImageIO( gdcmIO );
      
      try
        {
        sReader->Update();
        image0 = sReader->GetOutput();
        }
      catch (itk::ExceptionObject &excp)
        {
        std::cerr << "Exception thrown while writing the image" <<
          std::endl;
        std::cerr << excp << std::endl;
        return EXIT_FAILURE;
        }
      
      }
    else
      {
      subjectname = itksys::SystemTools::GetFilenameWithoutExtension(inputVolume);

      ImageReaderType::Pointer reader2 = ImageReaderType::New();
      reader2->SetFileName( inputVolume.c_str() );
      try
        {
        reader2->Update();
        image0 = reader2->GetOutput();
        }
      catch (itk::ExceptionObject &ex)
        {
        std::cout << ex << std::endl;
        return EXIT_FAILURE;
        }
      }
    }
  
  std::cout << "Input image orientation: " << image0->GetDirection() << std::endl;

  // convert image into RAS orientation
  ImageType::SpacingType spacing = image0->GetSpacing();
  float minSpace = spacing[0];
  minSpace = minSpace > spacing[1] ? spacing[0] : minSpace;
  minSpace = minSpace > spacing[2] ? spacing[0] : minSpace;
  spacing.Fill( minSpace );
  spacing[2] *= 1.5;

  // figure out extent of the volume in the physical space
  float xmax = itk::NumericTraits<float>::min();
  float ymax = itk::NumericTraits<float>::min();
  float zmax = itk::NumericTraits<float>::min();
  float xmin = itk::NumericTraits<float>::max();
  float ymin = itk::NumericTraits<float>::max();
  float zmin = itk::NumericTraits<float>::max();

  ImageType::IndexType startIndex = image0->GetLargestPossibleRegion().GetIndex();
  ImageType::SizeType size = image0->GetLargestPossibleRegion().GetSize();
  ImageType::RegionType region;

  ImageType::Pointer dummyImage = ImageType::New();
  region.SetIndex(0, 0);
  region.SetIndex(1, 0);
  region.SetIndex(2, 0);
  region.SetSize(0, 2);
  region.SetSize(1, 2);
  region.SetSize(2, 2);
  dummyImage->SetRegions( region );
  dummyImage->Allocate();
  
  itk::ImageRegionIteratorWithIndex<ImageType> itd( dummyImage, region );
  for (itd.GoToBegin(); !itd.IsAtEnd(); ++itd )
    {
    ImageType::IndexType idx = itd.GetIndex();
    std::cout << idx << "  ";
    ImageType::PointType pt;
    
    for (int k = 0; k < 3; k++)
      {
      idx[k] = idx[k]*size[k]+startIndex[k];
      }
    std::cout << idx << "  ";
    image0->TransformIndexToPhysicalPoint( idx, pt );
    std::cout << pt << std::endl;

    if (pt[0] < xmin)
      {
      xmin = pt[0];
      }
    if (pt[0] > xmax)
      {
      xmax = pt[0];
      }

    if (pt[1] < ymin)
      {
      ymin = pt[1];
      }
    if (pt[1] > ymax)
      {
      ymax = pt[1];
      }

    if (pt[2] < zmin)
      {
      zmin = pt[2];
      }
    if (pt[2] > zmax)
      {
      zmax = pt[2];
      }
    
    }

  std::cout << "Physical extent:\n";
  std::cout << "[ " 
    << xmin << ", " << ymin << ", " << zmin 
    << "] -- [ " 
    << xmax << ", " << ymax << ", " << zmax 
    << "]\n";

  ImageType::PointType origin;
  origin[0] = xmin; origin[1] = ymin; origin[2] = zmin; 
  
  region.SetIndex(0, 0);
  region.SetIndex(1, 0);
  region.SetIndex(2, 0);

  region.SetSize(0, static_cast<int>((xmax-xmin)/spacing[0]));
  region.SetSize(1, static_cast<int>((ymax-ymin)/spacing[1]));
  region.SetSize(2, static_cast<int>((zmax-zmin)/spacing[2]));
  
  ImageType::Pointer outImage = ImageType::New();
  outImage->SetRegions( region );
  outImage->SetSpacing( spacing );
  outImage->SetOrigin( origin );

  outImage->Allocate();
  outImage->FillBuffer( 0 );

  itk::LinearInterpolateImageFunction< ImageType, double >::Pointer interpolator = 
    itk::LinearInterpolateImageFunction< ImageType, double >::New();

  interpolator->SetInputImage( image0 );
  
  itk::ImageRegionIteratorWithIndex<ImageType> itOut( outImage, region );
  for (itOut.GoToBegin(); !itOut.IsAtEnd(); ++itOut)
  {
    ImageType::IndexType idx = itOut.GetIndex();
    ImageType::PointType pt;
    
    outImage->TransformIndexToPhysicalPoint( idx, pt );
    itk::ContinuousIndex<double, 3> cIdx;
    image0->TransformPhysicalPointToContinuousIndex( pt, cIdx );

    if( image0->GetLargestPossibleRegion().IsInside(cIdx) )
      {
      ImageType::PixelType p = static_cast<ImageType::PixelType>( interpolator->Evaluate(pt) );
      itOut.Set( p );
    }
  }

  //itk::ImageFileWriter<ImageType>::Pointer rasImage = itk::ImageFileWriter<ImageType>::New();
  //rasImage->SetFileName( "oimage.mhd" );
  //rasImage->SetInput( outImage );
  //rasImage->Update( );
  

  // figure out the top of the head
  int nTopSlice = FindTopSlice( outImage );
  int nStartSlice = nTopSlice - static_cast<int>(150.0/spacing[2]);
  int numberOfSlices = nTopSlice-nStartSlice +1;

  if (nStartSlice < region.GetIndex(2))
  {
    nStartSlice = region.GetIndex(2);
    numberOfSlices = nTopSlice-nStartSlice+1;
  }

  region.SetIndex(2, nStartSlice);
  region.SetSize(2, numberOfSlices);
   
  std::cout << "Extract region: " << region << std::endl;

  itk::ExtractImageFilter<ImageType, ImageType>::Pointer extract = itk::ExtractImageFilter<ImageType, ImageType>::New();
  extract->SetInput( outImage );
  extract->SetExtractionRegion( region );
  extract->Update();

  //itk::ImageFileWriter<ImageType>::Pointer extImage = itk::ImageFileWriter<ImageType>::New();
  //extImage->SetFileName( "eimage.mhd" );
  //extImage->SetInput( extract->GetOutput() );
  //extImage->Update( );

  ImageType::Pointer image = outImage;
  spacing = image->GetSpacing();

  // initialize label image
  LabelImageType::Pointer label = LabelImageType::New();
  label->CopyInformation( image );
  label->SetRegions( label->GetLargestPossibleRegion() );
  label->Allocate();

  LabelImageType::Pointer flabel = LabelImageType::New();
  flabel->CopyInformation( image );
  flabel->SetRegions( flabel->GetLargestPossibleRegion() );
  flabel->Allocate();
  flabel->FillBuffer( 0 );

  itk::ImageFileWriter<LabelImageType>::Pointer wlabel = itk::ImageFileWriter<LabelImageType>::New();

  // compute histogram
  AdaptorType::Pointer adaptor = AdaptorType::New();
  adaptor->SetImage( image );

  GeneratorType::Pointer generator = GeneratorType::New();

  HistogramType::SizeType bSize;
  bSize.Fill( 256 );

  generator->SetListSample( adaptor );
  generator->SetNumberOfBins( bSize );
  generator->SetMarginalScale( 10.0 );
  generator->Update();

  HistogramType::Pointer histogram = const_cast<HistogramType*>( generator->GetOutput() );
  PixelType t2 = static_cast<PixelType>(histogram->Quantile(0, 0.02));
  PixelType t98 = static_cast<PixelType> (histogram->Quantile(0, 0.98));
  PixelType tinit = static_cast<PixelType>(t2+0.1*static_cast<float>(t98-t2));  

  FindWhiteMatterPeak ( histogram );

  // compute brain size and center of gravity
  COG.Fill( 0.0 );
  unsigned long HeadVoxelCount = 0;
  itk::ImageRegionIteratorWithIndex<ImageType> itImg( image, image->GetLargestPossibleRegion() );
  itk::ImageRegionIteratorWithIndex<ImageType> itCrop( image, region );

  for ( itCrop.GoToBegin(); !itCrop.IsAtEnd(); ++itCrop )
    {
    PixelType a = itCrop.Get();
    if (a < tinit || a > t98)
      {
      continue;
      }
    HeadVoxelCount ++;
    ImageType::IndexType idx = itCrop.GetIndex();
    ImageType::PointType point;
    image->TransformIndexToPhysicalPoint( idx, point );
    for (::size_t k = 0; k < ImageDimension; k++)
      {
      COG[k] += point[k];
      }
    }
  
  float HeadVolume = static_cast<float>( HeadVoxelCount );
  for (::size_t k = 0; k < ImageDimension; k++)
    {
    COG[k] /= static_cast<float>( HeadVoxelCount );
    HeadVolume *= spacing[k];
    }

  // geometry you learn from middle school
  float radius = pow(HeadVolume*3.0/4.0/PI, 1.0/3.0);
  
  std::cout << "Threshold: \n";
  std::cout << "t2 = " << t2 << std::endl;
  std::cout << "tinit = " << tinit << std::endl;
  std::cout << "t98 = " << t98 << std::endl;
  std::cout << "number of head voxel = " << HeadVoxelCount << std::endl;
  std::cout << "volume of head = " << HeadVolume << std::endl;
  std::cout << "radius of head = " << radius << std::endl;
  std::cout << "COG = " << COG << std::endl;

  ImageType::IndexType COGIdx;
  image->TransformPhysicalPointToIndex( COG, COGIdx ); 
  std::cout << COGIdx << ": " << image->GetPixel( COGIdx ) << std::endl;

  // figure out aspects of the initial elipsoid
  ImageType::SizeType imageSize = image->GetLargestPossibleRegion().GetSize();
  ImageType::IndexType imageStart = image->GetLargestPossibleRegion().GetIndex();

  ImageType::IndexType Idx = COGIdx;
  PixelType tNonBackground = static_cast<PixelType>(0.25*static_cast<float>(tinit));
  int xStart = 0;
  int xEnd = 0;
  for (::size_t k = imageStart[0]; k < imageStart[0]+imageSize[0]; k++)
  {
    Idx[0] = k;
    if ( image->GetPixel(Idx) > tNonBackground )
    {
      xStart = k;
      break;
    }
  }
  for (int k = imageStart[0]+imageSize[0]-1; k >= imageStart[0]; k--)
  {
    Idx[0] = k;
    if ( image->GetPixel(Idx) > tNonBackground )
    {
      xEnd = k;
      break;
    }
  }
  float headWidth = static_cast<float>(xEnd-xStart)*spacing[0];

  Idx = COGIdx;
  int yStart = 0;
  int yEnd = 0;
  for (::size_t k = imageStart[1]; k < imageStart[1]+imageSize[1]; k++)
  {
    Idx[1] = k;
    if ( image->GetPixel(Idx) > tNonBackground )
    {
      yStart = k;
      break;
    }
  }
  for (int k = imageStart[1]+imageSize[1]-1; k >= imageStart[1]; k--)
  {
    Idx[1] = k;
    if ( image->GetPixel(Idx) > tNonBackground )
    {
      yEnd = k;
      break;
    }
  }
  float headLength = static_cast<float>(yEnd-yStart)*spacing[1];

  Idx = COGIdx;
  int zEnd=0;
  for (int k = imageStart[2]+imageSize[2]-1; k >= imageStart[2]; k--)
  {
    Idx[2] = k;
    if ( image->GetPixel(Idx) > tNonBackground )
    {
      zEnd = k;
      break;
    }
  }
  float headHeight = static_cast<float>(zEnd-COGIdx[2])*spacing[2]*2;

  std::cout << COG << " " << COGIdx << std::endl;
  std::cout << "spacing: " << spacing << std::endl;
  std::cout << xStart << " " << xEnd << "==" << headWidth << std::endl;
  std::cout << yStart << " " << yEnd << "==" << headLength << std::endl;
  std::cout << COGIdx[2] << " " << zEnd << "==" << headHeight << std::endl;

  // determain ellipsoid dimensions
  headLength *= (0.5/headWidth);
  headHeight *= (0.4/headWidth);
  headHeight = (headHeight > 0.5 ? headHeight : 0.5);
  headWidth = 0.5;

  std::cout << headWidth << " " << headLength << " " << headHeight << std::endl;

  vtkPolyData* polyData = TessellateIcosahedron( sphericalResolution );
  int nPoints = polyData->GetNumberOfPoints();
 
  // Build neighborhood structure
  std::vector<NeighborhoodType> NeighborhoodStructure;
  polyData->BuildLinks();
  for (int k = 0; k < nPoints; k++)
    {
    NeighborhoodType setIds;
    ComputeVertexNeighbors(k, polyData , setIds);
    NeighborhoodStructure.push_back( setIds );
    }

  // put mesh in the right position with right radius
  vtkPoints * allPoints = polyData->GetPoints();
  for (int k = 0; k < nPoints; k++)
    {
    double* point = polyData->GetPoint( k );
    point[0] = point[0]*radius*headWidth+COG[0];
    point[1] = point[1]*radius*headLength+COG[1];
    point[2] = point[2]*radius*headHeight+COG[2];
    allPoints->SetPoint( k, point[0], point[1], point[2] );
    }  

  std::cout << "Initial mash generated\n";
  PolyDataToLabelMap( polyData, label );
  label = BinaryDilateFilter3D( label, 5 );

  // tissue classification
  itk::CastImageFilter<ImageType, FloatImageType>::Pointer imageCaster = itk::CastImageFilter<ImageType, FloatImageType>::New();
  itk::CastImageFilter<LabelImageType, FloatImageType>::Pointer labelCaster = itk::CastImageFilter<LabelImageType, FloatImageType>::New();

  imageCaster->SetInput( image );
  imageCaster->Update();

  labelCaster->SetInput( label );
  labelCaster->Update();

  typedef itk::FuzzyClassificationImageFilter<FloatImageType, FloatImageType> ClassifierType;
  ClassifierType::Pointer classifier = ClassifierType::New();
  classifier->SetInput( imageCaster->GetOutput() );

  classifier->SetNumberOfClasses( 3 );
  classifier->SetBiasCorrectionOption( 1 );
  classifier->SetImageMask( labelCaster->GetOutput() );

  try
    {
    classifier->Update();
    }
  catch (itk::ExceptionObject &ex)
    {
    std::cout << ex << std::endl;
    return EXIT_FAILURE;
    }

  // extrapolate
  FloatImageType::Pointer gain = classifier->GetBiasField();
  FloatImageType::Pointer csf = classifier->GetOutput(0);
  FloatImageType::Pointer gm = classifier->GetOutput(1);
  FloatImageType::Pointer wm = classifier->GetOutput(2);

  FloatImageType::Pointer feature = FloatImageType::New();
  feature->CopyInformation( csf );
  feature->SetRegions( feature->GetLargestPossibleRegion() );
  feature->Allocate();
  feature->FillBuffer( 0 );
  
  FloatImageType::Pointer featureWM = FloatImageType::New();
  featureWM->CopyInformation( wm );
  featureWM->SetRegions( featureWM->GetLargestPossibleRegion() );
  featureWM->Allocate();
  featureWM->FillBuffer( 0 );

  const std::vector<float>& classcenter = classifier->GetClassCentroid();
  const std::vector<float>& classstd = classifier->GetClassStandardDeviation();

  for (itImg.GoToBegin(); !itImg.IsAtEnd(); ++itImg)
  {
    ImageType::IndexType idx = itImg.GetIndex();
    float p = static_cast<float>( itImg.Get() );
    if (label->GetPixel(idx) != 0)
    {
      continue;
    }
    float g = gain->GetPixel( idx );
    float csf0 = p - classcenter[0] * g;
    if (csf0 != 0)
    {
      csf0 = 1/(csf0*csf0);
    }
    float gm0 = p - classcenter[1] * g;
    if (gm0 != 0)
    {
      gm0 = 1/(gm0*gm0);
    }
    float wm0 = p - classcenter[2] * g;
    if (wm0 != 0)
    {
      wm0 = 1/(wm0*wm0);
    }
    g = csf0+wm0+gm0;
    csf->SetPixel( idx, csf0/g ); 
    gm->SetPixel( idx, gm0/g ); 
    wm->SetPixel( idx, wm0/g ); 
  }

  for (itImg.GoToBegin(); !itImg.IsAtEnd(); ++itImg)
  {
    ImageType::IndexType idx = itImg.GetIndex();
    float p = csf->GetPixel(idx);
    p += p; p -= wm->GetPixel(idx); p -= gm->GetPixel(idx);

    feature->SetPixel(idx, p);

    p = static_cast<float>( itImg.Get() );
    float g = gain->GetPixel( idx );
    float wm0 = (p - classcenter[2] * g)/classstd[2];
    
    featureWM->SetPixel( idx, wm0 );
    p = static_cast<float>( itImg.Get() )/g;
    itImg.Set( static_cast<ImageType::PixelType>(p) );
  }

  // do iteration

  double radiusMin = 3.33;
  double radiusMax = 10;

  double E = (radiusMin + radiusMax)/(2*radiusMin*radiusMax);
  double F = 6*radiusMin*radiusMax/(radiusMax-radiusMin);

  int iter = 0;
  int nSearchPoints = 40;
  double stepSize = 0.5;
  double relaxFactor = 0.75;

  std::vector<ImageType::PixelType> IntensityOnLine(nSearchPoints);


  double change = 0;
  double change1 = 0;
  double change2 = 0;
  double change3 = 0;

  float lThreshold = (classcenter[0]+classcenter[1])/2;
  float uThreshold = (classcenter[2]+2*classstd[2]);

  std::cout << "threshold: " << lThreshold << ", " << uThreshold << std::endl;

  itk::LinearInterpolateImageFunction< ImageType, double >::Pointer imgInterpolator = 
    itk::LinearInterpolateImageFunction< ImageType, double >::New();
  imgInterpolator->SetInputImage( image );

  itk::LinearInterpolateImageFunction< FloatImageType, double >::Pointer fInterpolator = 
    itk::LinearInterpolateImageFunction< FloatImageType, double >::New();
  itk::LinearInterpolateImageFunction< FloatImageType, double >::Pointer wInterpolator = 
    itk::LinearInterpolateImageFunction< FloatImageType, double >::New();

  itk::LinearInterpolateImageFunction< FloatImageType, double >::Pointer csfInterpolator = 
    itk::LinearInterpolateImageFunction< FloatImageType, double >::New();
  itk::LinearInterpolateImageFunction< FloatImageType, double >::Pointer wmInterpolator = 
    itk::LinearInterpolateImageFunction< FloatImageType, double >::New();
  itk::LinearInterpolateImageFunction< FloatImageType, double >::Pointer gmInterpolator = 
    itk::LinearInterpolateImageFunction< FloatImageType, double >::New();

  fInterpolator->SetInputImage( feature );
  wInterpolator->SetInputImage( featureWM );
  csfInterpolator->SetInputImage( csf );
  gmInterpolator->SetInputImage( gm );
  wmInterpolator->SetInputImage( wm );

  int LeftBound = 5;
  int RightBound = 30;

  std::vector<float> imgProfile(LeftBound+RightBound+1);
  std::vector<float> fProfile(LeftBound+RightBound+1);
  std::vector<float> wProfile(LeftBound+RightBound+1);
  std::vector<float> csfProfile(LeftBound+RightBound+1);
  std::vector<float> gmProfile(LeftBound+RightBound+1);
  std::vector<float> wmProfile(LeftBound+RightBound+1);

  while (1)
    {
    if (iter == nIterations)
    {
      break;
    }
    if (change > 0 && change1 > 0 && change2 > 0 && change3 > 0 && (2*change/(change1+change3)) < 0.05)
    {
      break;
    }

    change = 0;
    change1 = 0;
    change2 = 0;
    change3 = 0;
    for (int k = 0; k < nPoints; k++)  // for each point
      {
      double update[3];
      double update1[3];
      double update2[3];
      double update3[3];      

      // 1. compute normal at the point and average position of its neighbors
      double pc[3];
      double * p = polyData->GetPoint( k );
      pc[0] = p[0]; pc[1] = p[1]; pc[2] = p[2]; 

      NeighborhoodType nbr = NeighborhoodStructure[k];
      int nNeighbors = nbr.size();
      double normal[3];
      double average[3];
      double averageEdgeLength = 0;
      for (int m = 0; m < 3; m++)
        {
        normal[m] = 0;
        average[m] = 0;
        update[m] = 0;
        update1[m] = 0;
        update2[m] = 0;
        update3[m] = 0;
        }
      
      for (int m = 0; m < nNeighbors; m ++)
        {
        vtkIdType id = nbr[m];
        double p0[3];
        double p1[3];
        p = polyData->GetPoint( id );
        p0[0] = p[0]; p0[1] = p[1]; p0[2] = p[2]; 
        id = nbr[(m+1)%nNeighbors];
        p = polyData->GetPoint( id );
        p1[0] = p[0]; p1[1] = p[1]; p1[2] = p[2]; 
        
        for (int n = 0; n < 3; n++)
          {
          average[n] += p0[n];
          p0[n] -= pc[n];
          p1[n] -= pc[n];
          }
        
        averageEdgeLength += sqrt(p0[0]*p0[0]+p0[1]*p0[1]+p0[2]*p0[2]);

        double op[3];
        op[0] = p0[1]*p1[2]-p0[2]*p1[1];
        op[1] = p0[2]*p1[0]-p0[0]*p1[2];
        op[2] = p0[0]*p1[1]-p0[1]*p1[0];
        
        for (int n = 0; n < 3; n++)
          {
          normal[n] += op[n];
          }
        }
      
      average[0] /= static_cast<float>( nNeighbors );
      average[1] /= static_cast<float>( nNeighbors );
      average[2] /= static_cast<float>( nNeighbors );

      averageEdgeLength /= static_cast<float>( nNeighbors );

      float mag = sqrt(normal[0]*normal[0]+normal[1]*normal[1]+normal[2]*normal[2]);
      normal[0] /= mag;
      normal[1] /= mag;
      normal[2] /= mag;
      
      double diffvector[3];
      diffvector[0] = average[0]-pc[0];
      diffvector[1] = average[1]-pc[1];
      diffvector[2] = average[2]-pc[2];
      
      double normalcomponent = fabs(diffvector[0]*normal[0]+diffvector[1]*normal[1]+diffvector[2]*normal[2]);
      double diffnormal[3];
      diffnormal[0] = normalcomponent*normal[0];
      diffnormal[1] = normalcomponent*normal[1];
      diffnormal[2] = normalcomponent*normal[2];

      double difftangent[3];
      difftangent[0] = diffvector[0]-diffnormal[0];
      difftangent[1] = diffvector[1]-diffnormal[1];
      difftangent[2] = diffvector[2]-diffnormal[2];

      update1[0] = difftangent[0]*0.5*stepSize;
      update1[1] = difftangent[1]*0.5*stepSize;
      update1[2] = difftangent[2]*0.5*stepSize;

      double radiusOfCurvature = fabs(averageEdgeLength*averageEdgeLength/2.0);
      radiusOfCurvature /= normalcomponent;
      double w2 = 0.5*(1.0+tanh(F*(1/radiusOfCurvature-E)));

      update2[0] = 0.25*w2*diffnormal[0]*stepSize;
      update2[1] = 0.25*w2*diffnormal[1]*stepSize;
      update2[2] = 0.25*w2*diffnormal[2]*stepSize;
      
      // xk = pc is a physical point
      for (int d = -LeftBound; d <= RightBound; d++)
      {
        ImageType::PointType point;  

        // set point values
        for( int m = 0; m <3; m ++)
        {
          point[m] = pc[m]-d*normal[m]*stepSize;         
        }
        itk::ContinuousIndex<double, 3> cidx;    
        image->TransformPhysicalPointToContinuousIndex( point, cidx );
        if (image->GetLargestPossibleRegion().IsInside(cidx))
        {
          imgProfile[d+LeftBound] = static_cast<float>( imgInterpolator->EvaluateAtContinuousIndex( cidx ) );
          //fProfile[d+LeftBound] = fInterpolator->EvaluateAtContinuousIndex( cidx );
          //wProfile[d+LeftBound] = wInterpolator->EvaluateAtContinuousIndex( cidx );
          //csfProfile[d+LeftBound] = csfInterpolator->EvaluateAtContinuousIndex( cidx );
          //gmProfile[d+LeftBound] = gmInterpolator->EvaluateAtContinuousIndex( cidx );
          //wmProfile[d+LeftBound] = wmInterpolator->EvaluateAtContinuousIndex( cidx );

        }
        else
        {
          imgProfile[d+LeftBound] = 0;
          //fProfile[d+LeftBound] = 0;
          //wProfile[d+LeftBound] = 0;
          //csfProfile[d+LeftBound] = 0;
          //gmProfile[d+LeftBound] = 0;
          //wmProfile[d+LeftBound] = 0;

        }
      }

      // using the profiles to compute image force
      float imageforceiter;
      float imageforce = 1.0;

      for ( int d = 0; d <=RightBound; d++ )
      {
        int i = d+LeftBound;
      double tanhvalue = tanh((static_cast <float> (imgProfile[i])-lThreshold)/(lThreshold/4));
        //double tanhvalue = tanh((static_cast <float> (value*24)/static_cast <float> (t98))-4);
        imageforceiter = (0 < tanhvalue ? tanhvalue : 0);
        if ( iter > 50 && imageforceiter == 0)
        {
          imageforce = -0.05;
        }
        else if ( imageforceiter > 0 && imageforce > 0 )
        {
          imageforce *= imageforceiter;
        }

        //take care of areas around eyes 
        double tanhvalue2 = tanh((static_cast <float> (imgProfile[i])-uThreshold)/(uThreshold/4));
        //double tanhvalue2 = tanh((static_cast <float> (value*6)/static_cast <float> (t98))-4);
        if ( tanhvalue2 > 0 )
        {
          imageforce = -(fabs(imageforce));
          update1[0] *= relaxFactor;
          update1[1] *= relaxFactor;
          update1[2] *= relaxFactor;
          update2[0] *= relaxFactor;
          update2[1] *= relaxFactor;
          update2[2] *= relaxFactor;
        }
      }

      imageforce *= (stepSize*1.25);
      update3[0] = imageforce*normal[0];
      update3[1] = imageforce*normal[1];
      update3[2] = imageforce*normal[2];
            
      if ( iter > 1000 )
        {
        update[0] = update1[0]+update2[0]+update3[0]*0.75;
        update[1] = update1[1]+update2[1]+update3[1]*0.75;
        update[2] = update1[2]+update2[2]+update3[2]*0.75;
        }
      else
        {
        update[0] = (update1[0]+update2[0])+update3[0];
        update[1] = (update1[1]+update2[1])+update3[1];
        update[2] = (update1[2]+update2[2])+update3[2];
        }

      pc[0] += update[0];
      pc[1] += update[1];
      pc[2] += update[2];

      allPoints->SetPoint( k, pc[0], pc[1], pc[2] );

      change += sqrt(update[0]*update[0]+update[1]*update[1]+update[2]*update[2]);
      change1 += sqrt(update1[0]*update1[0]+update1[1]*update1[1]+update1[2]*update1[2]);
      change2 += sqrt(update2[0]*update2[0]+update2[1]*update2[1]+update2[2]*update2[2]);
      change3 += sqrt(update3[0]*update3[0]+update3[1]*update3[1]+update3[2]*update3[2]);

      }

      if (iter % 25 == 0)
      {
        std::cout << "EOI " << iter << ": ";
      std::cout << "  C = " << change <<  "  C1 = " << change1 <<  "  C2 = " << change2 <<  "  C3 = " << change3 << std::endl;
      }

      iter++;

    }
  std::cout << "Total iterations: " << iter << std::endl;

  PolyDataToLabelMap( polyData, label );

  allPoints = polyData->GetPoints();
  for (int k = 0; k < nPoints; k++)
  {
    double* point = polyData->GetPoint( k );
    point[0] = -point[0];
    point[1] = -point[1];
    allPoints->SetPoint( k, point[0], point[1], point[2] );
  }  

  vtkPolyDataWriter *wPoly = vtkPolyDataWriter::New();
  wPoly->SetFileName(brainSurface.c_str());
  wPoly->SetInput(polyData);    
  wPoly->Update();

  allPoints = polyData->GetPoints();
  for (int k = 0; k < nPoints; k++)
  {
    double* point = polyData->GetPoint( k );
    point[0] = -point[0];
    point[1] = -point[1];
    allPoints->SetPoint( k, point[0], point[1], point[2] );
  }  

  // binary dilation with radius 2
  LabelImageType::Pointer imgDilate;
  if (postDilationRadius > 0)
  {
    imgDilate = BinaryDilateFilter3D( label, postDilationRadius );
  }
  else
  {
    imgDilate = label;
  }
  wlabel->SetFileName( brainMask.c_str() );
  wlabel->SetInput( imgDilate );
  wlabel->Update();

  LabelImageType::Pointer finalLabel = LabelImageType::New();
  finalLabel->CopyInformation( image );
  finalLabel->SetRegions( finalLabel->GetLargestPossibleRegion() );
  finalLabel->Allocate();
  finalLabel->FillBuffer( 0 );

  for (itImg.GoToBegin(); !itImg.IsAtEnd(); ++itImg)
  {
    ImageType::IndexType idx = itImg.GetIndex();
    if ( imgDilate->GetPixel(idx) == 0 )
    {
      csf->SetPixel( idx, 0 );
      gm->SetPixel( idx, 0 );
      wm->SetPixel( idx, 0 );
    }
    float csf0 = csf->GetPixel(idx);
    float gm0 = gm->GetPixel(idx);
    float wm0 = wm->GetPixel(idx);
    if (csf0 > gm0 && csf0 > wm0)
    {
      finalLabel->SetPixel( idx, 1 );
    }
    if (gm0 > csf0 && gm0 > wm0)
    {
      finalLabel->SetPixel( idx, 2 );
    }
    if (wm0 > gm0 && wm0 > csf0)
    {
      finalLabel->SetPixel( idx, 3 );
    }
  }

  itk::ImageFileWriter<FloatImageType>::Pointer fWriter = itk::ImageFileWriter<FloatImageType>::New();
  fWriter->SetInput( csf );
  std::string fname = "csf-" + brainMask;
  fWriter->SetFileName( fname.c_str() );
  fWriter->Update();
  fWriter->SetInput( gm );
  fname = "gm-" + brainMask;
  fWriter->SetFileName( fname.c_str() );
  fWriter->Update();
  fWriter->SetInput( wm );
  fname = "wm-" + brainMask;
  fWriter->SetFileName( fname.c_str() );
  fWriter->Update();

  fname = "label-" + brainMask;
  wlabel->SetInput( finalLabel );
  wlabel->SetFileName( fname.c_str() );
  wlabel->Update();

  return 0;
}


