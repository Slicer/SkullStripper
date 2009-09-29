/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkFuzzyClassificationImageFilter.txx,v $
  Language:  C++
  Date:      $Date: 2009/07/02 13:59:34 $
  Version:   $Revision: 1.4 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

  Portions of this code are covered under the VTK copyright.
  See VTKCopyright.txt or http://www.kitware.com/VTKCopyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkFuzzyClassificationImageFilter_txx
#define __itkFuzzyClassificationImageFilter_txx

#include "itkFuzzyClassificationImageFilter.h"
#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIterator.h"
#include "itkNumericTraits.h"
#include "itkObjectFactory.h"
#include "itkProgressReporter.h"

namespace itk
{

bool func4sortpairs( const std::pair<int, float> & a, const std::pair<int, float> & b)
{
  if (a.second < b.second)
    {
    return true;
    }
  else
    {
    return false;
    }
}

/**
 *
 */
template <class TInputImage, class TOutputImage>
FuzzyClassificationImageFilter<TInputImage, TOutputImage>
::FuzzyClassificationImageFilter()
{
  m_NumberOfClasses = 3;
  m_BiasCorrectionOption = 0;
  m_ImageMask = NULL;

  typename TOutputImage::Pointer output = TOutputImage::New();
  this->ProcessObject::SetNumberOfOutputs( m_NumberOfClasses );
  this->ProcessObject::SetNthOutput(1, output.GetPointer());

  this->m_ClassCentroid.resize( m_NumberOfClasses );
  this->m_ClassStandardDeviation.resize( m_NumberOfClasses );
}


/**
 *
 */
template <class TInputImage, class TOutputImage>
void
FuzzyClassificationImageFilter<TInputImage, TOutputImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);

  os << indent << "Number Of Classes: "
     << m_NumberOfClasses
     << std::endl;
  os << indent << "Bias Correction Option: "
     << m_BiasCorrectionOption
     << std::endl;
}


/**
 *
 */
template <class TInputImage, class TOutputImage>
void 
FuzzyClassificationImageFilter<TInputImage, TOutputImage>
::GenerateData( )
{
  itkDebugMacro(<<"Actually executing");

  this->m_ClassStandardDeviation.resize( m_NumberOfClasses );

  // Get the input and output pointers
  InputImageConstPointer  inputPtr  = this->GetInput();

  // generate a masked image
  InputImagePointer img = InputImageType::New();
  img->CopyInformation( inputPtr );
  img->SetRegions( inputPtr->GetLargestPossibleRegion() );
  img->Allocate();

  itk::ImageRegionIteratorWithIndex<InputImageType> it( img, img->GetLargestPossibleRegion() );
  if (m_ImageMask)
    {
    for (it.GoToBegin(); !it.IsAtEnd(); ++it)
      {
      typename InputImageType::IndexType idx = it.GetIndex();
      typename InputImageType::PointType pt;
      img->TransformIndexToPhysicalPoint( idx, pt );
      m_ImageMask->TransformPhysicalPointToIndex( pt, idx );
      if ( !m_ImageMask->GetLargestPossibleRegion().IsInside(idx) )
      {
        it.Set( 0 );
      }
      else if (m_ImageMask->GetPixel(idx) == 0)
      {
        it.Set( 0 );
      }
      else
      {
        it.Set( inputPtr->GetPixel(it.GetIndex()) );
      }
      }
    }
  else
    {
    for (it.GoToBegin(); !it.IsAtEnd(); ++it)
      {
      typename InputImageType::IndexType idx = it.GetIndex();
      it.Set( inputPtr->GetPixel(idx) );
      }
    }

  // allocate local image variables
  //Initialize the image of gain field g[] to 1.
  InputImagePointer gain_field_g = InputImageType::New();  
  gain_field_g->CopyInformation (img);
  gain_field_g->SetRegions (img->GetLargestPossibleRegion());
  gain_field_g->Allocate();
  gain_field_g->FillBuffer (1.0f);

  //Initialize the images of the membership functions u1[], u2[], u3[]
  //and a updated storage u1n[], u2n[], u3n[].
  vcl_vector<InputImagePointer> mem_fun_u (this->m_NumberOfClasses);
  vcl_vector<InputImagePointer> mem_fun_un (this->m_NumberOfClasses);
  for (int k = 0; k < this->m_NumberOfClasses; k++) {
    mem_fun_u[k] = InputImageType::New();
    mem_fun_u[k] -> CopyInformation( img );
    mem_fun_u[k] -> SetRegions (img->GetLargestPossibleRegion());
    mem_fun_u[k] -> Allocate();
    mem_fun_u[k]->FillBuffer (0.0f);
    mem_fun_un[k] = InputImageType::New();
    mem_fun_un[k] -> CopyInformation( img );
    mem_fun_un[k] -> SetRegions (img->GetLargestPossibleRegion());
    mem_fun_un[k] -> Allocate();
    mem_fun_un[k]->FillBuffer (0.0f);
  }

  vcl_vector<float> centroid_v;

  if (this->m_BiasCorrectionOption == 0 || this->m_BiasCorrectionOption == 1 || this->m_BiasCorrectionOption == 2) 
    {
    afcm_segmentation (img, this->m_NumberOfClasses, 200, 25.0f, 1500.0f,
                       0, this->m_BiasCorrectionOption, 
                       0.8, 0.8,
                       0.01, gain_field_g,
                       mem_fun_u, mem_fun_un, this->m_ClassCentroid);
    }
  else 
    {
    afcm_segmentation_grid (img, this->m_NumberOfClasses, 200, 25.0f, 1500.0f,
                            0, this->m_BiasCorrectionOption, 
                            0.8, 0.8,
                            0.01, 3,
                            gain_field_g,
                            mem_fun_u, mem_fun_un, this->m_ClassCentroid);
    
    }

  // mask out output
  if (m_ImageMask)
  {
    for (it.GoToBegin(); !it.IsAtEnd(); ++it)
    {
      typename InputImageType::IndexType idx = it.GetIndex();
      typename InputImageType::PointType pt;
      img->TransformIndexToPhysicalPoint( idx, pt );
      m_ImageMask->TransformPhysicalPointToIndex( pt, idx );
      if ( m_ImageMask->GetLargestPossibleRegion().IsInside(idx) )
      {
        if (m_ImageMask->GetPixel(idx) != 0)
        {
          continue;
        }
      }
      idx = it.GetIndex();
      for (int k = 0; k < this->m_NumberOfClasses; k++)
      {
        mem_fun_u[k]->SetPixel( idx, 0 );
      }
    }
  }

  // copy bias field;
  this->m_BiasField = InputImageType::New();
  this->m_BiasField->CopyInformation( gain_field_g );
  this->m_BiasField->SetRegions( gain_field_g->GetLargestPossibleRegion() );
  this->m_BiasField->Allocate();

  itk::ImageRegionIteratorWithIndex<InputImageType> itg(gain_field_g, gain_field_g->GetLargestPossibleRegion());
  for (itg.GoToBegin(); !itg.IsAtEnd(); ++itg)
  {
    typename InputImageType::IndexType idx = itg.GetIndex();
    this->m_BiasField->SetPixel( idx, itg.Get() );
  }

  // map things to output of the filter
  // sort classes into ascending order the class centroid
  std::vector< std::pair<int, float> > classCentroidWithIndex( this->m_NumberOfClasses );
  std::vector<float> stdCopy( this->m_NumberOfClasses );
  for (int k = 0; k < this->m_NumberOfClasses; k++)
  {
    classCentroidWithIndex[k].first = k;
    classCentroidWithIndex[k].second = this->m_ClassCentroid[k];
    stdCopy[k] = this->m_ClassStandardDeviation[k];
  }

  std::sort(classCentroidWithIndex.begin(), classCentroidWithIndex.end(),  func4sortpairs );

  this->SetNumberOfOutputs( this->m_NumberOfClasses );

  for (int k = 0; k < this->m_NumberOfClasses; k++)
  {
    OutputImagePointer oPtr = OutputImageType::New();
    oPtr->CopyInformation( inputPtr );
    oPtr->SetRegions( oPtr->GetLargestPossibleRegion() );
    this->SetNthOutput(k, oPtr);
  }

  this->AllocateOutputs();

  for (int k = 0; k < this->m_NumberOfClasses; k++)
  {
    this->m_ClassCentroid[k] = classCentroidWithIndex[k].second;
    this->m_ClassStandardDeviation[k] = stdCopy[classCentroidWithIndex[k].first];

    OutputImagePointer oPtr = this->GetOutput( classCentroidWithIndex[k].first );
    itk::ImageRegionIteratorWithIndex<OutputImageType> itcpy(oPtr, oPtr->GetLargestPossibleRegion());
    for (itcpy.GoToBegin(); !itcpy.IsAtEnd(); ++itcpy)
    {
      //std::cout << itcpy.GetIndex() << std::endl;
      itcpy.Set( mem_fun_u[k]->GetPixel( itcpy.GetIndex() ) );
    }
  }

  std::cout << "class centroid: " ;
  for (int k = 0; k < this->m_NumberOfClasses; k++)
  {
    std::cout << this->m_ClassCentroid[k] << ", ";
  }
  std::cout << "\n";

  std::cout << "class std: " ;
  for (int k = 0; k < this->m_NumberOfClasses; k++)
  {
    std::cout << this->m_ClassStandardDeviation[k] << ", ";
  }
  std::cout << "\n";

}


template <class TInputImage, class TOutputImage>
void 
FuzzyClassificationImageFilter<TInputImage, TOutputImage>
::afcm_segmentation (InputImagePointer img_y, 
                        const int n_class, const int n_bin,
                        const float low_th, const float high_th,
                        const float bg_thresh,
                        const int gain_fit_option, 
                        const float gain_th, const float gain_min,
                        const float conv_thresh,
                        InputImagePointer& gain_field_g,
                        vcl_vector<InputImagePointer>& mem_fun_u, 
                        vcl_vector<InputImagePointer>& mem_fun_un, 
                        vcl_vector<float>& centroid_v)
{
  //Initializtion:  
  //Find the initial guess of the centroid for different classes v1, v2, v3.
  compute_init_centroid (img_y, n_class, n_bin, low_th, high_th, centroid_v);

  //Iteration: five steps:
  bool conv;
  int iter = 0;
  do {
    // // vcl_printf ("\nIteration %d:\n", iter);
    //1) Compute new membership functions u1[], u2[], u3[].
    compute_new_mem_fun_u (centroid_v, gain_field_g, img_y, bg_thresh, mem_fun_u);

    ///debug:
    ///save_mem_fun_u ("output_im", mem_fun_u);

    //2) Compute the new centroids v1, v2, v3.
    compute_new_centroids (mem_fun_u, gain_field_g, img_y, centroid_v);

    //3) Compute a new gain field g[]:
    //   Initially, we assume g[]=1 is know and fixed in our case.
    //   Here we update it by a regression fit of the white matter (mem_fun_u[2])
    if (gain_fit_option == 1 || gain_fit_option == 2) {
      compute_new_gain_field (mem_fun_u, img_y, gain_field_g, 
                              gain_fit_option, gain_th);
      
      //debug: save gain field file for debugging.
      ///save_01_img8 ("gain_field_g.mhd", gain_field_g);
    }

    //4) Compute a new membership function u1n[], u2n[], u3n[] using step 1.
    compute_new_mem_fun_u (centroid_v, gain_field_g, img_y, bg_thresh, mem_fun_un);
    
    //5) Test convergence.
    //   if max(u1n[]-u1[], u2n[]-u2[], u3n[]-u3[]) < 0.01, converge and finish.
    conv = test_convergence (mem_fun_u, mem_fun_un, conv_thresh);
    iter++;
  }
  while (conv == false);

  itk::ImageRegionIteratorWithIndex<InputImageType> it( img_y, img_y->GetLargestPossibleRegion() );
  std::vector <float> count( this->m_NumberOfClasses );
  for (int k = 0; k < this->m_NumberOfClasses; k++)
  {
    this->m_ClassStandardDeviation[k] = 0;
    count[k] = 0;
  }

  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    typename InputImageType::IndexType idx = it.GetIndex();
    float p = static_cast<float>( it.Get() );

    for (int k = 0; k < this->m_NumberOfClasses; k++)
    {
      if (mem_fun_u[k]->GetPixel(idx) <= 0.5)
      {
        continue;
      }
      else
      {
        count[k] += 1.0;
        p -= this->m_ClassCentroid[k];
        this->m_ClassStandardDeviation[k] += (p*p);
      }
    }
  }

  for (int k = 0; k < this->m_NumberOfClasses; k++)
  {
    this->m_ClassStandardDeviation[k] /= (count[k]-1);
    this->m_ClassStandardDeviation[k] = sqrt( this->m_ClassStandardDeviation[k] );
  }
}

//===================================================================

template <class TInputImage, class TOutputImage>
void
FuzzyClassificationImageFilter<TInputImage, TOutputImage>
::compute_init_centroid (InputImagePointer image, 
                            const int n_class, const int n_bin,
                            const float low_th, const float high_th,
                            vcl_vector<float>& centroid_v)
{
  // vcl_printf ("\ncompute_init_centroid():\n");
  // vcl_printf ("  n_class %d, n_bin %d, low_t %f, high_t %f.\n",
  //            n_class, n_bin, low_th, high_th);

  // Exclude pixels whose intensity is outside the range [low_th, high_th]
  // from computation. The range is divided into n_bin bins to computer the kernal 
  // estimator, and then used to compute the parameters
  vcl_vector<float> histVector;
  vcl_vector<float> binMin;
  vcl_vector<float> binMax;
  int nBinHistogram = 0;

  // let program decide how many bins are there for the histogram
  compute_histogram (image, histVector, binMax, binMin, nBinHistogram);
  assert (histVector.size() == static_cast<unsigned long>(nBinHistogram));
  assert (binMin.size() == static_cast<unsigned long>(nBinHistogram));
  assert (binMax.size() == static_cast<unsigned long>(nBinHistogram));  

  // the variable n_bin below is used to devide the range of intensity used for kernal
  // estimator calculation.  
  vcl_vector<float> kernalEstimator;
  kernalEstimator.resize (n_bin);
  assert (n_bin != 1);
  float deltaX = (high_th-low_th)/(n_bin-1);
  vcl_vector<float> xVector (n_bin);
  for (int k = 0; k < n_bin; k++) {
    xVector[k] = low_th + k*deltaX;
  }

  bool Done = false;
  float h0 = 0;
  float h1 = 50;
  float h = (h0 + h1)/2;
  while (!Done) {
    for (int k = 0; k < n_bin; k++) {
      kernalEstimator[k] = 0;      
      for (int n = 0; n < nBinHistogram; n ++ ) {
        float b = binMin[n];
        if ( b < low_th )
          continue;
        float d = binMax[n];
        if ( d > high_th )
          continue;
        d = (d  +  b) / 2.0;
        d = exp(-(xVector[k]-d)*(xVector[k]-d)/(2*h*h));
        kernalEstimator[k] = kernalEstimator[k] + d * histVector[n];
      }
    }
    int C = CountMode (kernalEstimator);
    if (C > n_class)
      h0 = h;
    else
      h1 = h;
    float hNew = (h0 + h1)/2;
    if (fabs(hNew-h) < 0.01)
      Done = true;
    h = hNew;
  }

  centroid_v.clear();

  int kernalLength = kernalEstimator.size();
  assert (kernalLength > 0);
  assert (xVector.size() == static_cast<unsigned long>(kernalLength));

  for (int k = 1; k < kernalLength-1; k++) {
    if (kernalEstimator[k] < kernalEstimator[k-1])
      continue;
    if (kernalEstimator[k] < kernalEstimator[k+1])
      continue;
    centroid_v.push_back (xVector[k]);
    if (static_cast<int>(centroid_v.size()) >= n_class)
      break;
  }

  // vcl_printf ("  centroid_v: C0 %f,   C1 %f,   C2 %f.\n\n", 
  //            centroid_v[0], centroid_v[1], centroid_v[2]);
}

// Compute new membership functions u1[], u2[], u3[].
template <class TInputImage, class TOutputImage>
void 
FuzzyClassificationImageFilter<TInputImage, TOutputImage>
::compute_new_mem_fun_u (const vcl_vector<float>& centroid_v,
                            InputImagePointer gain_field_g, 
                            InputImagePointer img_y,
                            const float bg_thresh,
                            vcl_vector<InputImagePointer>& mem_fun_u)
{
  // vcl_printf ("  compute_new_mem_fun_u(): \n");

  const int n_class = mem_fun_u.size();
  
  for (int k = 0; k < n_class; k++) {
    //iterate through each pixel j:
    typedef itk::ImageRegionConstIterator< InputImageType > ConstIteratorType;
    typedef itk::ImageRegionIterator< InputImageType > IteratorType;
    ConstIteratorType ity (img_y, img_y->GetRequestedRegion());
    ConstIteratorType itg (gain_field_g, gain_field_g->GetRequestedRegion());
    IteratorType itu (mem_fun_u[k], mem_fun_u[k]->GetRequestedRegion());

    for (ity.GoToBegin(), itg.GoToBegin(), itu.GoToBegin(); 
      !ity.IsAtEnd(); 
      ++ity, ++itg, ++itu) {
        //Skip background pixels.
        float img_y_j = ity.Get();
        if (img_y_j < bg_thresh)
          continue;

        float gain_field_g_j = itg.Get();

        ///double numerator = img_y[j] - centroid_v[k] * gain_field_g[j];
        double numerator = img_y_j - centroid_v[k] * gain_field_g_j;

        if (numerator != 0)
          numerator = 1 / (numerator * numerator);
        else if (gain_field_g_j == 1) {
          //The divide-by-zero happens when img_y[j] == centroid_v[k].
          //In this case, the membership function should be 1 for this class and 
          //0 for all other classes (for normalization).
          itu.Set (1);
          continue; //Done for the current pixel.
        }
        else {
          //Keep numerator as 0 for this unlikely-to-happen case.
          ///assert (0);
        }

        double denominator = 0;
        for (int l = 0; l < n_class; l++) {
          ///double denominator_l = img_y[j] - centroid_v[l] * gain_field_g[j];
          double denominator_l = img_y_j - centroid_v[l] * gain_field_g_j;

          if (denominator_l != 0) 
            denominator_l = 1 / (denominator_l * denominator_l);
          else {
            //This is the case when the same pixel of other class than k has mem_fun == 1.
            //Set the membership function to 0.          
            itu.Set (0);
            continue;
          }

          denominator += denominator_l;
        }
        ///mem_fun_u[k][j] = numerator / denominator;
        assert (denominator != 0);
        itu.Set (numerator / denominator);
    }
  }
}

// Compute the new centroids v1, v2, v3.
template <class TInputImage, class TOutputImage>
void 
FuzzyClassificationImageFilter<TInputImage, TOutputImage>
::compute_new_centroids (const vcl_vector<InputImagePointer>& mem_fun_u, 
                            InputImagePointer& gain_field_g, 
                            InputImagePointer& img_y, 
                            vcl_vector<float>& centroid_v)
{
  // vcl_printf ("  compute_new_centroids(): ");
  const int n_class = mem_fun_u.size();

  for (int k = 0; k < n_class; k++) {
    //iterate through each pixel j:
    typedef itk::ImageRegionConstIterator< InputImageType > ConstIteratorType;
    ConstIteratorType ity (img_y, img_y->GetRequestedRegion());
    ConstIteratorType itg (gain_field_g, gain_field_g->GetRequestedRegion());
    ConstIteratorType itu (mem_fun_u[k], mem_fun_u[k]->GetRequestedRegion());

    double numerator = 0;
    double denominator = 0;
    for (ity.GoToBegin(), itg.GoToBegin(), itu.GoToBegin(); !ity.IsAtEnd(); ++ity, ++itg, ++itu) {      
      float mem_fun_u_kj = itu.Get();
      assert (vnl_math_isnan (mem_fun_u_kj) == false);
      float gain_field_g_j = itg.Get();
      assert (vnl_math_isnan (gain_field_g_j) == false);
      float img_y_j = ity.Get();
      assert (vnl_math_isnan (img_y_j) == false);

      ///double numerator = mem_fun_u[k][j] * mem_fun_u[k][j] * gain_field_g[j] * img_y[j];
      numerator += mem_fun_u_kj * mem_fun_u_kj * gain_field_g_j * img_y_j;
      assert (vnl_math_isnan (numerator) == false);
      ///double denominator = mem_fun_u[k][j] * mem_fun_u[k][j] * gain_field_g[j] * gain_field_g[j];
      denominator += mem_fun_u_kj * mem_fun_u_kj * gain_field_g_j * gain_field_g_j;
      assert (vnl_math_isnan (denominator) == false);
    }

    if (denominator == 0) {
      if (numerator == 0)
        centroid_v[k] = 0;
      else {
        // vcl_printf ("  Error: divide by 0!\n");
        centroid_v[k] = itk::NumericTraits<float>::min();
      }
    }
    else {
      centroid_v[k] = numerator / denominator;
    }
  }
  // vcl_printf ("C0 %f,   C1 %f,   C2 %f.\n", 
  //            centroid_v[0], centroid_v[1], centroid_v[2]);
}

// Compute a new gain field g[]:
//   Initially, we assume g[]=1 is know and fixed in our case.
//   Here we update it by a regression fit of the white matter (mem_fun_u[2])
template <class TInputImage, class TOutputImage>
void
FuzzyClassificationImageFilter<TInputImage, TOutputImage>
::compute_new_gain_field (vcl_vector<InputImagePointer>& mem_fun_u, 
                             InputImagePointer& img_y, 
                             InputImagePointer& gain_field_g,
                             const int gain_fit_option,
                             const float gain_th)
{
  assert (gain_fit_option == 1 || gain_fit_option == 2);
  // vcl_printf ("  compute_new_gain_field():\n");
  // vcl_printf ("    %s fitting, gain_th %f.\n",
  //            (gain_fit_option==1) ? "linear" : "quadratic", 
  //            gain_th);

  //Quadratic regression fiting to get the parameter B
  vnl_matrix<double> B;

  if (gain_fit_option == 1) {
    img_regression_linear (mem_fun_u[2], gain_th, B);
    //Use B to compute a new gain_field_g[]
    compute_linear_fit_img (B, gain_field_g);
  }
  else if (gain_fit_option == 2) {
    img_regression_quadratic (mem_fun_u[2], gain_th, B);
    //Use B to compute a new gain_field_g[]
    compute_quadratic_fit_img (B, gain_field_g);
  }
}

// Test convergence.
template <class TInputImage, class TOutputImage>
bool
FuzzyClassificationImageFilter<TInputImage, TOutputImage>
::test_convergence (const vcl_vector<InputImagePointer>& mem_fun_u, 
                       const vcl_vector<InputImagePointer>& mem_fun_un, 
                       const float conv_thresh)
{
  // vcl_printf ("  test_convergence(): ");

  const int n_class = mem_fun_u.size();
  float max_value = 0;

  for (int k = 0; k < n_class; k++) {
    //iterate through each pixel j:
    typedef itk::ImageRegionConstIterator< InputImageType > ConstIteratorType;
    ConstIteratorType it (mem_fun_u[k], mem_fun_u[k]->GetRequestedRegion());
    ConstIteratorType itn (mem_fun_un[k], mem_fun_un[k]->GetRequestedRegion());

    for (it.GoToBegin(), itn.GoToBegin(); !it.IsAtEnd(); ++it, ++itn) {
      float mem_fun_u_kj = it.Get();
      assert (vnl_math_isnan (mem_fun_u_kj) == false);
      float mem_fun_un_kj = itn.Get();
      assert (vnl_math_isnan (mem_fun_un_kj) == false);

      ///float diff = member_fun_u[k][j] - member_fun_un[k][j];
      float diff = mem_fun_u_kj - mem_fun_un_kj;
      diff = vcl_fabs (diff);
      if (diff > max_value)
        max_value = diff;
    }    
  }

  // vcl_printf ("max_value %f (conv_th %f).\n", max_value, conv_thresh);

  if (max_value < conv_thresh)
    return true;
  else
    return false;
}


template <class TInputImage, class TOutputImage>
int 
FuzzyClassificationImageFilter<TInputImage, TOutputImage>
::CountMode (const vcl_vector<float>& v)
{
  int c = 0;
  for (unsigned int k = 1; k < v.size()-1; k++) {
    if ( v[k] > v[k-1] && v[k] > v[k+1])
      c++;
  }
  return (c);
}


 
template <class TInputImage, class TOutputImage>
void 
FuzzyClassificationImageFilter<TInputImage, TOutputImage>
::img_regression_linear (InputImagePointer& image,
                            const float thresh,
                            vnl_matrix<double>& B)
{
  // vcl_printf ("    img_regression_linear(): \n");
  int i;

  //Put image intensity into y[].
  //Put image pixel coordinates into x1[], x2[], x3[].
  typedef itk::ImageRegionIteratorWithIndex < InputImageType > IndexedIteratorType;
  IndexedIteratorType iit (image, image->GetRequestedRegion());
  iit.GoToBegin();
  assert (iit.GetIndex().GetIndexDimension() == 3);
  
  //Determine the total number of pixels > thresh.
  ///InputImageType::SizeType requestedSize = image->GetRequestedRegion().GetSize();  
  ///int SZ = requestedSize[0] * requestedSize[1] * requestedSize[2];
  int SZ = 0;
  for (i=0, iit.GoToBegin(); !iit.IsAtEnd(); ++iit) {
    typename InputImageType::IndexType idx = iit.GetIndex();
    float pixel = iit.Get();
    if (pixel > thresh)
      SZ++;
  }
  // vcl_printf ("      # pixels > thresh (%f) = %d\n", thresh, SZ);
  
  vnl_matrix<double> y (SZ,1);
  vnl_matrix<double> x1 (SZ,1);
  vnl_matrix<double> x2 (SZ,1);
  vnl_matrix<double> x3 (SZ,1);

  for (i=0, iit.GoToBegin(); !iit.IsAtEnd(); ++iit) {
    typename InputImageType::IndexType idx = iit.GetIndex();
    float pixel = iit.Get();
    if (pixel > thresh) {
      assert (i < SZ);
      y(i, 0) = pixel;
      int x_1 = idx[0];
      int x_2 = idx[1];
      int x_3 = idx[2];
      x1(i, 0) = x_1;
      x2(i, 0) = x_2;
      x3(i, 0) = x_3;
      i++;
    }
  }

  //Prepare the design matrix X
  vnl_matrix<double> X (SZ,4);
  X.set_column (0, 1.0);
  X.update (x1, 0, 1);
  X.update (x2, 0, 2);
  X.update (x3, 0, 3);
  ///vcl_cerr << X;  
  x1.clear();
  x2.clear();
  x3.clear();

  vnl_matrix<double> Xt = X.transpose();
  vnl_matrix<double> Xt_X = Xt * X; //(x'*x)
  X.clear();
  vnl_matrix<double> Xt_y = Xt * y; //(x'*y)
  Xt.clear();
  y.clear();
  //Solve for the linear normal equation: (x'*x) * b = (x'*y)
  vnl_matrix<double> Xt_X_inv = vnl_matrix_inverse<double>(Xt_X);
  Xt_X.clear();
  //b = inv(x'*x) * (x'*y);
  B = Xt_X_inv * Xt_y;
  
  // vcl_printf ("B: \n");
  // vcl_cerr << B;
}

//Use B to compute a new fitting
template <class TInputImage, class TOutputImage>
void 
FuzzyClassificationImageFilter<TInputImage, TOutputImage>
::compute_linear_fit_img (const vnl_matrix<double>& B, 
                             InputImagePointer& fit_image)
{
  // vcl_printf ("    compute_linear_fit_img(): \n");

  //Traverse through the fit_image and compute a new quadratic value via B.
  //Image coordinates into x1[], x2[], x3[].
  typedef itk::ImageRegionIteratorWithIndex < InputImageType > IndexedIteratorType;
  IndexedIteratorType iit (fit_image, fit_image->GetRequestedRegion());
  assert (iit.GetIndex().GetIndexDimension() == 3);
  assert (B.rows() == 4);

  for (iit.GoToBegin(); !iit.IsAtEnd(); ++iit) {
    typename InputImageType::IndexType idx = iit.GetIndex();
    int x_1 = idx[0];
    int x_2 = idx[1];
    int x_3 = idx[2];
    double pixel = B(0,0) + B(1,0)*x_1 + B(2,0)*x_2 + B(3,0)*x_3;
    iit.Set (pixel);
  }
}

template <class TInputImage, class TOutputImage>
void 
FuzzyClassificationImageFilter<TInputImage, TOutputImage>
::img_regression_quadratic (InputImagePointer& image,
                               const float thresh,
                               vnl_matrix<double>& B)
{
  // vcl_printf ("    img_regression_quadratic(): \n");
  int i;

  //Put image intensity into y[].
  //Put image pixel coordinates into x1[], x2[], x3[].
  typedef itk::ImageRegionIteratorWithIndex < InputImageType > IndexedIteratorType;
  IndexedIteratorType iit (image, image->GetRequestedRegion());
  assert (iit.GetIndex().GetIndexDimension() == 3);
  
  //Determine the total number of pixels > thresh.
  ///InputImageType::SizeType requestedSize = image->GetRequestedRegion().GetSize();  
  ///int SZ = requestedSize[0] * requestedSize[1] * requestedSize[2];
  int SZ = 0;  
  for (i=0, iit.GoToBegin(); !iit.IsAtEnd(); ++iit) {
    typename InputImageType::IndexType idx = iit.GetIndex();
    float pixel = iit.Get();
    if (pixel > thresh)
      SZ++;
  }
  // vcl_printf ("      # pixels > thresh (%f) = %d\n", thresh, SZ);

  vnl_matrix<double> y (SZ,1);
  vnl_matrix<double> x1 (SZ,1);
  vnl_matrix<double> x2 (SZ,1);
  vnl_matrix<double> x3 (SZ,1);
  for (i=0, iit.GoToBegin(); !iit.IsAtEnd(); ++iit) {
    typename InputImageType::IndexType idx = iit.GetIndex();
    double pixel = iit.Get();
    if (pixel > thresh) {
      assert (i < SZ);
      y(i, 0) = pixel;
      int x_1 = idx[0];
      int x_2 = idx[1];
      int x_3 = idx[2];
      x1(i, 0) = (double) x_1;
      x2(i, 0) = (double) x_2;
      x3(i, 0) = (double) x_3;
      i++;
    }
  }

  //Prepare the design matrix X
  vnl_matrix<double> X (SZ,10);
  X.set_column (0, 1.0);
  X.update (x1, 0, 1);
  X.update (x2, 0, 2);
  X.update (x3, 0, 3);
  x1.clear();
  x2.clear();
  x3.clear();

  vnl_matrix<double> x1x2 (SZ,1);
  vnl_matrix<double> x1x3 (SZ,1);
  vnl_matrix<double> x2x3 (SZ,1);  
  for (i=0, iit.GoToBegin(); !iit.IsAtEnd(); ++iit) {
    typename InputImageType::IndexType idx = iit.GetIndex();
    double pixel = iit.Get();
    if (pixel > thresh) {
      assert (i < SZ);
      int x_1 = idx[0];
      int x_2 = idx[1];
      int x_3 = idx[2];
      x1x2 (i, 0) = x_1 * x_2;
      x1x3 (i, 0) = x_1 * x_3;
      x2x3 (i, 0) = x_2 * x_3;
      i++;
    }
  }
  X.update (x1x2, 0, 4);
  X.update (x1x3, 0, 5);
  X.update (x2x3, 0, 6);
  x1x2.clear();
  x1x3.clear();
  x2x3.clear();
  
  vnl_matrix<double> x1x1 (SZ,1);
  vnl_matrix<double> x2x2 (SZ,1);
  vnl_matrix<double> x3x3 (SZ,1);
  for (i=0, iit.GoToBegin(); !iit.IsAtEnd(); ++iit) {
    typename InputImageType::IndexType idx = iit.GetIndex();
    double pixel = iit.Get();
    if (pixel > thresh) {
      assert (i < SZ);
      int x_1 = idx[0];
      int x_2 = idx[1];
      int x_3 = idx[2];
      x1x1 (i, 0) = x_1 * x_1;
      x2x2 (i, 0) = x_2 * x_2;
      x3x3 (i, 0) = x_3 * x_3;
      i++;
    }
  }
  X.update (x1x1, 0, 7);
  X.update (x2x2, 0, 8);
  X.update (x3x3, 0, 9);
  x1x1.clear();
  x2x2.clear();
  x3x3.clear();

  ///// vcl_printf ("X: \n");
  ///vcl_cerr << X;

  vnl_matrix<double> Xt = X.transpose();
  vnl_matrix<double> Xt_X = Xt * X; //(x'*x)
  X.clear();
  vnl_matrix<double> Xt_y = Xt * y; //(x'*y)
  Xt.clear();
  y.clear();
  //Solve for the linear normal equation: (x'*x) * b = (x'*y)
  vnl_matrix<double> Xt_X_inv = vnl_matrix_inverse<double>(Xt_X);  
  Xt_X.clear();
  //b = inv(x'*x) * (x'*y);
  B = Xt_X_inv * Xt_y;
  
  // vcl_printf ("B: \n");
  // vcl_cerr << B;
}

//Use B to compute a new fitting
template <class TInputImage, class TOutputImage>
void 
FuzzyClassificationImageFilter<TInputImage, TOutputImage>
::compute_quadratic_fit_img (const vnl_matrix<double>& B,
                                InputImagePointer& fit_image)
{
  // vcl_printf ("    compute_quadratic_fit_img(): \n");

  //Traverse through the fit_image and compute a new quadratic value via B.
  //Image coordinates into x1[], x2[], x3[].
  typedef itk::ImageRegionIteratorWithIndex < InputImageType > IndexedIteratorType;
  IndexedIteratorType iit (fit_image, fit_image->GetRequestedRegion());
  assert (iit.GetIndex().GetIndexDimension() == 3);
  assert (B.rows() == 10);

  for (iit.GoToBegin(); !iit.IsAtEnd(); ++iit) {
    typename InputImageType::IndexType idx = iit.GetIndex();
    int x_1 = idx[0];
    int x_2 = idx[1];
    int x_3 = idx[2];
    double pixel = B(0,0) + B(1,0)*x_1 + B(2,0)*x_2 + B(3,0)*x_3 +
                   B(4,0)*x_1*x_2 + B(5,0)*x_1*x_3 + B(6,0)*x_2*x_3 +
                   B(7,0)*x_1*x_1 + B(8,0)*x_2*x_2 + B(9,0)*x_3*x_3;
    iit.Set (pixel);
  }
}


template <class TInputImage, class TOutputImage>
void 
FuzzyClassificationImageFilter<TInputImage, TOutputImage>
::afcm_segmentation_grid (InputImagePointer img_y, 
                          const int n_class, const int n_bin,
                          const float low_th, const float high_th,
                          const float bg_thresh,
                          const int gain_fit_option, 
                          const float gain_th, const float gain_min,
                          const float conv_thresh,
                          const int n_grid,
                          InputImagePointer& gain_field_g,
                          vcl_vector<InputImagePointer>& mem_fun_u, 
                          vcl_vector<InputImagePointer>& mem_fun_un, 
                          vcl_vector<float>& centroid_v)
{
  int i;
  assert (gain_fit_option == 3 || gain_fit_option == 4);

  //Detect the bounding box of the non-background brain block B.
  int xmin=0;
  int ymin=0;
  int zmin=0;
  int xmax=0;
  int ymax=0;
  int zmax=0;
  
  //Space division: into nxnxn: 3x3x3 or 4x4x4 blocks.
  vcl_vector<InputImagePointer> img_y_grid;
  vcl_vector<typename InputImageType::IndexType> grid_center_index;
  compute_grid_imgs (img_y, xmin, ymin, zmin, xmax, ymax, zmax, n_grid, 
                     img_y_grid, grid_center_index);

  //Allocate space for the centroid_v_grid[] and centroid_vn_grid.
  const int total_grids = (int) img_y_grid.size();
  vcl_vector<vcl_vector<float> > centroid_v_grid (total_grids);
  vcl_vector<float> centroid_vn_grid (total_grids);

  //Allocate space for the gain_field_g_grid[].
  vcl_vector<InputImagePointer> gain_field_g_grid (total_grids);

  //Allocate space for the mem_fun_u_grid[][] and mem_fun_un_grid[][]
  vcl_vector<vcl_vector<InputImagePointer> > mem_fun_u_grid (total_grids);
  vcl_vector<vcl_vector<InputImagePointer> > mem_fun_un_grid (total_grids);

  for (i=0; i<total_grids; i++) {
    //Initialize gain_field_g_grid[].
    gain_field_g_grid[i] = InputImageType::New();
    gain_field_g_grid[i]->SetRegions (img_y_grid[i]->GetLargestPossibleRegion());
    gain_field_g_grid[i]->SetSpacing (img_y_grid[i]->GetSpacing());
    gain_field_g_grid[i]->SetOrigin (img_y_grid[i]->GetOrigin());
    gain_field_g_grid[i]->Allocate();

    //Initialize the images of the membership functions u1[], u2[], u3[]
    //and a updated storage u1n[], u2n[], u3n[].
    mem_fun_u_grid[i].resize (n_class);
    mem_fun_un_grid[i].resize (n_class);
    
    for (int k = 0; k < n_class; k++) {
      mem_fun_u_grid[i][k] = InputImageType::New();
      mem_fun_u_grid[i][k]->SetRegions (img_y_grid[i]->GetLargestPossibleRegion());
      mem_fun_u_grid[i][k]->SetSpacing (img_y_grid[i]->GetSpacing());
      mem_fun_u_grid[i][k]->SetOrigin (img_y_grid[i]->GetOrigin());
      mem_fun_u_grid[i][k]->Allocate();
      mem_fun_un_grid[i][k] = InputImageType::New();
      mem_fun_un_grid[i][k]->SetRegions (img_y_grid[i]->GetLargestPossibleRegion());
      mem_fun_un_grid[i][k]->SetSpacing (img_y_grid[i]->GetSpacing());
      mem_fun_un_grid[i][k]->SetOrigin (img_y_grid[i]->GetOrigin());
      mem_fun_un_grid[i][k]->Allocate();
    }
  }

  //Grid Regression Iteration:
  bool conv = false;
  double SSD_old = 100000000; //DBL_MAX;
  double SSD;
  vcl_vector<double> SSD_history;
  int iter = 0;
  vnl_matrix<double> B;

  do {
    // vcl_printf ("\n============================================================\n");
    // vcl_printf ("\nGrid Regression Iteration %d:\n", iter);
    // vcl_printf ("\n============================================================\n");

    //Run the AFCM segmentation in each block.
    //Assume each block contains sufficient GM and WM.
    //Obtain the intensity centroids CGM and CWM for each block.
    for (i=0; i<total_grids; i++) {
      //Reset the gain field and membership functions.
      gain_field_g_grid[i]->FillBuffer (1.0f);
      for (int k = 0; k < n_class; k++) {
        mem_fun_u_grid[i][k]->FillBuffer (0.0f);
        mem_fun_un_grid[i][k]->FillBuffer (0.0f);
      }

      afcm_segmentation (img_y_grid[i], n_class, n_bin, low_th, high_th, 
                         bg_thresh, 0, 
                         gain_th, gain_min,
                         conv_thresh, gain_field_g_grid[i],
                         mem_fun_u_grid[i], mem_fun_un_grid[i], centroid_v_grid[i]);

      // vcl_printf ("\n==============================\n");
      // vcl_printf ("Iter %d Grid %d : ", iter, i);
      // vcl_printf ("  C0 %f, C1 %f, C2 %f.\n", 
      //            centroid_v_grid[i][0], centroid_v_grid[i][1], centroid_v_grid[i][2]);
      // vcl_printf ("==============================\n");
    }

    // vcl_printf ("\n============================================================\n");
    // vcl_printf ("  Start gain field fitting (regression) for iter %d.\n", iter);
    // vcl_printf ("============================================================\n");
    //Linear or quadratic regression on the WM of the nxnxn grids, 
    //with value at the center of each block.
    //WM is the centroid_v_grid[i][2].
    assert (n_class == 3);
    assert (gain_fit_option == 3 || gain_fit_option == 4);

    if (gain_fit_option == 3) {
      //Linear regression
      grid_regression_linear (centroid_v_grid, grid_center_index, B);

      //Linear fitting for the new centroid_v_grid.
      centroid_linear_fit (grid_center_index, B, centroid_vn_grid);

      ///conv = test_1st_convergence (B);
    }
    else {
      //Quadratic regression
      grid_regression_quadratic (centroid_v_grid, grid_center_index, B);

      //Quadratic fitting for the new centroid_v_grid.
      centroid_quadratic_fit (grid_center_index, B, centroid_vn_grid);

      ///conv = test_2nd_convergence (B);
    }

    //Compute difference norm of the fitting.
    SSD = compute_diff_norm (centroid_v_grid, centroid_vn_grid);
    // vcl_printf ("\n\n  Iter %d: SSD_old %4.0f, SSD %4.0f.\n", iter, SSD_old, SSD);

    //Test convergence: 
    SSD_history.push_back (SSD);
    if (SSD >= SSD_old) {
      // vcl_printf ("\n SSD > SSD_old, converges, stop iteration.\n");
      conv = true;
    }
    else {
      conv = false;
      SSD_old = SSD;
    }

    if (conv == false) {
      for (i=0; i<total_grids; i++) {
        //Update each new gain_field_g_grid[i]
        if (gain_fit_option == 3)        
          compute_linear_fit_img (B, gain_field_g_grid[i]);
        else
          compute_quadratic_fit_img (B, gain_field_g_grid[i]);
      }

      //Compute a global pixel mean to generate a new gain_field_g[].
      compute_gain_from_grids (gain_field_g_grid, img_y, bg_thresh, gain_field_g);

      for (i=0; i<total_grids; i++) {
        //Update each new img_y_grid[i] for each grid: yn[] = y[] / g[].
        update_gain_to_image (gain_field_g_grid[i], img_y_grid[i]);
      }
    }

    iter++;
  }
  while (conv == false);

  // vcl_printf ("\n==============================================================================\n");
  // vcl_printf ("  Summary for grid division gain field correction:\n");
  // vcl_printf ("    %s fitting: totally %d iteration(s).\n", 
  //            (gain_fit_option==3) ? "Linear" : "Quadratic",
  //            SSD_history.size());
  // vcl_printf ("    SSD in iterations: ");
  // for (unsigned int i=0; i<SSD_history.size(); i++)
  //   vcl_printf ("%4.0f ", SSD_history[i]);
  //   vcl_printf ("\n==============================================================================\n");
  
  //Compute the final img_y[] from the gain_field_g[].
  update_gain_to_image (gain_field_g, img_y);

  ///Debug
  ///save_img_f16 ("temp_gain_corrected.mhd", img_y);

  //mask the final gain_field with img_y[].
  mask_gain_field (img_y, bg_thresh, gain_field_g);

  //Initialize the image of gain field g[] to 1.
  InputImagePointer gain_field_tmp = InputImageType::New();  
  gain_field_tmp->CopyInformation (img_y);
  gain_field_tmp->SetRegions (img_y->GetLargestPossibleRegion());
  gain_field_tmp->Allocate();
  gain_field_tmp->FillBuffer (1.0f);

  //Run the original AFCM again on the gain corrected img_y[].
  afcm_segmentation (img_y, n_class, n_bin, low_th, high_th, bg_thresh, 
                     0, gain_th, gain_min, conv_thresh, 
                     gain_field_tmp, mem_fun_u, mem_fun_un, centroid_v);
}



template <class TInputImage, class TOutputImage>
void
FuzzyClassificationImageFilter<TInputImage, TOutputImage>
::grid_regression_linear (const vcl_vector<vcl_vector<float> >& centroid_v_grid,
                          const vcl_vector<typename InputImageType::IndexType>& grid_center_index,
                          vnl_matrix<double>& B)
{
  // vcl_printf ("    grid_regression_linear(): \n");

  //Put centroid_v_grid[2] into y[] (only use the WM for now).
  assert (centroid_v_grid[0].size() == 3);
  //Put centroid index coordinates into x1[], x2[], x3[].
  assert (centroid_v_grid.size() == grid_center_index.size());

  //Determine the total number of qualified inputs.
  // vcl_printf ("      WM centroid(s): ");
  int SZ = 0;
  for (unsigned int i=0; i<centroid_v_grid.size(); i++) {
    // vcl_printf ("%.2f ", centroid_v_grid[i][2]);
    if (centroid_v_grid[i][2] >= 0) {
      SZ++;
    }
  }
  // vcl_printf ("\n      Total # WM centroid(s) used = %d.\n", SZ);
  
  vnl_matrix<double> y (SZ,1);
  vnl_matrix<double> x1 (SZ,1);
  vnl_matrix<double> x2 (SZ,1);
  vnl_matrix<double> x3 (SZ,1);

  int c = 0;
  for (unsigned int i=0; i<centroid_v_grid.size(); i++) {
    if (centroid_v_grid[i][2] < 0) 
      continue;

    assert (c < SZ);
    y(c, 0) = centroid_v_grid[i][2]; //only use the WM for now
    int x_1 = grid_center_index[i][0];
    int x_2 = grid_center_index[i][1];
    int x_3 = grid_center_index[i][2];
    x1(c, 0) = x_1;
    x2(c, 0) = x_2;
    x3(c, 0) = x_3;
    c++;
  }

  //Prepare the design matrix X
  vnl_matrix<double> X (SZ,4);
  X.set_column (0, 1.0);
  X.update (x1, 0, 1);
  X.update (x2, 0, 2);
  X.update (x3, 0, 3);
  ///vcl_cerr << X;  
  x1.clear();
  x2.clear();
  x3.clear();

  vnl_matrix<double> Xt = X.transpose();
  vnl_matrix<double> Xt_X = Xt * X; //(x'*x)
  X.clear();
  vnl_matrix<double> Xt_y = Xt * y; //(x'*y)
  Xt.clear();
  y.clear();
  //Solve for the linear normal equation: (x'*x) * b = (x'*y)
  vnl_matrix<double> Xt_X_inv = vnl_matrix_inverse<double>(Xt_X);
  Xt_X.clear();
  //b = inv(x'*x) * (x'*y);
  B = Xt_X_inv * Xt_y;
  
  // vcl_printf ("B: \n");
  // vcl_cerr << B;
}

template <class TInputImage, class TOutputImage>
void
FuzzyClassificationImageFilter<TInputImage, TOutputImage>
::grid_regression_quadratic (const vcl_vector<vcl_vector<float> >& centroid_v_grid,
                             const vcl_vector<typename InputImageType::IndexType>& grid_center_index,
                             vnl_matrix<double>& B)
{
  // vcl_printf ("    grid_regression_quadratic(): \n");
  //Put centroid_v_grid[2] into y[] (only use the WM for now).
  assert (centroid_v_grid[0].size() == 3);
  //Put centroid index coordinates into x1[], x2[], x3[].
  assert (centroid_v_grid.size() == grid_center_index.size());
  
  //Determine the total number of qualified inputs.
  // vcl_printf ("      WM centroid(s): ");
  int SZ = 0;
  for (unsigned int i=0; i<centroid_v_grid.size(); i++) {
    // vcl_printf ("%.2f ", centroid_v_grid[i][2]);
    if (centroid_v_grid[i][2] >= 0) {
      SZ++;
    }
  }
  // vcl_printf ("\n      Total # WM centroid(s) used = %d.\n", SZ);

  vnl_matrix<double> y (SZ,1);
  vnl_matrix<double> x1 (SZ,1);
  vnl_matrix<double> x2 (SZ,1);
  vnl_matrix<double> x3 (SZ,1);

  int c = 0;
  for (unsigned int i=0; i<centroid_v_grid.size(); i++) {
    if (centroid_v_grid[i][2] < 0) 
      continue;

    assert (c < SZ);
    y(c, 0) = centroid_v_grid[i][2]; //only use the WM for now
    int x_1 = grid_center_index[i][0];
    int x_2 = grid_center_index[i][1];
    int x_3 = grid_center_index[i][2];
    x1(c, 0) = x_1;
    x2(c, 0) = x_2;
    x3(c, 0) = x_3;
    c++;
  }

  //Prepare the design matrix X
  vnl_matrix<double> X (SZ,10);
  X.set_column (0, 1.0);
  X.update (x1, 0, 1);
  X.update (x2, 0, 2);
  X.update (x3, 0, 3);
  ///vcl_cerr << X;  
  x1.clear();
  x2.clear();
  x3.clear();

  vnl_matrix<double> x1x2 (SZ,1);
  vnl_matrix<double> x1x3 (SZ,1);
  vnl_matrix<double> x2x3 (SZ,1); 
  c = 0;
  for (unsigned int i=0; i<centroid_v_grid.size(); i++) {
    if (centroid_v_grid[i][2] < 0) 
      continue;

    assert (c < SZ);
    int x_1 = grid_center_index[i][0];
    int x_2 = grid_center_index[i][1];
    int x_3 = grid_center_index[i][2];
    x1x2 (c, 0) = x_1 * x_2;
    x1x3 (c, 0) = x_1 * x_3;
    x2x3 (c, 0) = x_2 * x_3;
    c++;
  }
  X.update (x1x2, 0, 4);
  X.update (x1x3, 0, 5);
  X.update (x2x3, 0, 6);
  x1x2.clear();
  x1x3.clear();
  x2x3.clear();
  
  vnl_matrix<double> x1x1 (SZ,1);
  vnl_matrix<double> x2x2 (SZ,1);
  vnl_matrix<double> x3x3 (SZ,1);
  c = 0;
  for (unsigned int i=0; i<centroid_v_grid.size(); i++) {
    if (centroid_v_grid[i][2] < 0) 
      continue;
    assert (i < static_cast<unsigned int>(SZ));
    int x_1 = grid_center_index[i][0];
    int x_2 = grid_center_index[i][1];
    int x_3 = grid_center_index[i][2];
    x1x1 (i, 0) = x_1 * x_1;
    x2x2 (i, 0) = x_2 * x_2;
    x3x3 (i, 0) = x_3 * x_3;
    c++;
  }
  X.update (x1x1, 0, 7);
  X.update (x2x2, 0, 8);
  X.update (x3x3, 0, 9);
  x1x1.clear();
  x2x2.clear();
  x3x3.clear();

  ///// vcl_printf ("X: \n");
  ///vcl_cerr << X;

  vnl_matrix<double> Xt = X.transpose();
  vnl_matrix<double> Xt_X = Xt * X; //(x'*x)
  X.clear();
  vnl_matrix<double> Xt_y = Xt * y; //(x'*y)
  Xt.clear();
  y.clear();
  //Solve for the linear normal equation: (x'*x) * b = (x'*y)
  vnl_matrix<double> Xt_X_inv = vnl_matrix_inverse<double>(Xt_X);  
  Xt_X.clear();
  //b = inv(x'*x) * (x'*y);
  B = Xt_X_inv * Xt_y;
  
  // vcl_printf ("B: \n");
  // vcl_cerr << B;
}

//===================================================================

template <class TInputImage, class TOutputImage>
void
FuzzyClassificationImageFilter<TInputImage, TOutputImage>
::centroid_linear_fit (const vcl_vector<typename InputImageType::IndexType>& grid_center_index,  
                          const vnl_matrix<double>& B, 
                          vcl_vector<float>& centroid_vn_grid)
{  
  // vcl_printf ("    centroid_linear_fit(): \n");
  assert (B.rows() == 4);
  assert (grid_center_index.size() == centroid_vn_grid.size());

  //Compute a new value for the grid_center_index[] from B.
  for (unsigned int i=0; i<grid_center_index.size(); i++) {
    assert (grid_center_index[i].GetIndexDimension() == 3);
    int x_1 = grid_center_index[i][0];
    int x_2 = grid_center_index[i][1];
    int x_3 = grid_center_index[i][2];
    double pixel = B(0,0) + B(1,0)*x_1 + B(2,0)*x_2 + B(3,0)*x_3;
    centroid_vn_grid[i] = pixel;
  }
}

template <class TInputImage, class TOutputImage>
void
FuzzyClassificationImageFilter<TInputImage, TOutputImage>
::centroid_quadratic_fit (const vcl_vector<typename InputImageType::IndexType>& grid_center_index,  
                          const vnl_matrix<double>& B, 
                          vcl_vector<float>& centroid_vn_grid)
{
  // vcl_printf ("    centroid_quadratic_fit(): \n");
  assert (B.rows() == 10);
  assert (grid_center_index.size() == centroid_vn_grid.size());

  //Compute a new value for the grid_center_index[] from B.
  for (unsigned int i=0; i<grid_center_index.size(); i++) {
    assert (grid_center_index[i].GetIndexDimension() == 3);
    int x_1 = grid_center_index[i][0];
    int x_2 = grid_center_index[i][1];
    int x_3 = grid_center_index[i][2];
    double pixel = B(0,0) + B(1,0)*x_1 + B(2,0)*x_2 + B(3,0)*x_3 +
                   B(4,0)*x_1*x_2 + B(5,0)*x_1*x_3 + B(6,0)*x_2*x_3 +
                   B(7,0)*x_1*x_1 + B(8,0)*x_2*x_2 + B(9,0)*x_3*x_3;
    centroid_vn_grid[i] = pixel;
  }
}


template <class TInputImage, class TOutputImage>
void
FuzzyClassificationImageFilter<TInputImage, TOutputImage>
::compute_histogram (InputImagePointer& image, 
                        vcl_vector<float>& histVector,
                        vcl_vector<float>& binMax,
                        vcl_vector<float>& binMin,
                        int& nBin)
{
  // vcl_printf ("compute_histogram(): \n");

  typedef itk::Statistics::ScalarImageToListAdaptor< InputImageType > AdaptorType;
  typename AdaptorType::Pointer adaptor = AdaptorType::New();
  adaptor->SetImage (image);
  typedef typename InputImageType::PixelType  HistogramMeasurementType;
  typedef itk::Statistics::ListSampleToHistogramGenerator< 
                AdaptorType, HistogramMeasurementType> GeneratorType;
  typename GeneratorType::Pointer generator = GeneratorType::New();
  typedef typename GeneratorType::HistogramType  HistogramType;

  // let the program decide the number of bins 
  // using the maximum and minimum intensity values
  if (nBin == 0) {
    typedef itk::ImageRegionIterator< InputImageType > IteratorType;
    IteratorType it (image, image->GetLargestPossibleRegion());
    typename InputImageType::PixelType bMin = it.Get();
    typename InputImageType::PixelType bMax = it.Get();

    for ( it.GoToBegin(); !it.IsAtEnd(); ++it) {
      typename InputImageType::PixelType d = it.Get();
      if (bMin > d ) {
        bMin = d;
      }
      if (bMax < d) {
        bMax = d;
      }
    }
    nBin = static_cast<int> (bMax-bMin+1);
  }

  typename HistogramType::SizeType histogramSize;
  histogramSize.Fill (nBin);

  generator->SetListSample (adaptor);
  generator->SetNumberOfBins (histogramSize);
  generator->SetMarginalScale (10.0);
  generator->Update();

  typename HistogramType::ConstPointer histogram = generator->GetOutput();
  const unsigned int hs = histogram->Size();

  histVector.clear();
  binMax.clear();
  binMin.clear();

  ///debug: // vcl_printf ("\n");
  for (unsigned int k = 0; k < hs; k++) {
    float hist_v = histogram->GetFrequency(k, 0);
    float bin_min = histogram->GetBinMin(0, k);
    float bin_max = histogram->GetBinMax(0, k);
    binMin.push_back (bin_min);
    binMax.push_back (bin_max);
    histVector.push_back (hist_v);
    ///// vcl_printf ("h(%.1f,%.1f)=%.0f ", bin_min, bin_max, hist_v);
    ///if (k % 3 == 0)
      ///// vcl_printf ("\n");
  }
  ///// vcl_printf ("\t done.\n");
}

template <class TInputImage, class TOutputImage>
void 
FuzzyClassificationImageFilter<TInputImage, TOutputImage>
::HistogramEqualization (InputImagePointer& image)
{
  // vcl_printf ("HistogramEqualization(): \n");

  typedef itk::ImageRegionIterator< InputImageType > IteratorType;

  IteratorType it(  image, image->GetLargestPossibleRegion()  );

  typename InputImageType::PixelType bMin = it.Get();
  typename InputImageType::PixelType bMax = it.Get();

  for ( it.GoToBegin(); !it.IsAtEnd(); ++it) {
    typename InputImageType::PixelType d = it.Get();
    if (bMin > d ) {
      bMin = d;
    }
    if (bMax < d) {
      bMax = d;
    }
  }

  int nBin = static_cast<int> (bMax-bMin+1);
  vcl_vector<float> histVector;
  vcl_vector<float> binMax;
  vcl_vector<float> binMin;

  compute_histogram (image, histVector, binMax, binMin, nBin);

  vnl_vector<float> intensityMap ( nBin );
  intensityMap[0] = histVector[0];
  for ( int k = 1; k < nBin; k++ ) {
    intensityMap[k] = intensityMap[k-1] + histVector[k];
  }

  double totCount = intensityMap[nBin-1];
  for ( int k = 0; k < nBin; k++ ) {
    intensityMap[k] = 255 * intensityMap[k]/totCount;
  }

  bMax = (bMax-bMin)/(nBin-1);

  for ( it.GoToBegin(); !it.IsAtEnd(); ++it) {
    typename InputImageType::PixelType d = it.Get();
    // now bMax is the width of the bins
    int idx = (d-bMin)/bMax; 
    it.Set(intensityMap[idx]);
  }

  // vcl_printf ("\t done.\n\n");
}


template <class TInputImage, class TOutputImage>
bool
FuzzyClassificationImageFilter<TInputImage, TOutputImage>
::detect_bnd_box (InputImagePointer& image, 
                     const float bg_thresh, 
                     int& xmin, int& ymin, int& zmin, 
                     int& xmax, int& ymax, int& zmax)
{
  // vcl_printf ("    detect_bnd_box(): bg_thresh %f.\n", bg_thresh);

  typedef itk::ImageRegionIteratorWithIndex < InputImageType > IndexedIteratorType;
  IndexedIteratorType iit (image, image->GetRequestedRegion());
  assert (iit.GetIndex().GetIndexDimension() == 3);

  xmin = INT_MAX;
  ymin = INT_MAX;
  zmin = INT_MAX;
  xmax = -INT_MAX;
  ymax = -INT_MAX;
  zmax = -INT_MAX;

  for (iit.GoToBegin(); !iit.IsAtEnd(); ++iit) {
    if (iit.Get() <= bg_thresh)
      continue;

    typename InputImageType::IndexType idx = iit.GetIndex();
    int x = idx[0];
    int y = idx[1];
    int z = idx[2];
    
    if (x < xmin)
      xmin = x;
    if (y < ymin)
      ymin = y;
    if (z < zmin)
      zmin = z;
    if (x > xmax)
      xmax = x;
    if (y > ymax)
      ymax = y;
    if (z > zmax)
      zmax = z;
  }

  if (xmin == INT_MAX || ymin == INT_MAX || zmin == INT_MAX ||
      xmax == INT_MIN || ymax == INT_MIN || zmax == INT_MIN) {
    //error: no pixel with intensity > bf_thresh.
    // vcl_printf ("Error: no pixel with intensity > bf_thresh!\n");
    return false;
  }

  // vcl_printf ("      (%d, %d, %d) - (%d, %d, %d).\n\n",
  //            xmin, ymin, zmin, xmax, ymax, zmax);
  return true;
}

template <class TInputImage, class TOutputImage>
void 
FuzzyClassificationImageFilter<TInputImage, TOutputImage>
::compute_grid_imgs (InputImagePointer& image, 
                        const int xmin, const int ymin, const int zmin, 
                        const int xmax, const int ymax, const int zmax, 
                        const int n_grid, 
                        vcl_vector<InputImagePointer>& image_grid,
                        vcl_vector<typename InputImageType::IndexType>& grid_center_index)
{
  // vcl_printf ("    compute_grid_imgs(): %d * %d * %d grids.\n", 
  //            n_grid, n_grid, n_grid);

  const int total_grids = n_grid * n_grid * n_grid;

  const int total_size_x = xmax-xmin+1;
  const int total_size_y = ymax-ymin+1;
  const int total_size_z = zmax-zmin+1;
  // vcl_printf ("      size of region in interest: %d * %d * %d.\n", 
  //            total_size_x, total_size_y, total_size_z);

  const int grid_size_x = static_cast<int>( vcl_ceil (double(total_size_x) / n_grid) );
  const int grid_size_y = static_cast<int>( vcl_ceil (double(total_size_y) / n_grid) );
  const int grid_size_z = static_cast<int>( vcl_ceil (double(total_size_z) / n_grid) );
  // vcl_printf ("      regular grid size: %d * %d * %d.\n", 
  //            grid_size_x, grid_size_y, grid_size_z);

  const int grid_size_last_x = total_size_x - grid_size_x * (n_grid-1);
  const int grid_size_last_y = total_size_y - grid_size_y * (n_grid-1);
  const int grid_size_last_z = total_size_z - grid_size_z * (n_grid-1);
  // vcl_printf ("      last slice, column, row grid size: %d * %d * %d.\n", 
  //            grid_size_last_x, grid_size_last_y, grid_size_last_z);
    
  typename InputImageType::RegionType region;
  typename InputImageType::IndexType index;
  typename InputImageType::SizeType  size;

  image_grid.resize (total_grids);
  int i = 0;
  for (int x=0; x<n_grid; x++) {
    int start_x = grid_size_x * x + xmin;
    int grid_x = grid_size_x;
    if (x==n_grid-1) //last row
      grid_x = grid_size_last_x;    

    for (int y=0; y<n_grid; y++) {
      int start_y = grid_size_y * y + ymin;
      int grid_y = grid_size_y;
      if (y==n_grid-1) //last column
        grid_y = grid_size_last_y;

      for (int z=0; z<n_grid; z++) {
        int start_z = grid_size_z * z + zmin;
        int grid_z = grid_size_z;
        if (z==n_grid-1) //last slice
          grid_z = grid_size_last_z;

        //create image_grid[i] with size grid_x * grid_y * grid_z.
        image_grid[i] = InputImageType::New();

        //setup grid index
        index[0] = start_x; // first index on X
        index[1] = start_y; // first index on Y
        index[2] = start_z; // first index on Z

        //setup grid size
        size[0] = grid_x;
        size[1] = grid_y;
        size[2] = grid_z;

        //setup grid center
        typename InputImageType::IndexType grid_center;
        grid_center[0] = start_x + grid_x / 2;
        grid_center[1] = start_y + grid_y / 2;
        grid_center[2] = start_z + grid_z / 2;
        grid_center_index.push_back (grid_center);

        region.SetIndex (index);
        region.SetSize (size);
        image_grid[i]->SetRegions (region);

        image_grid[i]->SetOrigin (image->GetOrigin());        
        image_grid[i]->SetSpacing (image->GetSpacing());
        image_grid[i]->Allocate();

        //copy image pixel values into image_grid[i].
        typedef itk::ImageRegionConstIterator < InputImageType > ConstIteratorType;
        typedef itk::ImageRegionIterator < InputImageType > IteratorType;
        ConstIteratorType it (image, region);
        IteratorType itg (image_grid[i], image_grid[i]->GetRequestedRegion());

        float max_pixel = -itk::NumericTraits<float>::min();
        for (it.GoToBegin(), itg.GoToBegin(); !it.IsAtEnd(); ++it, ++itg) {
          //debug:
          //ImageType::IndexType idx = it.GetIndex();
          //int itx = idx[0];
          //int ity = idx[1];
          //int itz = idx[2];
          //ImageType::IndexType idxg = itg.GetIndex();
          //int itgx = idxg[0];
          //int itgy = idxg[1];
          //int itgz = idxg[2];

          typename InputImageType::PixelType pixel = it.Get();
          itg.Set (pixel);
          if (pixel > max_pixel)
            max_pixel = pixel;
        }

        // vcl_printf ("      grid %d [%d * %d * %d] max_pixel %f, center (%d, %d, %d).\n", 
        //            i, grid_x, grid_y, grid_z, max_pixel,
        //            grid_center[0], grid_center[1], grid_center[2]);

        ///debug: write image_grid[i] to file for further examination.
        ///save_img8 ("grid.mhd", image_grid[i]);

        i++;
      }
    }
  }

}

template <class TInputImage, class TOutputImage>
void 
FuzzyClassificationImageFilter<TInputImage, TOutputImage>
::compute_gain_from_grids (const vcl_vector<InputImagePointer>& gain_field_g_grid, 
                              InputImagePointer& img_y, const float bg_thresh,
                              InputImagePointer& gain_field_g)
{
  //Given a set of fitting images, compute a new gain field image.
  // vcl_printf ("    compute_gain_from_grids(): \n");

  typedef itk::ImageRegionIterator < InputImageType > IteratorType;
  typedef itk::ImageRegionIteratorWithIndex < InputImageType > IndexIteratorType;  

  //Compute the global mean pixel value.
  //Use only non-background pixels!!
  IteratorType yit (img_y, img_y->GetRequestedRegion());
  double sum = 0;
  int count = 0;
  for (unsigned int i=0; i<gain_field_g_grid.size(); i++) {
    IndexIteratorType iit (gain_field_g_grid[i], gain_field_g_grid[i]->GetRequestedRegion());
    for (iit.GoToBegin(); !iit.IsAtEnd(); ++iit) {
      //Skip if this pixel is in background.
      typename InputImageType::IndexType idx = iit.GetIndex();
      yit.SetIndex (idx);
      if (yit.Get() <= bg_thresh)
        continue;

      double pixel = iit.Get();
      sum += pixel;
      count++;
    }
  }    
  double mean = sum / count;
  // vcl_printf ("      %d non-background pixels, intensity mean = %f\n", count, mean);

  
  IteratorType git (gain_field_g, gain_field_g->GetRequestedRegion());
  float max = -itk::NumericTraits<float>::min();
  float min = itk::NumericTraits<float>::min();

  for (unsigned int i=0; i<gain_field_g_grid.size(); i++) {
    IndexIteratorType iit (gain_field_g_grid[i], gain_field_g_grid[i]->GetRequestedRegion());
    for (iit.GoToBegin(); !iit.IsAtEnd(); ++iit) {
      double pixel = iit.Get();
      //gain_field = pixel / mean.
      pixel = pixel / mean;
      iit.Set (pixel);

      //update gain_field_g[]
      typename InputImageType::IndexType idx = iit.GetIndex();
      git.SetIndex (idx);
      double value = git.Get();      
      //Update global gain field 
      //gain_field_g(x,y,z) = gain_field_g(x,y,z) * gain_field_g_grid(x,y,z)
      value *= pixel; 
      git.Set (value);
      if (value > max)
        max = value;
      if (value < min)
        min = value;
    }
  }

  // vcl_printf ("      min / max of g[]: %f / %f.\n", min, max);
}

template <class TInputImage, class TOutputImage>
void FuzzyClassificationImageFilter<TInputImage, TOutputImage>
::update_gain_to_image (InputImagePointer& gain_field, 
                           InputImagePointer& image)
{
  // vcl_printf ("    update_gain_to_image():\n");

  //Both use gain_field's region.
  typedef itk::ImageRegionConstIteratorWithIndex < InputImageType > ConstIndexIteratorType;
  typedef itk::ImageRegionIterator < InputImageType > IteratorType;
  ConstIndexIteratorType itg (gain_field, gain_field->GetRequestedRegion());
  IteratorType it (image, gain_field->GetRequestedRegion());

  for (itg.GoToBegin(), it.GoToBegin(); !itg.IsAtEnd(); ++itg, ++it) {
    typename InputImageType::PixelType gain = itg.Get();
    typename InputImageType::PixelType pixel = it.Get();

    ///Debug
    ///ImageType::IndexType idxg = itg.GetIndex();
    ///if (idxg[0]==97 && idxg[1]==87 && idxg[2]==33) {
    ///  // vcl_printf ("\n  pixel = %f, gain (%d, %d, %d) = %f.", 
    ///              pixel, idxg[0], idxg[1], idxg[2], gain);
    ///}

    pixel = pixel / gain;

    it.Set (pixel);
  }
}

template <class TInputImage, class TOutputImage>
double
FuzzyClassificationImageFilter<TInputImage, TOutputImage>
::compute_diff_norm (const vcl_vector<vcl_vector<float> >& centroid_v_grid, 
                          const vcl_vector<float>& centroid_vn_grid)
{
  // vcl_printf ("    compute_diff_norm(): \n");

  double SSD = 0;
  assert (centroid_v_grid.size() == centroid_vn_grid.size());
  for (unsigned int i=0; i<centroid_v_grid.size(); i++)
    {
    double diff = centroid_v_grid[i][2] - centroid_vn_grid[i];
    SSD += (diff * diff);
  }
  return SSD;
}

//mask the final gain_field with image and bg_thresh.
template <class TInputImage, class TOutputImage>
void 
FuzzyClassificationImageFilter<TInputImage, TOutputImage>
::mask_gain_field (InputImagePointer& image, 
                      const float bg_thresh,
                      InputImagePointer& gain_field_g)
{
  typedef itk::ImageRegionConstIterator < InputImageType > ConstIteratorType;
  typedef itk::ImageRegionIterator < InputImageType > IteratorType;
  //both use gain_field's region.
  ConstIteratorType it (image, image->GetRequestedRegion());
  IteratorType itg (gain_field_g, image->GetRequestedRegion());
  
  for (it.GoToBegin(), itg.GoToBegin(); !it.IsAtEnd(); ++it, ++itg) {
    typename InputImageType::PixelType pixel = it.Get();    
    if (pixel <= bg_thresh)
      itg.Set (0);
  }
}

} // end namespace itk

#endif
