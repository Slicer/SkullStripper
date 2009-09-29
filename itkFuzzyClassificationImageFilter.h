/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkFuzzyClassificationImageFilter.h,v $
  Language:  C++
  Date:      $Date: 2009/06/29 21:35:08 $
  Version:   $Revision: 1.1 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

  Portions of this code are covered under the VTK copyright.
  See VTKCopyright.txt or http://www.kitware.com/VTKCopyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkFuzzyClassificationImageFilter_h
#define __itkFuzzyClassificationImageFilter_h

#include "limits.h"
#include "vcl_cstdio.h"
#include "vcl_vector.h"
#include "itkImageToImageFilter.h"
#include "itkConceptChecking.h"
#include "itkScalarImageToListAdaptor.h"

namespace itk
{

/** \class FuzzyClassificationImageFilter
 * \brief Set image values to a user-specified value if they are below, 
 * above, or between simple threshold values.
 *
 * FuzzyClassificationImageFilter sets image values to a user-specified "outside"
 * value (by default, "black") if the image values are below, above, or
 * between simple threshold values. 
 *
 * The pixels must support the operators >= and <=.
 * 
 * \ingroup IntensityImageFilters Multithreaded
 */
template <class TInputImage, class TOutputImage>
class ITK_EXPORT FuzzyClassificationImageFilter : public ImageToImageFilter<TInputImage,TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef FuzzyClassificationImageFilter                Self;
  typedef ImageToImageFilter<TInputImage,TOutputImage>  Superclass;
  typedef SmartPointer<Self>                            Pointer;
  typedef SmartPointer<const Self>                      ConstPointer;

  /** Some additional typedefs.  */
  typedef TInputImage                           InputImageType;
  typedef typename InputImageType::ConstPointer InputImageConstPointer;
  typedef typename InputImageType::Pointer      InputImagePointer;
  typedef typename InputImageType::RegionType   InputImageRegionType;
  typedef typename InputImageType::PixelType    InputImagePixelType;

  /** Some additional typedefs.  */
  typedef TOutputImage                          OutputImageType;
  typedef typename OutputImageType::Pointer     OutputImagePointer;
  typedef typename OutputImageType::RegionType  OutputImageRegionType;
  typedef typename OutputImageType::PixelType   OutputImagePixelType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);  

  /** Run-time type information (and related methods). */
  itkTypeMacro(FuzzyClassificationImageFilter, ImageToImageFilter);

  /** Typedef to describe the type of pixel. */
  typedef typename TInputImage::PixelType PixelType;
    
  /** Set/Get methods for number of classes */
  void SetNumberOfClasses( int n )
  {
    this->m_NumberOfClasses = n;
    this->SetNumberOfOutputs( n );
    return;
  }

  itkGetMacro(NumberOfClasses, int);

  const std::vector<float>& GetClassCentroid()
  {
    return this->m_ClassCentroid;
  }

  const std::vector<float>& GetClassStandardDeviation()
  {
    return this->m_ClassStandardDeviation;
  }

  /** Set/Get methods for bias correction option */
  itkSetMacro(BiasCorrectionOption, int);
  itkGetMacro(BiasCorrectionOption, int);

  /** Set/Get the Bias field. */
  itkSetObjectMacro( BiasField,  OutputImageType );
  itkGetObjectMacro( BiasField,  OutputImageType );

  /** Set/Get the Image Mask. */
  itkSetObjectMacro( ImageMask,  InputImageType );

  virtual void GenerateData();

protected:
  FuzzyClassificationImageFilter();
  ~FuzzyClassificationImageFilter() {};
  void PrintSelf(std::ostream& os, Indent indent) const;

private:
  FuzzyClassificationImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  int m_NumberOfClasses;
  int m_BiasCorrectionOption;
  OutputImagePointer m_BiasField;
  InputImagePointer  m_ImageMask;
  std::vector<float> m_ClassCentroid;
  std::vector<float> m_ClassStandardDeviation;



private:

void afcm_segmentation (InputImagePointer img_y, 
                        const int n_class, const int n_bin,
                        const float low_th, const float high_th,
                        const float bg_thresh,
                        const int gain_fit_option, 
                        const float gain_th, const float gain_min,
                        const float conv_thresh,
                        InputImagePointer& gain_field_g,
                        vcl_vector<InputImagePointer>& mem_fun_u, 
                        vcl_vector<InputImagePointer>& mem_fun_un, 
                        vcl_vector<float>& centroid_v);

void compute_init_centroid (InputImagePointer image, 
                            const int nClass, const int nBin,
                            const float lowThreshold,
                            const float highThreshold,
                            vcl_vector<float>& initCentroid);

// Compute new membership functions u1[], u2[], u3[].
void compute_new_mem_fun_u (const vcl_vector<float>& centroid_v,
                            InputImagePointer gain_field_g, 
                            InputImagePointer img_y,
                            const float bg_thresh,
                            vcl_vector<InputImagePointer>& mem_fun_u);

// Compute the new centroids v1, v2, v3.
void compute_new_centroids (const vcl_vector<InputImagePointer>& mem_fun_u, 
                            InputImagePointer& gain_field_g, 
                            InputImagePointer& img_y, 
                            vcl_vector<float>& centroid_v);

// Compute a new gain field g[].
void compute_new_gain_field (vcl_vector<InputImagePointer>& mem_fun_u, 
                             InputImagePointer& img_y, 
                             InputImagePointer& gain_field_g,
                             const int option, const float gain_th);

// Test convergence.
bool test_convergence (const vcl_vector<InputImagePointer>& mem_fun_u, 
                       const vcl_vector<InputImagePointer>& mem_fun_un, 
                       const float conv_thresh);

int CountMode (const vcl_vector<float>& v);

void img_regression_linear (InputImagePointer& image,
                            const float thresh,
                            vnl_matrix<double>& B);

//Use B to compute a new fitting
void compute_linear_fit_img (const vnl_matrix<double>& B, 
                             InputImagePointer& fit_image);

void img_regression_quadratic (InputImagePointer& image,
                               const float thresh,
                               vnl_matrix<double>& B);

//Use B to compute a new fitting
void compute_quadratic_fit_img (const vnl_matrix<double>& B, 
                                InputImagePointer& fit_image);

void afcm_segmentation_grid (InputImagePointer img_y, 
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
                             vcl_vector<float>& centroid_v);


void grid_regression_linear (const vcl_vector<vcl_vector<float> >& centroid_v_grid,
                             const vcl_vector<typename InputImageType::IndexType>& grid_center_index,
                             vnl_matrix<double>& B);

void grid_regression_quadratic (const vcl_vector<vcl_vector<float> >& centroid_v_grid,
                                const vcl_vector<typename InputImageType::IndexType>& grid_center_index,
                                vnl_matrix<double>& B);

//===================================================================

void centroid_linear_fit (const vcl_vector<typename InputImageType::IndexType>& grid_center_index,  
                          const vnl_matrix<double>& B, 
                          vcl_vector<float>& centroid_vn_grid);

void centroid_quadratic_fit (const vcl_vector<typename InputImageType::IndexType>& grid_center_index,  
                             const vnl_matrix<double>& B, 
                             vcl_vector<float>& centroid_vn_grid);

void compute_histogram (InputImagePointer& image, 
                        vcl_vector<float>& histVector,
                        vcl_vector<float>& binMax,
                        vcl_vector<float>& binMin,
                        int& nBin);

void HistogramEqualization (InputImagePointer& image);

bool detect_bnd_box (InputImagePointer& image, 
                     const float bg_thresh, 
                     int& xmin, int& ymin, int& zmin, 
                     int& xmax, int& ymax, int& zmax);

void compute_grid_imgs (InputImagePointer& image, 
                        const int xmin, const int ymin, const int zmin, 
                        const int xmax, const int ymax, const int zmax, 
                        const int n_grid, 
                        vcl_vector<InputImagePointer>& image_grid,
                        vcl_vector<typename InputImageType::IndexType>& grid_center_index);

void compute_gain_from_grids (const vcl_vector<InputImagePointer>& gain_field_g_grid, 
                              InputImagePointer& img_y, const float bg_thresh,
                              InputImagePointer& gain_field_g);

void update_gain_to_image (InputImagePointer& gain_field, 
                           InputImagePointer& image);

double compute_diff_norm (const vcl_vector<vcl_vector<float> >& centroid_v_grid, 
                          const vcl_vector<float>& centroid_vn_grid);

//mask the final gain_field with image and bg_thresh.
void mask_gain_field (InputImagePointer& image, 
                      const float bg_thresh,
                      InputImagePointer& gain_field_g);

};

  
} // end namespace itk
  
#ifndef ITK_MANUAL_INSTANTIATION
#include "itkFuzzyClassificationImageFilter.txx"
#endif
  
#endif
