PROJECT(SkullStripper)

cmake_minimum_required(VERSION 2.4)

include_directories( ${SlicerBaseCLI_BINARY_DIR} ${SlicerBaseCLI_SOURCE_DIR} )

link_directories(${ModuleDescriptionParser_BINARY_DIR})
link_directories(${SlicerBaseCLI_BINARY_DIR})

if(NOT Slicer3_SOURCE_DIR)
  find_package(Slicer3 REQUIRED)
  include(${Slicer3_USE_FILE})
  slicer3_set_default_install_prefix_for_external_projects()
endif(NOT Slicer3_SOURCE_DIR)


################################################
SET (CLP SkullStripper)

set ( ${CLP}_SOURCE ${CLP}.cxx)
generateclp(${CLP}_SOURCE ${CLP}.xml ${Slicer3_DIR}../Slicer3/Resources/ITKLogo.h)
add_executable(${CLP} ${${CLP}_SOURCE})
target_link_libraries (${CLP} ITKCommon ITKNumerics ITKNrrdIO ITKIO ITKStatistics vtkIO vtkCommon vtkFiltering vtkGraphics vtkRendering)

