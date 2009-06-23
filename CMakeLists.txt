PROJECT(SkullStripper)

cmake_minimum_required(VERSION 2.4)

find_package(GenerateCLP REQUIRED)
if(GenerateCLP_FOUND)
  include(${GenerateCLP_USE_FILE})
endif(GenerateCLP_FOUND)
find_package(vtkITK REQUIRED)

link_directories(${ModuleDescriptionParser_BINARY_DIR})

include_directories( ${SlicerBaseCLI_BINARY_DIR} ${SlicerBaseCLI_SOURCE_DIR} )
link_directories(${SlicerBaseCLI_BINARY_DIR})

FIND_PACKAGE(ITK)
IF(ITK_FOUND)
  INCLUDE(${ITK_USE_FILE})
ELSE(ITK_FOUND)
  MESSAGE(FATAL_ERROR
    "ITK not found.  Please set ITK_DIR.")
ENDIF(ITK_FOUND)


FIND_PACKAGE(VTK)
IF(VTK_FOUND)
  INCLUDE(${VTK_USE_FILE})
ELSE(VTK_FOUND)
  MESSAGE(FATAL_ERROR
            "VTK not found.  Please set VTK_DIR.")
ENDIF(VTK_FOUND)


################################################
SET (CLP SkullStripper)

set ( ${CLP}_SOURCE ${CLP}.cxx)
generateclp(${CLP}_SOURCE ${CLP}.xml ${Slicer3_SOURCE_DIR}/Resources/ITKLogo.h)
add_executable(${CLP} ${${CLP}_SOURCE})
target_link_libraries (${CLP} ITKCommon ITKNumerics ITKNrrdIO ITKIO ITKStatistics vtkIO vtkCommon vtkFiltering vtkGraphics vtkRendering)

