PROJECT(SkullStripper)

cmake_minimum_required(VERSION 2.4)

include_directories( ${SlicerBaseCLI_BINARY_DIR} ${SlicerBaseCLI_SOURCE_DIR} )

link_directories(${ModuleDescriptionParser_BINARY_DIR})
link_directories(${SlicerBaseCLI_BINARY_DIR})

FIND_PACKAGE(Slicer3 REQUIRED NO_DEFAULT_PATH)

IF (Slicer3_FOUND)
  include(${Slicer3_USE_FILE})
  slicer3_set_default_install_prefix_for_external_projects()

  SET (CLP SkullStripper)

  set ( ${CLP}_SOURCE ${CLP}.cxx)
  generateclp(${CLP}_SOURCE ${CLP}.xml)
  add_executable(${CLP} ${${CLP}_SOURCE})
  target_link_libraries (${CLP} ITKCommon ITKNumerics ITKNrrdIO ITKIO ITKStatistics vtkIO vtkCommon vtkFiltering vtkGraphics vtkRendering)

  slicer3_set_plugins_output_path(SkullStripper)
  slicer3_install_plugins(SkullStripper)

ELSE (Slicer3_FOUND)

MESSAGE(WARNING "Cannot build with a Slicer3 build tree or a Slicer3 installation. Please set Slicer3_DIR. If using a Slicer3 build, Slicer3_DIR needs to be set to the root of the build tree of Slicer3.  If using an installation of Slicer3, Slicer3_DIR needs to be set to <SlicerInstallation>/lib/Slicer3.")

ENDIF (Slicer3_FOUND)

################################################

