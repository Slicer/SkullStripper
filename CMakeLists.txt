cmake_minimum_required(VERSION 2.8.7)

#-----------------------------------------------------------------------------
set(MODULE_NAME SkullStripper) # Do not use 'project()'

#-----------------------------------------------------------------------------
set(MODULE_HOMEPAGE "http://www.slicer.org/slicerWiki/index.php/Documentation/4.1/Modules/${MODULE_NAME}")
set(MODULE_CATEGORY "Segmentation")
set(MODULE_ICONURL "http://www.slicer.org/slicerWiki/images/f/f2/SkullStripper.png")
set(MODULE_STATUS "Beta")
set(MODULE_CONTRIBUTORS "Xiaodong Tao (GE)")
set(MODULE_DESCRIPTION "Skull stripping for structural MR images of the brain, tested on T1 and T2 contrast.")
set(MODULE_SCREENSHOTURLS "http://wiki.slicer.org/slicerWiki/images/a/ab/Slicer-r19441-CLIExtensionTemplate-screenshot.png http://wiki.slicer.org/slicerWiki/images/1/1e/Slicer-r19441-CLIExtensionTemplate-screenshot-2.png")
set(MODULE_ACKNOWLEDGEMENTS "This work is part of the National Alliance for Medical Image Computing (NAMIC), funded by the National Institutes of Health through the NIH Roadmap for Medical Research, Grant U54 EB005149.")
set(MODULE_MAJOR_VERSION 0)
set(MODULE_MINOR_VERSION 1)
set(MODULE_PATCH_VERSION 0)

#-----------------------------------------------------------------------------
if(NOT Slicer_SOURCE_DIR)
  find_package(Slicer REQUIRED)
  include(${Slicer_USE_FILE})
endif()

# Configure XML description
configure_file(
  ${MODULE_NAME}.xml.in
  ${CMAKE_CURRENT_BINARY_DIR}/${MODULE_NAME}.xml
  @ONLY
  )

#-----------------------------------------------------------------------------
set(cli_module_target_libraries
  ${ITK_LIBRARIES}
  ${VTK_LIBRARIES}
  )

#-----------------------------------------------------------------------------
SEMMacroBuildCLI(
  NAME ${MODULE_NAME}
  TARGET_LIBRARIES ${cli_module_target_libraries}
  )

#-----------------------------------------------------------------------------
if(BUILD_TESTING)
  add_subdirectory(Testing)
endif()

#-----------------------------------------------------------------------------
if(NOT Slicer_SOURCE_DIR)
  include(${Slicer_EXTENSION_CPACK})
endif()
