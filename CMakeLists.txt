cmake_minimum_required(VERSION 3.10.2)

project(SkullStripper)

#-----------------------------------------------------------------------------
if(NOT Slicer_SOURCE_DIR)
  set(EXTENSION_NAME SkullStripper)
  set(EXTENSION_HOMEPAGE "https://www.slicer.org/slicerWiki/index.php/Documentation/4.1/Modules/SkullStripper")
  set(EXTENSION_CATEGORY "Segmentation")
  set(EXTENSION_CONTRIBUTORS "Xiaodong Tao (GE), Jean-Christophe Fillion-Robin (Kitware)")
  set(EXTENSION_DESCRIPTION "Skull stripping for structural MR images of the brain, tested on T1 and T2 contrast.")
  set(EXTENSION_ICONURL "https://www.slicer.org/slicerWiki/images/f/f2/SkullStripper.png")
  set(EXTENSION_SCREENSHOTURLS "https://wiki.slicer.org/slicerWiki/images/a/ab/Slicer-r19441-CLIExtensionTemplate-screenshot.png https://wiki.slicer.org/slicerWiki/images/1/1e/Slicer-r19441-CLIExtensionTemplate-screenshot-2.png")
endif()

#-----------------------------------------------------------------------------
set(MODULE_NAME SkullStripper)

#-----------------------------------------------------------------------------
if(NOT Slicer_SOURCE_DIR)
  find_package(Slicer REQUIRED)
  include(${Slicer_USE_FILE})
endif()

#-----------------------------------------------------------------------------
set(MODULE_TARGET_LIBRARIES
  ${ITK_LIBRARIES}
  ${VTK_LIBRARIES}
  )

#-----------------------------------------------------------------------------
SEMMacroBuildCLI(
  NAME ${MODULE_NAME}
  TARGET_LIBRARIES ${MODULE_TARGET_LIBRARIES}
  )

#-----------------------------------------------------------------------------
if(BUILD_TESTING)
  add_subdirectory(Testing)
endif()

#-----------------------------------------------------------------------------
if(NOT Slicer_SOURCE_DIR)
  include(${Slicer_EXTENSION_CPACK})
endif()
