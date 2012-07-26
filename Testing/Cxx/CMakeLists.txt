
#-----------------------------------------------------------------------------
set(TEMP ${SkullStripper_BINARY_DIR}/Testing/Temporary)
set(INPUT ${SkullStripper_SOURCE_DIR}/Data/Input)
set(BASELINE ${SkullStripper_SOURCE_DIR}/Data/Baseline)
set(CLP SkullStripper)

#-----------------------------------------------------------------------------
add_executable(${CLP}Test ${CLP}Test.cxx)
target_link_libraries(${CLP}Test ${CLP}Lib)
set_target_properties(${CLP}Test PROPERTIES LABELS ${CLP})

#-----------------------------------------------------------------------------
set(testname ${CLP}Test)
ExternalData_add_test(${CLP}Data NAME ${testname} COMMAND ${Slicer_LAUNCH_COMMAND} $<TARGET_FILE:${CLP}Test>
  --compare DATA{${BASELINE}/SkullStripperTest.nrrd}
            ${TEMP}/SkullStripperTest.nrrd
  ModuleEntryPoint
   DATA{${INPUT}/MR-head.nrrd}
   ${TEMP}/SkullStripperTest.vtkp
   ${TEMP}/SkullStripperTest.nrrd
  )
set_property(TEST ${testname} PROPERTY LABELS ${CLP})

#-----------------------------------------------------------------------------
ExternalData_add_target(${CLP}Data)
