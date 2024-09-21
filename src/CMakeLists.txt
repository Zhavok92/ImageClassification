target_include_directories(ImageRecognition PRIVATE .)
target_sources(ImageRecognition
PRIVATE
    main.cpp
    ImageRecognition.cpp
)

target_link_libraries(ImageRecognition ${TORCH_LIBRARIES} ${OpenCV_LIBS} tensorboard_logger)
set_target_properties(ImageRecognition
PROPERTIES
    CXX_STANDARD 20
    COMPILE_OPTIONS "/wd4267"
)