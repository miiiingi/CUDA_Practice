cmake_minimum_required(VERSION 2.6)
project(vectorSummation CUDA CXX)  # CUDA 언어 지원 추가

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
# CUDA 아키텍처 설정 (예: 7.5, 8.0 등)
set(CMAKE_CUDA_ARCHITECTURES 87)

# CUDA 설정
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc")

find_package(CUDA REQUIRED)  # CUDA 패키지 찾기

include_directories(
    ${CUDA_INCLUDE_DIRS}  # CUDA include 디렉토리 추가
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/DS_Timer  # DS_Timer 디렉토리 추가
    ${PROJECT_SOURCE_DIR}/LargeMatrixCalculation  # DS_Timer 디렉토리 추가
)

link_directories(
    ${CUDA_LIBRARY_DIRS}  # CUDA 라이브러리 디렉토리 추가
)

#vectorSummation Code
add_executable(vectorSummation vectorSummation.cu DS_Timer/DS_timer.cpp)
set_target_properties(vectorSummation PROPERTIES CUDA_SEPARABLE_COMPILATION ON)  # CUDA 분리 컴파일 활성화

target_link_libraries(vectorSummation 
    ${CUDA_LIBRARIES} 
)

#checkLayout Code
add_executable(checkLayout checkLayout.cu)
set_target_properties(checkLayout PROPERTIES CUDA_SEPARABLE_COMPILATION ON)  # CUDA 분리 컴파일 활성화

target_link_libraries(checkLayout 
    ${CUDA_LIBRARIES} 
)

#Large Mat Add Code
add_executable(largeMatAdd LargeMatrixCalculation/matadd.cpp LargeMatrixCalculation/matadd.cu LargeMatrixCalculation/matadd.cuh DS_Timer/DS_timer.cpp)
set_target_properties(largeMatAdd PROPERTIES CUDA_SEPARABLE_COMPILATION ON)  # CUDA 분리 컴파일 활성화

target_link_libraries(largeMatAdd 
    ${CUDA_LIBRARIES} 
)
