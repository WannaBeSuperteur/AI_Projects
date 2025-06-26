
from generate_ombre_images import generate_ombre_image_using_v7_all_process, generate_ombre_image_using_v8_all_process
from generate_gif import generate_gif
from generate_opencv_screen import create_opencv_screen


if __name__ == '__main__':
    generate_ombre_image_using_v7_all_process()
    generate_ombre_image_using_v8_all_process()

    v7_test_ohlora_nos = [672, 1277, 1836, 1918, 2137]
    v8_test_ohlora_nos = [83, 194, 1180, 1313, 1996]

    # GIF 생성 테스트
    for v7_test_ohlora_no in v7_test_ohlora_nos:
        generate_gif(vectorfind_ver='v7', ohlora_no=v7_test_ohlora_no)

    for v8_test_ohlora_no in v8_test_ohlora_nos:
        generate_gif(vectorfind_ver='v8', ohlora_no=v8_test_ohlora_no)

    # OpenCV 움직이는 화면 생성 테스트
    for v7_test_ohlora_no in v7_test_ohlora_nos:
        create_opencv_screen(vectorfind_ver='v7', ohlora_no=v7_test_ohlora_no)

    for v8_test_ohlora_no in v8_test_ohlora_nos:
        create_opencv_screen(vectorfind_ver='v8', ohlora_no=v8_test_ohlora_no)
